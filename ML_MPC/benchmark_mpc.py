"""
benchmark_mpc.py  (versão corrigida)

Benchmark LMPC vs. ML-MPC Híbrido durante uma transição de startup severa
(onde a não-linearidade de Arrhenius é acentuada).

CORREÇÕES APLICADAS
-------------------
1. Planta trocada de `plant.RealCSTR` (2 estados, step(Qsig, dt)) para
   `model.CSTR` (5 estados: T, CA, CB, CC, V), que é a interface realmente
   esperada pelos controladores deste benchmark. Isso elimina o erro
   `AttributeError: 'RealCSTR' object has no attribute 'Temperature'`.

2. O sinal split-range Qsig (0-100 %) produzido pelo MPC é convertido para
   potência térmica em Watts (`qsig_to_watts`) antes de entrar em
   `CSTR.step(..., Q_heating=...)`. Sem isso havia descasamento de ~5 ordens
   de grandeza (o MPC pensa em %, a planta espera W) e a temperatura divergia.

3. Coeficientes UA reduzidos de ~3,3e5 para ~3,0e3 W/K (fisicamente coerentes
   para um reator de ~1 m³) e passo de integração dt=0,05 s, evitando o
   runaway térmico do Euler explícito.

4. Execução principal protegida por `if __name__ == "__main__":` para permitir
   `from benchmark_mpc_fixed import LMPC` sem disparar a simulação no import.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Planta multi-estados compatível com os controladores (5 estados)
from model import CSTR
from mpc import HybridNMPC
from nn_forward_numpy import make_nn_forward

class LMPC:
    """
    Controlador Preditivo Linear (LMPC).
    Utiliza um modelo linearizado (Matrizes A e B) ao redor de um 
    ponto de operação fixo (x_op, u_op).
    """
    def __init__(self, x_op, u_op, dt_pred=2.0, N=12, Q=5.0, R=0.5, nominal_params=None, plant_params=None):
        self.N = N
        self.Q = Q
        self.R = R
        self.dt = dt_pred
        self.params = nominal_params
        self.plant_params = plant_params
        
        self.x_op = np.array(x_op)
        self.u_op = u_op
        
        self.u_min, self.u_max = 0.0, 100.0
        self.dU_max = 20.0
        self.u_seq_prev = np.ones(N) * u_op
        
        # 1. Linearização da Planta (Calcula Matrizes A e B numéricamente)
        self.A, self.B = self._linearize(self.x_op, self.u_op)

    def _nominal_dynamics(self, x, u_qsig):
        """Modelo Fenomenológico Nominal (Sem Rede Neural)"""
        T, CA, CB, CC, V = x
        k1_nom = self.params['A1'] * np.exp(-self.params['E1'] / (8.314 * T))
        k2_nom = self.params['A2'] * np.exp(-self.params['E2'] / (8.314 * T))
        r1 = k1_nom * CA * CB
        r2 = k2_nom * CA * CC
        
        if u_qsig <= 50:
            Tj = 25.0 + (u_qsig / 50.0) * 20.0
            UA = self.plant_params['UA_cool']
        else:
            Tj = 50.0 + ((u_qsig - 50) / 50.0) * 90.0
            UA = self.plant_params['UA_heat']
            
        Fin = self.plant_params['F_in']
        Fout = Fin
        dVdt = Fin - Fout
        
        if V > 1e-6:
            dCadt = (Fin * self.plant_params['Cain'] - Fout * CA) / V - r1 - r2
            dCbdt = (Fin * self.plant_params['CBin'] - Fout * CB) / V - r1
            dCcdt = - (Fout * CC) / V + r1 - r2
            Q_rxn = (-self.plant_params['dH1']) * r1 + (-self.plant_params['dH2']) * r2
            Q_in = Fin * self.plant_params['rho'] * self.plant_params['Cp'] * (self.plant_params['Tin'] - T)
            Q_jacket = UA * ((Tj + 273.15) - T)
            dTdt = (Q_in + Q_rxn + Q_jacket) / (self.plant_params['rho'] * self.plant_params['Cp'] * V)
        else:
            dCadt = dCbdt = dCcdt = dTdt = 0.0
            
        return np.array([T + dTdt * self.dt, max(0, CA + dCadt * self.dt), 
                         max(0, CB + dCbdt * self.dt), max(0, CC + dCcdt * self.dt), V + dVdt * self.dt])

    def _linearize(self, x_op, u_op, eps=1e-4):
        """Calcula Jacobiano (A = df/dx, B = df/du) via diferenças finitas"""
        n_x = len(x_op)
        A = np.zeros((n_x, n_x))
        B = np.zeros((n_x, 1))
        
        f_op = self._nominal_dynamics(x_op, u_op)
        
        # Perturba os estados (Para A)
        for i in range(n_x):
            x_pert = np.copy(x_op)
            x_pert[i] += eps
            f_pert = self._nominal_dynamics(x_pert, u_op)
            A[:, i] = (f_pert - f_op) / eps
            
        # Perturba as entradas (Para B)
        f_pert_u = self._nominal_dynamics(x_op, u_op + eps)
        B[:, 0] = (f_pert_u - f_op) / eps
        
        return A, B

    def _predict_linear(self, x, u):
        """Usa a aproximação linear: x_{k+1} = x_op + A*(x_k - x_op) + B*(u_k - u_op)"""
        dx = x - self.x_op
        du = u - self.u_op
        dx_next = self.A @ dx + (self.B @ [du]).flatten()
        return self.x_op + dx_next

    def step(self, x0, sp_T_K, u_prev):
        def objective(u_seq):
            cost = 0.0
            x = np.array(x0)
            u_current = u_prev
            for i in range(self.N):
                u = u_seq[i]
                cost += self.R * (u - u_current)**2
                # O LMPC usa apenas multiplicações de matriz para prever o futuro (rápido, mas impreciso longe de x_op)
                x = self._predict_linear(x, u) 
                cost += self.Q * (x[0] - sp_T_K)**2
                u_current = u
            return cost

        bounds = [(self.u_min, self.u_max) for _ in range(self.N)]
        u_init = np.roll(self.u_seq_prev, -1)
        u_init[-1] = u_init[-2]
        
        res = minimize(objective, u_init, method='L-BFGS-B', bounds=bounds)
        self.u_seq_prev = res.x
        
        u_opt = np.clip(res.x[0], u_prev - self.dU_max, u_prev + self.dU_max)
        return u_opt, res.x

# =====================================================================
# PARÂMETROS DO PROCESSO (compartilhados por planta e controladores)
# =====================================================================
NOMINAL_PARAMS = {'A1': 2.0e6, 'E1': 80000.0, 'A2': 1.0e8, 'E2': 90000.0}

PLANT_PARAMS = {
    # UA reduzido para faixa fisicamente coerente (~3000 W/K). Os valores
    # originais (3,3e5 / 3,0e4) eram ~100x grandes demais para um reator de
    # 1 m³ e causavam runaway térmico com Euler explícito.
    'UA_cool': 3000.0, 'UA_heat': 3000.0, 'T_cool': 298.15, 'T_steam': 408.15,
    'rho': 900.0, 'Cp': 2500.0, 'R': 8.314,
    'Cain': 1000.0, 'CBin': 2000.0, 'Tin': 308.15, 'F_in': 0.0012,
    'dH1': -184000.0, 'dH2': -220000.0
}


def qsig_to_watts(qsig, T_reactor_K):
    """
    Converte o sinal split-range do controlador (Qsig, 0-100 %) para a
    potência térmica Q_jacket [W] que a planta `model.CSTR` consome.

        0-50 %  -> refrigeração (Tj de 25 a 45 °C), UA_cool
        50-100% -> aquecimento  (Tj de 50 a 140 °C), UA_heat
        Q_jacket = UA * (Tj_K - T_reator_K)
    """
    if qsig <= 50:
        Tj = 25.0 + (qsig / 50.0) * 20.0
        UA = PLANT_PARAMS['UA_cool']
    else:
        Tj = 50.0 + ((qsig - 50.0) / 50.0) * 90.0
        UA = PLANT_PARAMS['UA_heat']
    return UA * ((Tj + 273.15) - T_reactor_K)


def build_plant():
    """
    Instancia a planta CSTR de 5 estados (model.CSTR) com física alinhada a
    PLANT_PARAMS/NOMINAL_PARAMS e condição inicial no ponto de operação (~50 °C).
    """
    plant = CSTR(
        Area=2.0, H_max=5.0, Cv_out=0.05,
        rho=PLANT_PARAMS['rho'], Cp=PLANT_PARAMS['Cp'],
        A1=NOMINAL_PARAMS['A1'], E1=NOMINAL_PARAMS['E1'],
        A2=NOMINAL_PARAMS['A2'], E2=NOMINAL_PARAMS['E2'],
        deltaH1=PLANT_PARAMS['dH1'], deltaH2=PLANT_PARAMS['dH2'],
    )
    plant.Temperature = 50.0 + 273.15
    plant.Volume = 1.0
    plant.CA, plant.CB, plant.CC, plant.CD = 400.0, 800.0, 100.0, 0.0
    return plant


# =====================================================================
# SIMULAÇÃO DE BENCHMARK
# =====================================================================
def run_simulation(controller, name, setpoint=65.0, duration=250, dt=0.05,
                   mpc_every=40):
    """
    Executa a malha fechada controlador↔planta (model.CSTR) e devolve um
    DataFrame com temperatura, setpoint e ação de controle.
    """
    plant = build_plant()
    n_steps = int(duration / dt)

    rec = []
    Qsig_opt = 50.0

    for i in range(n_steps):
        t = i * dt
        sp = 50.0 if t < 50 else setpoint

        if i % mpc_every == 0:
            x0 = [plant.Temperature, plant.CA, plant.CB, plant.CC, plant.Volume]
            Qsig_opt, _ = controller.step(x0, sp + 273.15, u_prev=Qsig_opt)

        # Controle de nível simples da válvula de saída (mantém o volume estável)
        level = plant.Volume / plant.Area if plant.Area > 0 else 0.5
        valve = np.clip(50 + (2.0 - level) * 30, 0, 100)

        # Converte o sinal split-range (%) para potência térmica (W)
        q_watts = qsig_to_watts(np.clip(Qsig_opt, 0, 100), plant.Temperature)

        # step(dt, F_in, T_in, CA_in, CB_in, Valve_Open_Pct, Q_heating[W])
        res = plant.step(dt, 0.0012, 308.15, 1000.0, 2000.0, valve, q_watts)
        rec.append({'t': t, 'T_C': res[1] - 273.15, 'T_sp': sp, 'Qsig': Qsig_opt})

    return pd.DataFrame(rec)


def main():
    print("Iniciando Benchmark: LMPC vs. ML-MPC Híbrido...")

    # Ponto de Operação para Linearização do LMPC (50°C)
    x_op = [50.0 + 273.15, 400.0, 800.0, 100.0, 1.0]
    u_op = 50.0

    print("Configurando Controladores...")
    # 1. LMPC Linear (Sem Rede Neural, Matrizes fixas)
    lmpc = LMPC(x_op, u_op, N=12, Q=5.0, R=0.5,
                nominal_params=NOMINAL_PARAMS, plant_params=PLANT_PARAMS)

    # 2. ML-MPC Híbrido (Preditor não-linear + Rede Neural)
    # Forward mockado. No seu projeto real, use o carregamento dos pesos gerados.
    nn_forward_mock = lambda T, Ca: 0.06
    ml_mpc = HybridNMPC(N=12, Q=5.0, R=0.5, dt_pred=2.0,
                        nominal_params=NOMINAL_PARAMS, plant_params=PLANT_PARAMS,
                        nn_forward=nn_forward_mock, u_min=0, u_max=100, dU_max=20)

    print("Simulando LMPC Clássico...")
    df_lmpc = run_simulation(lmpc, "LMPC")

    print("Simulando ML-MPC Híbrido...")
    df_mlmpc = run_simulation(ml_mpc, "ML-MPC")

    # Exibindo e plotando resultados
    print("\nResultados do Benchmark:")
    for name, df in [("LMPC Linear", df_lmpc), ("ML-MPC Híbrido", df_mlmpc)]:
        err = df[df['t'] > 50]['T_sp'] - df[df['t'] > 50]['T_C']
        mae = np.mean(np.abs(err))
        overshoot = max(0, df['T_C'].max() - 65.0)
        print(f"[{name}] MAE: {mae:.2f}°C | Overshoot: {overshoot:.2f}°C")

    plt.figure(figsize=(10, 6))
    plt.plot(df_lmpc['t'], df_lmpc['T_C'], 'r-', linewidth=1.5, label='LMPC Clássico (Linearizado)')
    plt.plot(df_mlmpc['t'], df_mlmpc['T_C'], 'b-', linewidth=2.0, label='ML-MPC (Não-Linear + NN)')
    plt.plot(df_mlmpc['t'], df_mlmpc['T_sp'], 'k--', label='Setpoint')
    plt.title('Benchmark: Startup Severo (50°C -> 65°C)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_lmpc, df_mlmpc


if __name__ == "__main__":
    main()