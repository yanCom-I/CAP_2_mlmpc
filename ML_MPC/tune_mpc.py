"""
tune_mpc.py

Script standalone para Sintonização Automatizada do ML-MPC usando 
Evolução Diferencial (algoritmo de enxame do SciPy).
O objetivo é encontrar os melhores pesos (Q e R) minimizando o MAE 
e o esforço de controle.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Planta multi-estados compatível com o controlador (5 estados: T, CA, CB, CC, V)
from model import CSTR
from mpc import HybridNMPC
from nn_forward_numpy import make_nn_forward

# =========================================================
# Parâmetros do processo e utilitários da planta model.CSTR
# =========================================================
# CORREÇÕES em relação ao tune_mpc.py original:
#  - RealCSTR (2 estados, step(Qsig, dt)) trocada por model.CSTR (5 estados),
#    eliminando o erro "RealCSTR.step() takes 2..3 positional arguments but 8
#    were given" e o uso de atributos inexistentes (plant.Temperature, .CA...).
#  - Conversão split-range Qsig(%) -> potência térmica em Watts antes de step().
#  - UA reduzido a ~3000 W/K e dt=0,05 s para estabilidade do Euler explícito.

NOMINAL_PARAMS = {'A1': 2.0e6, 'E1': 80000.0, 'A2': 1.0e8, 'E2': 90000.0}

PLANT_PARAMS = {
    'UA_cool': 3000.0, 'UA_heat': 3000.0, 'T_cool': 298.15, 'T_steam': 408.15,
    'rho': 900.0, 'Cp': 2500.0, 'R': 8.314,
    'Cain': 1000.0, 'CBin': 2000.0, 'Tin': 308.15, 'F_in': 0.0012,
    'dH1': -184000.0, 'dH2': -220000.0,
}

DT_SIM = 0.05  # passo de integração da planta (s)


def qsig_to_watts(qsig, T_reactor_K):
    """Converte o sinal split-range Qsig (0-100 %) para potência térmica [W]."""
    if qsig <= 50:
        Tj = 25.0 + (qsig / 50.0) * 20.0
        UA = PLANT_PARAMS['UA_cool']
    else:
        Tj = 50.0 + ((qsig - 50.0) / 50.0) * 90.0
        UA = PLANT_PARAMS['UA_heat']
    return UA * ((Tj + 273.15) - T_reactor_K)


def build_plant():
    """Instancia model.CSTR (5 estados) no ponto de operação (~50 °C)."""
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

# =========================================================
# Definição da Arquitetura da Rede Neural
# =========================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48), 
            nn.Tanh(),
            nn.Linear(48, 24), 
            nn.Tanh(),
            nn.Linear(24, 1)
        )
    def forward(self, x): 
        return self.net(x)

# =========================================================
# Geração de Dados e Treinamento
# =========================================================
def load_surrogate_model():
    """
    Gera dados transientes da planta CSTR, treina a Rede Neural (Surrogate Model)
    com Early Stopping e retorna a função forward mapeada com os scalers para o MPC.
    """
    print("="*60)
    print("Iniciando Geração de Dados e Treinamento da Rede Neural...")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    dt = DT_SIM
    n_seq = 15
    T_seq = 300
    rec = []
    
    print("Gerando dados transientes (15 sequências de 300 passos)...")
    for seq in range(n_seq):
        if seq % 5 == 0: 
            print(f'  Sequência {seq+1}/{n_seq}')
            
        plant = build_plant()
        Qsig = 50.0
        Fin = 0.0012
        
        for step in range(T_seq):
            if step > 0 and step % np.random.randint(80, 120) == 0:
                Qsig = np.clip(Qsig + np.random.uniform(-40, 40), 0, 100)
                Fin = np.clip(0.0012 + np.random.uniform(-0.0005, 0.0005), 0.0005, 0.0025)
            
            # Converte Qsig(%) -> Watts antes de aplicar na planta model.CSTR
            q_watts = qsig_to_watts(Qsig, plant.Temperature)
            res = plant.step(dt, Fin, 308.15, 1000.0, 2000.0, 50.0, q_watts)
            Tk, CA, CB = res[1], res[3], res[4]
            
            k1r = 5e6 * np.exp(-75000.0 / (8.314 * Tk))
            k1n = 2e6 * np.exp(-70000.0 / (8.314 * Tk))
            f_target = k1r / k1n
            f_target = f_target / (1 + 0.008 * CA)
            
            rec.append({'T': Tk, 'CA': CA, 'CB': CB, 'f_theta': f_target})
            
    df_data = pd.DataFrame(rec)
    df_data = df_data[(df_data['T'] > 300) & (df_data['T'] < 400) & (df_data['CA'] > 1)]
    df_data = df_data.sample(n=min(2000, len(df_data)), random_state=42).reset_index(drop=True)
    
    X_data = df_data[['T', 'CA']].values
    y_target = df_data['f_theta'].values.reshape(-1, 1)
    
    print(f'\nDataset transiente processado: {X_data.shape[0]} amostras!')
    
    # Pre-processamento e normalização
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X_data)
    y_norm = scaler_y.fit_transform(y_target)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)), 
        batch_size=64, shuffle=True
    )
    
    # Inicialização da Rede e Otimizador
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    crit = nn.MSELoss()
    
    n_ep = 150
    best = 1e9
    best_ep = 0
    best_state = None
    
    print("\nTreinando o modelo (150 épocas)...")
    for ep in range(n_ep):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            l = crit(model(Xb), yb)
            l.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad(): 
            tl = crit(model(torch.FloatTensor(X_te)), torch.FloatTensor(y_te)).item()
            
        if tl < best: 
            best = tl
            best_ep = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
        if ep % 50 == 0: 
            print(f'  Ep {ep:3d} | Teste MSE: {tl:.6f} | Melhor = {best:.6f} @ {best_ep}')
            
    model.load_state_dict(best_state)
    print(f'\n[OK] Early stopping acionado: best_val={best:.6f} na epoca {best_ep}')
    print("="*60)
    
    return make_nn_forward(model, scaler_X, scaler_y)

# =========================================================
# Função de Fitness para o Otimizador
# =========================================================
def simulate_and_evaluate(weights, nn_forward_curried, nominal_params, plant_params):
    Q_val, R_val = weights
    
    if Q_val < 0 or R_val < 0:
        return 1e6

    mpc = HybridNMPC(
        N=12, Q=Q_val, R=R_val, dt_pred=2.0,
        nominal_params=nominal_params, plant_params=plant_params,
        nn_forward=nn_forward_curried, u_min=0.0, u_max=100.0, dU_max=20.0
    )
    
    plant = build_plant()
    n_steps = int(200 / DT_SIM)
    mpc_every = 40  # recalcula a cada 2 s de processo (40 * 0,05 s)
    
    Qsig_opt = 50.0
    mae_acumulado = 0.0
    esforco_controle = 0.0
    
    try:
        for i in range(n_steps):
            t = i * DT_SIM
            sp = 50.0 if t < 50 else 65.0 
            
            if i % mpc_every == 0:
                x0 = [plant.Temperature, plant.CA, plant.CB, plant.CC, plant.Volume]
                Qsig_next, _ = mpc.step(x0, sp + 273.15, u_prev=Qsig_opt)
                esforco_controle += abs(Qsig_next - Qsig_opt)
                Qsig_opt = Qsig_next
                
            cl = plant.Volume / plant.Area if plant.Area > 0 else 0
            vo = np.clip(50 + (2.0 - cl) * 30, 0, 100)
            q_watts = qsig_to_watts(np.clip(Qsig_opt, 0, 100), plant.Temperature)
            res = plant.step(DT_SIM, 0.0012, 308.15, 1000.0, 2000.0, vo, q_watts)
            
            if t >= 50:
                mae_acumulado += abs((res[1] - 273.15) - sp)
                
    except Exception:
        return 1e6
        
    return (1.0 * mae_acumulado) + (0.05 * esforco_controle)

# =========================================================
# Execução Principal (Auto-Tuning)
# =========================================================
def main():
    print("Iniciando Otimização de Parâmetros do NMPC (Auto-Tuning)...")
    
    nn_forward_curried = load_surrogate_model()
    
    # Usa os parâmetros compartilhados e corrigidos (mesma física da planta).
    nominal_params = NOMINAL_PARAMS
    plant_params_mpc = PLANT_PARAMS
    
    bounds = [(0.1, 50.0), (0.01, 10.0)]
    
    def objective_wrapper(w):
        return simulate_and_evaluate(w, nn_forward_curried, nominal_params, plant_params_mpc)
    
    start_time = time.time()
    print(f"Limites de busca: Q em {bounds[0]}, R em {bounds[1]}")
    print("Otimizador rodando (isso pode levar alguns minutos)...\n")
    
    # Orçamento reduzido (maxiter/popsize) para tempo de execução prático.
    # Aumente para refinar o ótimo.
    result = differential_evolution(
        objective_wrapper, 
        bounds, 
        strategy='best1bin', 
        maxiter=3,       
        popsize=3,        
        tol=0.01, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        seed=42,          
        disp=True         
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("="*50)
    if result.success:
        best_Q, best_R = result.x
        print(f"Sucesso! Encontrados os pesos ótimos em {elapsed_time:.1f} segundos:")
        print(f"  -> Q ideal : {best_Q:.4f}")
        print(f"  -> R ideal : {best_R:.4f}")
        print(f"Custo mínimo alcançado: {result.fun:.4f}")
    else:
        print("A otimização encontrou dificuldades para convergir.")
        print("Mensagem:", result.message)

    return result


if __name__ == "__main__":
    main()