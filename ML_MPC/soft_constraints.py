"""
mpc_soft_constraints.py

Implementação do NMPC Híbrido focado no Ponto 4 do Roadmap:
Restrições de Segurança (Soft Constraints) para T < 90°C e V < V_max.
Baseado na estrutura multi-estados do model.py.
"""

import numpy as np
import copy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Importa o modelo CSTR original fornecido
from model import CSTR

class SafeNMPC:
    def __init__(self, N=10, Q_track=10.0, R_move=0.1, dt=0.1):
        self.N = N            # Horizonte de predição
        self.Q_track = Q_track  # Peso para rastreamento de Setpoint de Temperatura
        self.R_move = R_move    # Peso para suavidade da ação de controlo
        self.dt = dt
        
        # --- LIMITES DE SEGURANÇA (PONTO 4 DO ROADMAP) ---
        self.T_max_K = 90.0 + 273.15  # 90°C convertido para Kelvin
        self.V_max = 4.5              # Volume máximo permitido (ex: capacidade física do reator)
        
        # Pesos de penalização (Valores muito altos para atuar como "parede" invisível)
        self.rho_T = 1e5  
        self.rho_V = 1e5
        
        # Limites HARD apenas para as Variáveis Manipuladas (MV)
        # Vamos assumir que controlamos o Q_heating (-100000 a 100000 Watts)
        self.u_min = -100000.0
        self.u_max = 100000.0

    def objective_function(self, u_seq, cstr_current_state, setpoint_T, u_prev, current_inputs):
        """
        Função objetivo calculada ao longo do horizonte N.
        """
        cost = 0.0
        
        # Fazemos uma cópia profunda do estado atual do CSTR para simular o futuro
        # sem alterar o reator real.
        cstr_sim = copy.deepcopy(cstr_current_state)
        
        # Desempacota os inputs constantes do processo
        F_in, T_in, CA_in, CB_in, Valve_Open_Pct = current_inputs
        
        u_last = u_prev
        
        for k in range(self.N):
            u_k = u_seq[k]
            
            # 1. Penaliza a variação da ação de controlo (Suavidade)
            cost += self.R_move * (u_k - u_last)**2
            u_last = u_k
            
            # 2. Avança o modelo interno de primeiros princípios (model.py)
            # step() retorna: level, Temperature, F_out, CA, CB, CC, CD
            _, T_pred, _, _, _, _, _ = cstr_sim.step(
                dt=self.dt, 
                F_in=F_in, 
                T_in=T_in, 
                CA_in=CA_in, 
                CB_in=CB_in, 
                Valve_Open_Pct=Valve_Open_Pct, 
                Q_heating=u_k
            )
            V_pred = cstr_sim.Volume
            
            # 3. Penaliza o desvio do Setpoint (Tracking)
            cost += self.Q_track * (T_pred - setpoint_T)**2
            
            # -----------------------------------------------------------------
            # ALGORITMO DO PONTO 4: SOFT CONSTRAINTS (Restrições de Segurança)
            # -----------------------------------------------------------------
            # Se T_pred > T_max_K, calcula o quadrado da violação e multiplica pelo peso rho_T
            violation_T = max(0.0, T_pred - self.T_max_K)
            cost += self.rho_T * (violation_T ** 2)
            
            # Se V_pred > V_max, calcula o quadrado da violação e multiplica pelo peso rho_V
            violation_V = max(0.0, V_pred - self.V_max)
            cost += self.rho_V * (violation_V ** 2)
            
        return cost

    def control_action(self, cstr_current_state, setpoint_T, u_prev, current_inputs):
        """
        Resolve o problema de otimização para o passo atual.
        """
        # Chute inicial para o horizonte (Warm-start mantendo a última ação)
        u_init = np.ones(self.N) * u_prev
        
        # Define os limites físicos da válvula/aquecedor (Hard Bounds da MV)
        bounds = [(self.u_min, self.u_max) for _ in range(self.N)]
        
        # Otimização via SciPy (L-BFGS-B lida bem com funções de custo contínuas e bounds)
        res = minimize(
            self.objective_function,
            u_init,
            args=(cstr_current_state, setpoint_T, u_prev, current_inputs),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-3, 'maxiter': 30}
        )
        
        # Retorna a primeira ação de controlo do horizonte ótimo (Princípio do Horizonte Recuado)
        return res.x[0]


# =====================================================================
# Cenário de Teste / Simulação da Trava de Segurança
# =====================================================================

print("Iniciando Simulação do NMPC com Restrições Suaves de Segurança...")

# Instancia a planta real a partir do model.py
reactor_plant = CSTR(Area=2.0, H_max=5.0, Cv_out=0.05)

# Condições iniciais perigosas ou agressivas para testar o limite
reactor_plant.Volume = 3.5
reactor_plant.Temperature = 330.0  # ~57°C
reactor_plant.CA = 0.5
reactor_plant.CB = 0.5

# Configura o Controlador
nmpc = SafeNMPC(N=8, Q_track=20.0, R_move=0.01, dt=0.1)

# Entradas fixas do processo para o teste
# F_in, T_in, CA_in, CB_in, Valve_Open_Pct
process_inputs = (1.5, 300.0, 2.0, 2.0, 40.0) 

# SETPOINT IMPOSSÍVEL/PERIGOSO: Pedimos 110°C (383.15 K)
# Mas a restrição de segurança do Ponto 4 bloqueia em 90°C (363.15 K)
target_T = 110.0 + 273.15 

history_T = []
history_V = []
history_Q = []

u_heating = 0.0

# Simula 50 passos de tempo (~5 horas de operação se dt=0.1h)
for step in range(50):
    # 1. Calcula a ação de controlo ótima considerando as restrições de estado
    u_heating = nmpc.control_action(reactor_plant, target_T, u_heating, process_inputs)
    
    # 2. Aplica na Planta Real (model.py)
    F_in, T_in, CA_in, CB_in, Valve_Open_Pct = process_inputs
    reactor_plant.step(0.1, F_in, T_in, CA_in, CB_in, Valve_Open_Pct, u_heating)
    
    # Salva dados para análise
    history_T.append(reactor_plant.Temperature - 273.15) # Celsius
    history_V.append(reactor_plant.Volume)
    history_Q.append(u_heating)
    
    print(f"Passo {step:02d} | T: {history_T[-1]:.2f}°C (Limite: 90°C) | V: {history_V[-1]:.2f}m³ | Q_heat: {u_heating:.1f}W")

# --- Plotagem dos Resultados ---
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(history_T, 'r-', label='Temperatura Real do Reator', linewidth=2)
plt.axhline(90.0, color='k', linestyle='--', label='Restrição de Segurança (90°C)')
plt.axhline(target_T - 273.15, color='g', linestyle=':', label='Setpoint Solicitado (110°C)')
plt.ylabel('Temperatura (°C)')
plt.title('NMPC com Soft Constraints (Ponto 4 do Roadmap)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(history_V, 'b-', label='Volume Real do Reator', linewidth=2)
plt.axhline(nmpc.V_max, color='k', linestyle='--', label='Limite Máximo de Volume')
plt.ylabel('Volume (m³)')
plt.xlabel('Passos de Controlo (dt=0.1h)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()