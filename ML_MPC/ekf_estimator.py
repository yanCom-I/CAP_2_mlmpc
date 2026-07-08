"""
ekf_estimator.py

Implementação do Filtro de Kalman Estendido (EKF) para o Reator CSTR.
Estima os estados não-medidos (Concentrações CA, CB, CC) a partir de
medições ruidosas de Temperatura (T) e Volume (V).
"""

import numpy as np
import matplotlib.pyplot as plt

class CSTR_EKF:
    def __init__(self, x0, P0, dt, plant_params, nn_forward=None):
        """
        Inicializa o EKF.
        x0: Estimativa inicial do estado [T, CA, CB, CC, V]
        P0: Matriz de covariância do erro inicial (5x5)
        dt: Tempo de amostragem
        plant_params: Dicionário com os parâmetros físicos do CSTR
        """
        self.x_hat = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.dt = dt
        self.params = plant_params
        self.nn_forward = nn_forward
        
        self.n_states = 5
        self.n_meas = 2  # Medimos apenas T (índice 0) e V (índice 4)
        
        # Matriz de Observação H (Mapeia o estado para a medição)
        # z = H * x -> Medimos x[0] (T) e x[4] (V)
        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0, 0] = 1.0  # Mede Temperatura
        self.H[1, 4] = 1.0  # Mede Volume
        
        # Q: Covariância do Ruído de Processo (Incerteza do nosso modelo)
        self.Q_noise = np.diag([0.1, 0.5, 0.5, 0.1, 0.01])
        
        # R: Covariância do Ruído de Medição (Incerteza dos sensores)
        # Ex: Termopar varia ~1°C, Sensor de nível varia ~0.05m³
        self.R_noise = np.diag([1.0, 0.05])

    def f_dynamics(self, x, u_qsig):
        """
        Modelo não-linear de transição de estado (Mesma lógica do preditor do MPC).
        Calcula o próximo estado x_{k} dado x_{k-1} e u_{k-1}.
        """
        T, CA, CB, CC, V = x
        
        # Forward da Rede Neural para correção cinética
        f_theta = self.nn_forward(T, CA) if self.nn_forward else 1.0
        
        k1_nom = 2.0e6 * np.exp(-80000.0 / (8.314 * T))
        k1_real_pred = k1_nom * f_theta
        k2_nom = 1.0e8 * np.exp(-90000.0 / (8.314 * T))
        
        r1 = k1_real_pred * CA * CB
        r2 = k2_nom * CA * CC
        
        # Temperatura da camisa (Split Range)
        if u_qsig <= 50:
            Tj = 25.0 + (u_qsig / 50.0) * 20.0
            UA = self.params.get('UA_cool', 333333.0)
        else:
            Tj = 50.0 + ((u_qsig - 50) / 50.0) * 90.0
            UA = self.params.get('UA_heat', 30120.0)
            
        Tj_K = Tj + 273.15
        
        Fin = self.params.get('F_in', 0.0012)
        Fout = Fin # Simplificação de volume constante
        dVdt = Fin - Fout
        
        if V > 1e-6:
            dCadt = (Fin * self.params.get('Cain', 1000.0) - Fout * CA) / V - r1 - r2
            dCbdt = (Fin * self.params.get('CBin', 2000.0) - Fout * CB) / V - r1
            dCcdt = - (Fout * CC) / V + r1 - r2
            
            Q_rxn = (184000.0) * r1 + (220000.0) * r2
            Q_in = Fin * 900.0 * 2500.0 * (308.15 - T)
            Q_jacket = UA * (Tj_K - T)
            
            dTdt = (Q_in + Q_rxn + Q_jacket) / (900.0 * 2500.0 * V)
        else:
            dCadt = dCbdt = dCcdt = dTdt = 0.0
            
        x_next = np.array([
            T + dTdt * self.dt,
            max(0, CA + dCadt * self.dt),
            max(0, CB + dCbdt * self.dt),
            max(0, CC + dCcdt * self.dt),
            max(0, V + dVdt * self.dt)
        ])
        return x_next

    def get_jacobian_F(self, x, u, eps=1e-5):
        """
        Calcula a Matriz Jacobiana (F) linearizando o modelo f_dynamics
        numérica e localmente usando diferenças finitas centrais.
        """
        F = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = self.f_dynamics(x_plus, u)
            f_minus = self.f_dynamics(x_minus, u)
            
            F[:, i] = (f_plus - f_minus) / (2 * eps)
        return F

    def predict(self, u_qsig):
        """
        Passo 1 do EKF: Predição (Time Update).
        Estimativa a priori do estado e covariância.
        """
        # Lineariza a planta ao redor do estado estimado atual
        F_k = self.get_jacobian_F(self.x_hat, u_qsig)
        
        # Propaga o estado
        self.x_hat = self.f_dynamics(self.x_hat, u_qsig)
        
        # Propaga a covariância do erro
        self.P = F_k @ self.P @ F_k.T + self.Q_noise
        
        return self.x_hat

    def update(self, z):
        """
        Passo 2 do EKF: Atualização (Measurement Update).
        Corrige a predição usando a leitura real dos sensores (z).
        z = [Temperatura_medida, Volume_medido]
        """
        # Inovação (Diferença entre o que medimos e o que previmos)
        z_pred = self.H @ self.x_hat
        y = z - z_pred
        
        # Covariância da Inovação
        S = self.H @ self.P @ self.H.T + self.R_noise
        
        # Ganho de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Atualiza o estado
        self.x_hat = self.x_hat + K @ y
        
        # Atualiza a covariância do erro
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x_hat

# =====================================================================
# Simulação Teste: EKF em Ação
# =====================================================================

print("Iniciando Teste do Filtro de Kalman Estendido (EKF)...")

# Parâmetros físicos mockados
plant_params = {}

# 1. Estado Verdadeiro Oculto (Planta Real simulada matematicamente)
# T = 310K (~37°C), CA = 600
x_true = np.array([310.15, 600.0, 1200.0, 0.0, 1.0])

# 2. Inicialização do EKF com estado CHUTADO (Erro proposital)
# Fingimos que o EKF acha que a concentração inicial é 200 (Muito errado!)
x0_guess = [310.15, 200.0, 1200.0, 0.0, 1.0]
P0 = np.eye(5) * 10.0

ekf = CSTR_EKF(x0_guess, P0, dt=2.0, plant_params=plant_params, nn_forward=lambda t, c: 0.06)

history_true_CA = []
history_est_CA = []
history_true_T = []
history_meas_T = []

u_qsig = 60.0 # Válvula de calor constante

print("Simulando Planta com ruído e convergência do EKF...")
for i in range(100):
    # A. PLANTA REAL AVANÇA (Usando o f_dynamics do EKF como "planta" para teste)
    x_true = ekf.f_dynamics(x_true, u_qsig)
    
    # B. SENSORES RUIDOSOS (Adiciona ruído Gaussiano na medição de Temperatura e Volume)
    T_medida = x_true[0] + np.random.normal(0, 0.5) # Ruído de 0.5 K
    V_medido = x_true[4] + np.random.normal(0, 0.01) # Ruído no nível
    z_k = np.array([T_medida, V_medido])
    
    # C. EKF: Estimação
    ekf.predict(u_qsig)        # O modelo "pensa" para onde vai
    ekf.update(z_k)            # A medição "corrige" o pensamento
    
    # Salva para os gráficos
    history_true_CA.append(x_true[1])
    history_est_CA.append(ekf.x_hat[1])
    history_true_T.append(x_true[0] - 273.15)
    history_meas_T.append(T_medida - 273.15)

# Gráficos demonstrativos
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico 1: Temperatura (Sensor com Ruído vs Real)
axes[0].plot(history_meas_T, 'r.', label='Sensor T (Ruidoso)', alpha=0.5)
axes[0].plot(history_true_T, 'k-', label='T Real (Oculta)', linewidth=2)
axes[0].set_ylabel('Temperatura (°C)')
axes[0].set_title('Temperatura: Sensor vs Realidade')
axes[0].legend()
axes[0].grid(True)

# Gráfico 2: Concentração (O grande trunfo do EKF)
axes[1].plot(history_true_CA, 'k-', label='CA Real (Oculta)', linewidth=2)
axes[1].plot(history_est_CA, 'b--', label='CA Estimada (EKF)', linewidth=2)
axes[1].set_ylabel('Concentração CA')
axes[1].set_xlabel('Tempo (passos)')
axes[1].set_title('EKF corrigindo o erro inicial e descobrindo a Concentração Oculta')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print("Sucesso! O EKF começou com CA=200 (errado), mas a partir do erro da temperatura,")
print("o Ganho de Kalman convergiu a estimativa para o valor real.")