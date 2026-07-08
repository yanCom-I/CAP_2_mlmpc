"""Simulador do Reator CSTR de Etoxilação."""

import numpy as np
from scipy.integrate import odeint


class RealCSTR:
    """
    Modelo dinâmico não-linear do CSTR de etoxilação.
    
    Reação: A + nEO -> B (produto)
    
    Estados: Ca (concentração de A), T (temperatura)
    Entrada de controle: Qsig (0-100%) via split-range
        - 0-50%: água de resfriamento a 25°C
        - 50-100%: vapor a 135°C
    
    Parâmetros baseados em etoxilação industrial (escala didática).
    """
    
    def __init__(self):
        # Parâmetros do processo
        self.V = 10.0        # m³
        self.F = 1.0         # m³/h (tempo de residência = 10h)
        self.Caf = 1.0       # kmol/m³
        self.Tf = 300.0      # K (temperatura de alimentação)
        
        # Parâmetros cinéticos - etoxilação moderada
        self.deltaH = -60000.0   # kJ/kmol (exotérmica)
        self.k0 = 5.0e7          # fator pré-exponencial (1/h)
        self.Ea = 45000.0        # J/mol
        self.R = 8.314           # J/(mol·K)
        self.Ceo = 5.0           # kmol/m³ (EO em excesso)
        
        # Troca térmica
        self.rho = 1000.0    # kg/m³
        self.cp = 4.2        # kJ/(kg·K)
        self.UA = 500.0       # kJ/(h·K)
        
        # Estado inicial - equilíbrio inicial
        self.Ca = 0.6        # kmol/m³
        self.T = 340.0       # K
        
        # Histórico
        self.history = {
            't': [0], 'Ca': [self.Ca], 'T': [self.T],
            'Qsig': [50], 'time': [0]
        }
    
    def kinetics(self, Ca, T):
        """Taxa de reação de etoxilação (1/h)."""
        k = self.k0 * np.exp(-self.Ea / (self.R * T))
        return k * Ca
    
    def heat_transfer(self, Qsig):
        """Temperatura da jaqueta via split-range."""
        if Qsig <= 50:
            Tj = 25.0 + (Qsig / 50.0) * 20.0  # 25°C a 45°C
        else:
            Tj = 50.0 + ((Qsig - 50) / 50.0) * 90.0  # 50°C a 140°C
        return Tj
    
    def derivatives(self, y, t, Qsig):
        """Sistema de EDOs do CSTR."""
        Ca, T = y
        r = self.kinetics(Ca, T)
        
        # Balanço de massa
        dCadt = self.F / self.V * (self.Caf - Ca) - r
        
        # Balanço de energia (convertendo kJ para J corretamente)
        Tj = self.heat_transfer(Qsig)
        dTdt = (self.F / self.V * (self.Tf - T) 
                - self.deltaH * 1000 / (self.rho * self.cp * 1000) * r
                + self.UA / (self.V * self.rho * self.cp * 1000) * (Tj - T))
        
        return [dCadt, dTdt]
    
    def step(self, Qsig, dt=0.1):
        """Avança o simulador por dt horas."""
        Qsig = np.clip(Qsig, 0, 100)
        
        y0 = [self.Ca, self.T]
        t = np.linspace(0, dt, 11)
        
        sol = odeint(self.derivatives, y0, t, args=(Qsig,))
        
        self.Ca = sol[-1, 0]
        self.T = sol[-1, 1]
        
        # Limites físicos
        self.Ca = np.clip(self.Ca, 0, self.Caf)
        self.T = np.clip(self.T, 250, 450)
        
        # Registra histórico
        self.history['t'].append(self.history['t'][-1] + dt)
        self.history['Ca'].append(self.Ca)
        self.history['T'].append(self.T)
        self.history['Qsig'].append(Qsig)
        
        return self.Ca, self.T
    
    def reset(self):
        """Reseta para condições iniciais."""
        self.Ca = 0.6
        self.T = 340.0
        self.history = {
            't': [0], 'Ca': [self.Ca], 'T': [self.T],
            'Qsig': [50], 'time': [0]
        }


def generate_training_data(cstr, n_samples=5000, n_steps=50):
    """
    Gera dados de treinamento via sequências de passos aleatórios.
    
    Crucial: dados transientes, não grade estacionária!
    """
    X, Y = [], []
    
    for _ in range(n_samples):
        cstr.reset()
        
        # Condição inicial aleatória
        cstr.Ca = np.random.uniform(0.1, 0.9)
        cstr.T = np.random.uniform(300, 380)
        
        # Sequência de passos aleatórios
        for _ in range(n_steps):
            Qsig = np.random.uniform(10, 90)
            
            # Estado atual
            x = [cstr.Ca, cstr.T, Qsig]
            
            # Próximo estado
            Ca_next, T_next = cstr.step(Qsig, dt=0.1)
            
            X.append(x)
            Y.append([Ca_next, T_next])
    
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    # Teste rápido
    cstr = RealCSTR()
    print(f"Estado inicial: Ca={cstr.Ca:.3f}, T={cstr.T:.1f}K")
    
    # Simula diferentes controles
    print("\n--- Resposta ao degrau ---")
    for qsig in [30, 60, 90]:
        cstr.reset()
        print(f"\nQsig = {qsig}%")
        for i in range(30):
            Ca, T = cstr.step(qsig, dt=0.1)
        print(f"  Ca = {Ca:.3f}, T = {T:.1f}K")
    
    # Gera dados
    X, Y = generate_training_data(cstr, n_samples=100)
    print(f"\nDados: X shape={X.shape}, Y shape={Y.shape}")