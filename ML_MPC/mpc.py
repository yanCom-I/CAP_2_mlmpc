"""
mpc.py

Módulo que abriga o ML-MPC (Controlador Preditivo Não-Linear Híbrido).
Esta implementação funde o modelo fenomenológico com a correção via Rede Neural
(NumPy), e incorpora as Restrições Suaves (Soft Constraints) do Ponto 4 do Roadmap.
"""

import numpy as np
from scipy.optimize import minimize

class HybridNMPC:
    def __init__(self, N=12, Q=5.0, R=0.5, dt_pred=2.0,
                 nominal_params=None, plant_params=None, nn_forward=None,
                 u_min=0.0, u_max=100.0, dU_max=20.0,
                 T_max=90.0, V_max=5.0, penalty_weight=1e5):
        """
        Inicialização do ML-MPC Robusto e Seguro.
        
        Args:
            N (int): Horizonte de predição (passos).
            Q (float): Peso da penalidade do erro de rastreamento (Setpoint).
            R (float): Peso da penalidade de esforço de controle (slew rate).
            dt_pred (float): Tamanho do passo de predição interno em segundos.
            T_max (float): Limite de segurança da Temperatura (°C). Soft constraint.
            V_max (float): Limite de segurança do Volume (m³). Soft constraint.
            penalty_weight (float): Multiplicador massivo para violações de segurança.
        """
        self.N = N
        self.Q = Q
        self.R = R
        self.dt = dt_pred
        
        # Parâmetros
        self.params = nominal_params
        self.plant_params = plant_params
        self.nn_forward = nn_forward
        
        # Limites Hard (Físicos dos atuadores)
        self.u_min = u_min
        self.u_max = u_max
        self.dU_max = dU_max
        
        # Limites Soft (Segurança do Reator)
        self.T_max_K = T_max + 273.15
        self.V_max = V_max
        self.penalty_weight = penalty_weight
        
        # Memória para o Warm-Start (Melhora drasticamente a convergência)
        self.u_seq_prev = np.ones(N) * 50.0 

    def predict_step(self, x, u_qsig):
        """
        Calcula o próximo estado x_{k+1} usando Balanços de Massa/Energia + NN.
        Estado x = [Temperatura, CA, CB, CC, Volume]
        """
        T, CA, CB, CC, V = x
        
        # 1. Correção Cinética Híbrida (Gray-box)
        # Se a rede neural estiver acoplada, busca o fator de correção; caso contrário, f = 1
        f_theta = self.nn_forward(T, CA) if self.nn_forward else 1.0
        
        k1_nom = self.params['A1'] * np.exp(-self.params['E1'] / (8.314 * T))
        k1_real_pred = k1_nom * f_theta  # Cinética ajustada
        
        k2_nom = self.params['A2'] * np.exp(-self.params['E2'] / (8.314 * T))
        
        r1 = k1_real_pred * CA * CB
        r2 = k2_nom * CA * CC
        
        # 2. Dinâmica da Camisa Térmica (Split-range)
        if u_qsig <= 50:
            Tj = 25.0 + (u_qsig / 50.0) * 20.0
            UA = self.plant_params['UA_cool']
        else:
            Tj = 50.0 + ((u_qsig - 50) / 50.0) * 90.0
            UA = self.plant_params['UA_heat']
            
        Tj_K = Tj + 273.15
        
        # 3. Dinâmica de Volume e Vazão
        Fin = self.plant_params['F_in']
        Fout = Fin # Assumindo nível estabilizado neste exemplo
        dVdt = Fin - Fout
        
        # 4. Integração das EDOs (Primeiros Princípios)
        if V > 1e-6:
            dCadt = (Fin * self.plant_params['Cain'] - Fout * CA) / V - r1 - r2
            dCbdt = (Fin * self.plant_params['CBin'] - Fout * CB) / V - r1
            dCcdt = - (Fout * CC) / V + r1 - r2
            
            Q_rxn = (-self.plant_params['dH1']) * r1 + (-self.plant_params['dH2']) * r2
            Q_in = Fin * self.plant_params['rho'] * self.plant_params['Cp'] * (self.plant_params['Tin'] - T)
            Q_jacket = UA * (Tj_K - T)
            
            dTdt = (Q_in + Q_rxn + Q_jacket) / (self.plant_params['rho'] * self.plant_params['Cp'] * V)
        else:
            dCadt = dCbdt = dCcdt = dTdt = 0.0
            
        # Passo de Euler explícito
        T_next = T + dTdt * self.dt
        CA_next = max(0, CA + dCadt * self.dt)
        CB_next = max(0, CB + dCbdt * self.dt)
        CC_next = max(0, CC + dCcdt * self.dt)
        V_next = max(0, V + dVdt * self.dt)
        
        return np.array([T_next, CA_next, CB_next, CC_next, V_next])

    def step(self, x0, sp_T_K, u_prev):
        """
        Encontra a ação de controle ótima para o instante k resolvendo o OCP
        (Optimal Control Problem) ao longo do horizonte N.
        """
        def cost_function(u_seq):
            cost = 0.0
            x = np.array(x0)
            u_current = u_prev
            
            for i in range(self.N):
                u = u_seq[i]
                
                # Custo: Penalidade por movimentação brusca do atuador (slew rate)
                cost += self.R * (u - u_current)**2
                
                # Avança 1 passo de predição
                x = self.predict_step(x, u)
                T_pred = x[0]
                V_pred = x[4]
                
                # Custo: Rastreamento do Setpoint (Tracking Error)
                cost += self.Q * (T_pred - sp_T_K)**2
                
                # ========================================================
                # SOFT CONSTRAINTS (Travas de Segurança)
                # ========================================================
                # 1. Trava Térmica
                if T_pred > self.T_max_K:
                    violation_T = T_pred - self.T_max_K
                    cost += self.penalty_weight * (violation_T ** 2)
                
                # 2. Trava de Transbordamento do Volume
                if V_pred > self.V_max:
                    violation_V = V_pred - self.V_max
                    cost += self.penalty_weight * (violation_V ** 2)

                u_current = u
                
            return cost

        # Configuração do Otimizador SciPy (L-BFGS-B)
        bounds = [(self.u_min, self.u_max) for _ in range(self.N)]
        
        # Warm-start: usa a solução deslocada do instante anterior como "chute" inicial
        u_init = np.roll(self.u_seq_prev, -1)
        u_init[-1] = u_init[-2] # O último passo copia o penúltimo
        
        res = minimize(
            cost_function, 
            u_init, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'ftol': 1e-4, 'maxiter': 50}
        )
        
        # Salva o horizonte ótimo para o warm-start da próxima iteração
        self.u_seq_prev = res.x
        
        # Slew Rate Hard Constraint na ação imediata que vai pra planta real
        u_opt = res.x[0]
        u_opt = np.clip(u_opt, u_prev - self.dU_max, u_prev + self.dU_max)
        
        return u_opt, res.x