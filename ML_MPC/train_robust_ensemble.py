"""
train_robust_ensemble.py

Script standalone para treinar um Ensemble Efetivo de Redes Neurais para o NMPC Robusto.
Utiliza Bootstrap Sampling e Dropout para garantir alta variância (incerteza epistêmica)
em regiões fora da distribuição de dados.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

from plant import RealCSTR, generate_training_data

# =====================================================================
# 1. Definição da Rede Neural com Dropout
# =====================================================================
class RobustNet(nn.Module):
    def __init__(self, p_drop=0.3):
        """
        Rede Neural para o preditor.
        p_drop: Taxa de Dropout (ex: 0.3 = 30% dos neurônios são desligados no treino)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh(),
            nn.Dropout(p_drop),  # Induz diversidade na rede
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Dropout(p_drop),  # Induz diversidade na rede
            nn.Linear(24, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 2. Função de Bootstrap Sampling
# =====================================================================
def get_bootstrap_sample(X, y):
    """
    Gera um dataset de treino por amostragem com reposição (Bootstrap).
    As amostras que não forem sorteadas (Out-Of-Bag) servem como validação.
    """
    n_samples = len(X)
    # Sorteia índices com reposição
    train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    # Índices que ficaram de fora viram teste
    test_indices = np.array(list(set(range(n_samples)) - set(train_indices)))
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, y_train, X_test, y_test

# =====================================================================
# 3. Pipeline Principal de Treinamento
# =====================================================================
def main():
    print("Iniciando Treinamento do Ensemble Robusto (Bootstrap + Dropout)...")
    
    # ---------------------------------------------------------
    # A. Carregar/Gerar Dados (Mock para o exemplo)
    # Substitua por sua lógica de generate_training_data(cstr)
    # ---------------------------------------------------------
    print("Gerando dados sinteticos transientes...")
    # Mock data: T em [300, 390], Ca em [1, 1000]
    n_total = 2000
    X_data = np.random.uniform(low=[300.0, 1.0], high=[390.0, 1000.0], size=(n_total, 2))
    # f_theta mockada (depende de T e Ca)
    y_target = (0.05 * (X_data[:, 0] / 300) / (1 + 0.008 * X_data[:, 1])).reshape(-1, 1)
    
    scaler_X = StandardScaler().fit(X_data)
    scaler_y = StandardScaler().fit(y_target)
    
    X_norm = scaler_X.transform(X_data)
    y_norm = scaler_y.transform(y_target)
    
    # ---------------------------------------------------------
    # B. Configuração do Ensemble
    # ---------------------------------------------------------
    N_ENSEMBLE = 5
    EPOCHS = 150
    BATCH_SIZE = 64
    PATIENCE = 20 # Para Early Stopping
    
    os.makedirs("ensemble_models", exist_ok=True)
    
    # ---------------------------------------------------------
    # C. Treinamento de cada Rede
    # ---------------------------------------------------------
    for k in range(N_ENSEMBLE):
        print(f"\n--- Treinando Rede {k+1}/{N_ENSEMBLE} ---")
        
        # Semente aleatória diferente para cada rede
        torch.manual_seed(42 + k)
        np.random.seed(42 + k)
        
        # 1. Aplica o Bootstrap
        X_tr, y_tr, X_te, y_te = get_bootstrap_sample(X_norm, y_norm)
        print(f"Dataset de Treino (Bootstrap): {len(X_tr)} amostras | Teste (OOB): {len(X_te)} amostras")
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        
        # 2. Inicializa o modelo Robusto
        model = RobustNet(p_drop=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.MSELoss()
        
        best_test_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0
        
        # 3. Loop de Épocas com Early Stopping
        for ep in range(EPOCHS):
            model.train() # Ativa o Dropout
            train_loss = 0.0
            
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            
            # Validação
            model.eval() # Desativa Dropout para avaliar
            with torch.no_grad():
                test_loss = criterion(model(torch.FloatTensor(X_te)), torch.FloatTensor(y_te)).item()
            
            # Checa Early Stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {key: val.clone() for key, val in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {ep}. Best Val MSE: {best_test_loss:.6f}")
                break
                
        # 4. Restaura melhores pesos e salva
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), f"ensemble_models/robust_net_{k}.pt")
        print(f"Rede {k+1} salva com sucesso.")

    print("\nTreinamento do Ensemble Finalizado!")
    print("Agora você pode carregar essas redes no 'RobustHybridNMPC' para avaliar a variância real.")


main()