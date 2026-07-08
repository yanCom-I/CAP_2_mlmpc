"""
nn/forward_numpy.py

Este módulo emula o Forward Pass de uma Rede Neural PyTorch usando apenas NumPy.
Isso acelera a otimização do SciPy (L-BFGS-B) eliminando a sobrecarga de tensores
e da engine de autograd do PyTorch.
"""

import numpy as np

def extract_weights(pytorch_model):
    """
    Extrai as matrizes de peso (weights) e viés (biases) do modelo PyTorch.
    Nota: As camadas Linear do PyTorch salvam pesos no formato (out_features, in_features).
    """
    weights = []
    biases = []
    
    # Extrai iterativamente os parâmetros do modelo PyTorch
    for name, param in pytorch_model.named_parameters():
        if 'weight' in name:
            weights.append(param.detach().numpy())
        elif 'bias' in name:
            biases.append(param.detach().numpy())
            
    return weights, biases

def numpy_nn_forward(T, Ca, weights, biases, scaler_X, scaler_y):
    """
    Recria matematicamente o forward pass:
    Input -> Normalização -> Tanh(Dense) -> Tanh(Dense) -> Linear(Dense) -> Desnormalização
    """
    # 1. Normaliza as entradas (T e Ca)
    X = np.array([[T, Ca]])
    X_norm = (X - scaler_X.mean_) / scaler_X.scale_
    
    # 2. Passagem pela Rede Neural Oculta
    # Camada Oculta 1 (48 neurônios, ativação Tanh)
    # y = tanh(X * W^T + b)
    h1 = np.tanh(np.dot(X_norm, weights[0].T) + biases[0])
    
    # Camada Oculta 2 (24 neurônios, ativação Tanh)
    h2 = np.tanh(np.dot(h1, weights[1].T) + biases[1])
    
    # Camada de Saída (1 neurônio, ativação Linear)
    y_norm = np.dot(h2, weights[2].T) + biases[2]
    
    # 3. Desnormaliza a saída para a escala real da planta
    f_theta_real = (y_norm * scaler_y.scale_) + scaler_y.mean_
    
    return float(f_theta_real[0, 0])

def make_nn_forward(pytorch_model, scaler_X, scaler_y):
    """
    Função Helper (Closure) que empacota os pesos e scalers, retornando
    uma função simples f(T, Ca) para o controlador MPC chamar.
    """
    # Extraímos apenas uma vez na inicialização
    weights, biases = extract_weights(pytorch_model)
    
    def f_curried(T, Ca):
        return numpy_nn_forward(T, Ca, weights, biases, scaler_X, scaler_y)
        
    return f_curried