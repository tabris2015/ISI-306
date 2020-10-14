#%%
import numpy as np
from utils import load_regression_data

#%%
# Funciones para la regresion lineal

def h(X, theta):
    'Hipotesis de la regresion lineal, devuelve una matriz de (1,m)'
    return np.dot(np.transpose(theta), X)

def J(X, y, theta):
    'Funcion de costo de la regresion lineal, devuelve un escalar'
    m = X.shape[1]
    return (1/(2*m)) * np.sum(np.square(h(X, theta) - y))

def dJ(X, y, theta):
    'Gradiente del costo con respecto a theta, devuelve una matriz de (n + 1, 1)'
    m = X.shape[1]
    return (1/m) * np.dot((h(X, theta) - y), np.transpose(X)).reshape((-1,1))

#%%
# Importar el dataset
X, y = load_regression_data()

m = X.shape[1]  # Numero de ejemplos
n = X.shape[0]  # Numero de caracteristicas

# dimensiones
print(f'X:{X.shape}, y: {y.shape}')

#%%
# Normalizar los vectores de entrada y salida
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# Agregar una fila de 1 al inicio del dataset
unos = np.ones((1, m))

X = np.append(unos, X, axis=0)
print(f'X_:{X.shape}, y: {y.shape}')

#%%
# ENTRENAMIENTO
# Inicializar los parametros 
theta = np.random.random((n + 1, 1))
print(f'Theta: {theta.shape}')
# Hiperparametros
alpha = 0.02   # learning rate
iters = 500     # numero de iteraciones
grads_history = []      # registro de gradientes
J_history = []       # registro de la funcion de costo
J_init = J(X, y, theta)     # costo inicial
print(f'Costo inicial: {J_init}')
J_history.append(J_init)

#%%
# DESCENSO DE GRADIENTE
# Batch Gradient Descent
for i in range(iters):
    # calculamos el gradiente
    dcost = dJ(X, y, theta)
    # actualizamos los parametros 
    theta = theta - alpha * dcost
    
    # guardamos para visualizacion posterior
    J_history.append(J(X, y, theta))
    grads_history.append(dcost)

print(f'Costo final: {J_history[-1]}')


# %%
import matplotlib.pyplot as plt
# it = list(range(10))
# plt.plot(it, J_history[:10])
it = list(range(iters + 1))
plt.plot(it, J_history)
plt.show()
# %%
