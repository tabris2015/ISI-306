#%%
import numpy as np
from utils import load_classification_data

#%%
# Funciones para la regresion logistica

def g(z):
    'Funcion sigmoide'
    return 1 / (1 + np.exp(-z))

def h(X, theta):
    'Hipotesis de la regresion Logistica, devuelve una matriz de (1,m)'
    return g(np.dot(np.transpose(theta), X))

def l(X, y, theta):
    'Funcion de verosimilitud logaritmica de la regresion logistica, devuelve un escalar'
    m = X.shape[1]
    return (1/m) * np.sum(y * np.log(h(X,theta)) + (1 - y) * (1 - np.log(h(X,theta))))

def dl(X, y, theta):
    'Gradiente de la verosimilitud con respecto a theta, devuelve una matriz de (n + 1, 1)'
    m = X.shape[1]
    return (1/m) * np.dot(y - (h(X, theta)), np.transpose(X)).reshape((-1,1))

#%%
# Importar el dataset
X, y = load_classification_data()

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
iters = 1000     # numero de iteraciones
grads_history = []      # registro de gradientes
l_history = []       # registro de la funcion de costo
l_init = l(X, y, theta)     # costo inicial
print(f'Costo inicial: {l_init}')
l_history.append(l_init)

#%%
# ASCENSO DE GRADIENTE
# Batch Gradient Descent
for i in range(iters):
    # calculamos el gradiente
    dlikelihood = dl(X, y, theta)
    # actualizamos los parametros 
    theta = theta + alpha * dlikelihood
    
    # guardamos para visualizacion posterior
    l_history.append(l(X, y, theta))
    grads_history.append(dlikelihood)

print(f'Costo final: {l_history[-1]}')

#%%

# %%
import matplotlib.pyplot as plt
# it = list(range(10))
# plt.plot(it, l_history[:10])
it = list(range(iters + 1))
plt.plot(it, l_history)
plt.show()
# %%
