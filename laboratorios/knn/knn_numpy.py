#%%
import numpy as np
import math
clf_data = [
    #   x   y
       [22, 0.1, 1],
       [23, 0.2, 1],
       [21, 1],
       [18, 1],
       [19, 1],
       [25, 0],
       [27, 0],
       [29, 0],
       [31, 0],
       [45, 0],
       [50, 1]      # outlier
    ]

dataset = np.array(clf_data)
#%%
def distancia(a, b):
    a = np.reshape(a, (-1,1))
    b = np.reshape(b, (-1,1))
    return np.sqrt(np.sum(np.square(a - b), axis=1)) 

def moda(array):
    return np.argmax(np.bincount(array))


def knn(data, consulta, k):
    distancias = distancia(data[:,0], np.repeat(consulta, data.shape[0]))
    ranking = np.argsort(distancias)  
    etiquetas = data[ranking[:k], 1]
    return moda(etiquetas)
#%%   
consulta = 35
k = 3
pred = knn(dataset, consulta, k)

print(pred)