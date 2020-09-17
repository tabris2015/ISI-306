import math
clf_data = [
    #   x   y
       [22, 1],
       [23, 1],
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

def distancia(a, b):
    'distancia euclidiana para 1 dim'
    return abs(a - b)

def moda(lista):
    'devuelve el valor mas comun en una lista'
    if sum(lista) > len(lista) // 2:
        return 1
    else:
        return 0
    
def knn(data, consulta, k):
    'clasificacion binaria usando KNN'
    distancias = []
    
    for i, ej in enumerate(data):
        dist = distancia(ej[0], consulta)
        #                  0  1
        distancias.append((i,dist))
    
    ranking = sorted(distancias, key=lambda x: x[1])
    
    k_etiquetas = [data[i][1] for i, d in ranking[:k]]
    
    return moda(k_etiquetas)
    
    
edad = 48
k = 1
print(knn(clf_data, edad, k))
