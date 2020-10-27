import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class KMeans3:
    def __init__(self, k, max_iter=300):
        """Constructor, """
        self.k = k
        self.max_iter = max_iter
        self.cluster_centers_ = np.zeros((k, 2))
        self.C = []
        self.inertia_ = 0

    def fit(self, X):
        """Funcion para entrenar el algoritmo"""
        # Inicializacion
        mu = X[np.random.randint(X.shape[0], size=self.k)]
        # bucle de entrenamiento
        for i in range(self.max_iter):
            # asignacion
            dmu = np.apply_along_axis(
                lambda c, x:
                np.linalg.norm(
                    np.tile(c, (x.shape[0], 1)) - x, axis=1
                ),
                1, mu, X).T

            self.C = np.argmin(dmu, axis=1)
            # actualizacion
            mu = np.apply_along_axis(
                lambda j: np.mean(X[np.where(self.C == j[0])[0]], axis=0),
                1, np.arange(self.k).reshape(self.k, 1))
        self.cluster_centers_ = mu
        self.inertia_ = np.sum(
            np.apply_along_axis(
                lambda j, x:
                np.sum(
                    np.linalg.norm(
                        np.tile(
                            self.cluster_centers_[j[0]],
                            (x[np.where(self.C == j[0])].shape[0], 1)
                        ) - x[np.where(self.C == j[0])], axis=1
                    )
                ), 1, np.arange(self.k).reshape(self.k, 1), X))


if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, centers=8,
                      cluster_std=0.70, random_state=0)
    print(X.shape)
    kmeans = KMeans3(7, max_iter=400)
    kmeans.fit(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(kmeans.cluster_centers_[
                :, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
