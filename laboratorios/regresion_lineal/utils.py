#%%
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer

def load_regression_data():
    X_raw, y_raw = load_diabetes(return_X_y=True)
    return np.transpose(X_raw), np.reshape(y_raw, (1,-1))


def load_classification_data():
    X_raw, y_raw = load_breast_cancer(return_X_y=True)
    return np.transpose(X_raw), np.reshape(y_raw, (1,-1))