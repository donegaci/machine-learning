import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def read_csv(filename):
    df = pd.read_csv(filename, comment="#")
    # print(df.head())
    X = np.array(df.iloc[:,0])
    X = X.reshape(-1, 1)
    y = np.array(df.iloc[:,1])
    y = y.reshape(-1,1)
    return X, y


def normalise(values):
    mean = values.mean()
    sigma = values.max() - values.min()
    norm = (values - mean) / sigma
    return norm

X, y = read_csv("week1.csv")
X_norm = normalise(X)

reg = LinearRegression().fit(X_norm, y)
print("y-intercept", reg.intercept_)
print("slope = ", reg.coef_)