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

def sgd(X, y, lr, epochs):
    m = len(X) # number of features
    assert len(X) == len(y) # ensure labels and fetaures are same size

    # initialise paramaters to 0
    theta_0 = theta_1 = 0

    past_costs = []
    for _ in range(epochs):
        # function to learn
        h = theta_0 + theta_1 * X
        # cost function
        cost = (1/2*m) * ((h - y)**2).sum()
        past_costs.append(cost)
        # update paramaters
        theta_0 -=  ( 2 * lr * (h-y).sum() ) / m
        theta_1 -=  ( 2 * lr * (X * (h-y)).sum() )  / m
    
    return theta_0, theta_1, past_costs


X, y = read_csv("week1.csv")
X_norm = normalise(X)
epochs = 500

plt.figure(1)
plt.plot(X, y)

learned_values = []

# Perfrom gradient descent for different learning rates and plot the results
for i, lr in enumerate([0.001, 0.01, 0.1, 1]):
    theta_0, theta_1, past_costs = sgd(X_norm, y, lr, epochs)
    print ("lr = ", lr, theta_0, theta_1)
    print("cost=", past_costs[-1])
    y_hat = theta_0 + theta_1 * X_norm
    
    plt.figure(1)
    plt.plot(X, y_hat, linewidth=3)
    plt.figure(2)
    plt.plot(range(epochs), past_costs)



plt.figure(1)
plt.legend(["data", "lr = 0.001", "lr = 0.01", "lr = 0.1", "lr = 1"])
plt.title("Learned models for different learning rates")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.savefig( "models.png")
plt.figure(2)
plt.legend(["0.001", "0.01", "0.1", "1"])
plt.title("$J(\Theta)$ vs itteration for different learning rates")
plt.xlabel("Itteration")
plt.ylabel("Cost $J(\Theta)$")
plt.grid()
plt.savefig("cost_function.png")