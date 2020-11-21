import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



def read_csv(filename):
    df = pd.read_csv(filename, comment="#")
    # print(df.head())
    X = np.array(df.iloc[:,0])
    X = X.reshape(-1, 1)
    y = np.array(df.iloc[:,1])
    y = y.reshape(-1,1)
    return X, y


def knn_and_ridge_regression_kernels(X, y):

    X_test = np.arange(-3, 3, 0.1).reshape(-1,1)

    def gaussina_kernel(distances):
        weights = np.exp(-gamma * (distances**2))
        return weights / np.sum(weights)

    plt.figure()
    plt.scatter(X, y, color='red', label="data")
    for gamma in [1000]:
        model = KNeighborsRegressor(n_neighbors=len(X), weights=gaussina_kernel)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        

        plt.plot(X_test, y_pred, label="$\gamma$="+str(gamma))
    plt.legend()
    plt.xlabel("input")
    plt.ylabel("output")
    plt.title("KNeighboursRegressor")
    plt.savefig(out_dir + "knn_optimized.png")

    for c in [20]:
        plt.figure()
        plt.scatter(X, y, color='red', label="data")
        for gamma in [1]:
            model = KernelRidge(alpha=1/(2*c), kernel='rbf', gamma=gamma)
            model.fit(X, y)
            y_pred = model.predict(X_test)
            # print("dual_coef", model.dual_coef_)
            plt.plot(X_test, y_pred, label="$\gamma=$"+str(gamma))
        plt.legend()
        plt.xlabel("input")
        plt.ylabel("output")
        plt.title("Kernel ridge regression. C = " + str(c) )
        plt.savefig(out_dir + "kernel_ridge_optimized" + str(c)+ ".png")


def do_kfold(X, y, method, gamma=0, C=1, kfolds=10):
    valid_methods = ["knn", "ridge_regression"]
    if method not in valid_methods:
        raise ValueError("Invalid type")

    def gaussina_kernel(distances):
        weights = np.exp(-gamma * (distances**2))
        return weights / np.sum(weights)

    kf = KFold(n_splits=kfolds)
    mean_error = []
    for train, test in kf.split(X):
        if method == "knn":
            model = KNeighborsRegressor(n_neighbors=len(X[train]), weights=gaussina_kernel)
        else:
            model = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        mean_error.append(mean_squared_error(y[test], y_pred))
    mse = np.array(mean_error).mean()
    std = np.array(mean_error).std()
    # print("MSE = ", mse)
    # print("STD = ", std)
    return model, mse, std



if __name__ == "__main__":

    out_dir = "output_figs/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # For dummy data use these 
    # data  = np.array([[-1,0], [0,1], [1,0]])
    # X = data[:,0].reshape(-1, 1)
    # y = data[:,1].reshape(-1,1)
    
    # Load week 6 dataset
    X, y =read_csv("week6.csv")

    # do k-fold for knn
    mse_list = []
    std_list = []
    gammas = [5, 10, 25, 30, 40, 1000]
    for gamma in gammas:
        model, mse, std = do_kfold(X, y, "knn", gamma)
        mse_list.append(mse)
        std_list.append(std)
        plt.figure()
    plt.xlabel("kernel $\gamma$ parameter")
    plt.ylabel("Mean squared error")
    plt.errorbar(gammas, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/knn_mse_vs_gamma.png")

    # do k-fold for ridge regression
    mse_list = []
    std_list = []
    c_vals = [ 1, 5, 10, 20]
    for C in c_vals:
        model, mse, std = do_kfold(X, y, "ridge_regression", C)
        mse_list.append(mse)
        std_list.append(std)
        plt.figure()
    plt.xlabel("regularisatoin $C$ parameter")
    plt.ylabel("Mean squared error")
    plt.errorbar(c_vals, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/ridge_mse_vs_C.png")

    mse_list = []
    std_list = []
    gammas = [ 1, 2, 3, 4, 5]
    for gamma in gammas:
        model, mse, std = do_kfold(X, y, "ridge_regression", gamma, C=20)
        mse_list.append(mse)
        std_list.append(std)
        plt.figure()
    plt.xlabel("kernel $\gamma$ parameter")
    plt.ylabel("Mean squared error")
    plt.errorbar(gammas, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/ridge_mse_vs_gamma.png")

    knn_and_ridge_regression_kernels(X,y)