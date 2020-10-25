import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def read_csv(filename):
    df = pd.read_csv(filename, comment="#", header=None)
    X1 = np.array(df.iloc[:,0])
    X2 = np.array(df.iloc[:,1])
    X = np.column_stack((X1, X2))
    y = np.array(df.iloc[:,2])
    return X, y


def plot_3d_scatter(x1, x2, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    plt.title(title)
    plt.savefig((title.lower()).replace(" ", "_") + ".png")


def train_lasso_model(X, y, C):
    alpha = 1 / (2*C)
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X,y)
    # print("Coefficients\n", clf.coef_)
    # print("Intercept\n", clf.intercept_)
    return clf


def train_ridge_model(X, y, C):
    alpha = 1 / (2*C)
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X,y)
    print("Coefficients\n", clf.coef_)
    print("Intercept\n", clf.intercept_)
    return clf


def get_test_features(min_val, max_val):
    X_test = []
    grid = np.linspace(min_val, max_val)
    for i in grid:
        for j in grid:
            X_test.append([i,j])
    return np.array(X_test)


def use_kfold(X, y, C, k, regression_type):
    valid_types = ["lasso", "ridge"]
    if regression_type not in valid_types:
        raise ValueError("Invalid reggression type")

    kf = KFold(n_splits=k)
    mean_error = []

    for train, test in kf.split(X):
        if regression_type == "lasso":
            model =  train_lasso_model(X[train], y[train], C)
        else:
            model = train_ridge_model(X[train], y[train], C)
        y_pred = model.predict(X[test])
        mean_error.append(mean_squared_error(y[test], y_pred))
    
    mse = np.array(mean_error).mean()
    std = np.array(mean_error).std()

    print("MSE = ", mse)
    print("STD = ", std)
    return mse, std

    
if __name__ == "__main__":

    X, y = read_csv("week3.csv")
    # plot training data
    plot_3d_scatter(X[:,0], X[:,1], y, title= "Training data")

    # add polynomial features up to degree 5
    poly = PolynomialFeatures(5)
    X_poly = poly.fit_transform(X)

    # train and test models with different regularisation (C) values
    # plot the predictions on a surface plot
    X_test = get_test_features(-3, 3)
    X_test = poly.fit_transform(X_test)

    # Plot the resilts of model predictions along with training data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:,0], X[:,1], y, c="k", label="Training data")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    c_list = [1, 10, 1000]
    for c in c_list:
        clf = train_ridge_model(X_poly, y, c)
        print(X_test.shape)
        y_pred = clf.predict(X_test)
        print(y_pred.shape)
        print(X_test[:,0].shape)
        trisurf = ax.plot_trisurf(X_test[:,1], X_test[:,2], y_pred, label="C="+str(c))
        trisurf._facecolors2d=trisurf._facecolors3d
        trisurf._edgecolors2d=trisurf._edgecolors3d
    plt.legend()
    plt.savefig("ridge_regression_models.png")


    # Perfrom k-fold cross validation
    k_values = [2, 5, 10, 25, 50, 100]
    mse_list = []
    std_list = []
    for k in k_values:
        mse, std = use_kfold(X_poly, y, C=1, k=k, regression_type="lasso")
        mse_list.append(mse)
        std_list.append(std)

    plt.figure()
    plt.errorbar(k_values, mse_list, yerr=std_list, ecolor="r", capsize=5)
    plt.xlabel("k")
    plt.ylabel("Mean squared error")
    plt.savefig("lasso_kfold_errorbars.png")


    # Plot MSE vs C using kfolds
    mse_list = []
    std_list = []
    c_list = [1, 2.5, 5, 10, 25]
    for c in c_list:
        mse, std = use_kfold(X_poly, y, C=c, k=10, regression_type="ridge")
        mse_list.append(mse)
        std_list.append(std)

    plt.figure()
    plt.errorbar(c_list, mse_list, yerr=std_list, ecolor="r", capsize=5)
    plt.xlabel("C")
    plt.ylabel("Mean squared error")
    plt.savefig("ridge_mse_vs_C.png")



