import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


def read_csv(filename):
    df = pd.read_csv(filename, comment="#", header=None)
    X1 = np.array(df.iloc[:,0])
    X2 = np.array(df.iloc[:,1])
    X = np.column_stack((X1, X2))
    y = np.array(df.iloc[:,2])
    return X, y


def plot_3d_scatter(x1, x2, y, out_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y)
    ax.set_xlabel('input $x_1$')
    ax.set_ylabel('input $x_2$')
    ax.set_zlabel('output $y$')
    plt.savefig( out_dir + "/data.png")


def train_logistic_with_l2(X, y, C):
    clf = linear_model.LogisticRegression(C=C, random_state=0)
    clf.fit(X,y)
    # print("Coefficients\n", clf.coef_)
    # print("Intercept\n", clf.intercept_)
    return clf


def train_knn_model(X, y, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X,y)
    return clf


def do_kfold(X, y,  method, order=None, C=None, knn=None, kfolds=10):
    valid_methods = ["logistic", "knn"]
    if method not in valid_methods:
        raise ValueError("Invalid type")

    kf = KFold(n_splits=kfolds)
    mean_error = []
    for train, test in kf.split(X):
        if method == "logistic":
            model = train_logistic_with_l2(X[train], y[train], C)
        else:
            model = train_knn_model(X, y, knn)
        y_pred = model.predict(X[test])
        mean_error.append(mean_squared_error(y[test], y_pred))
    mse = np.array(mean_error).mean()
    std = np.array(mean_error).std()
    # print("MSE = ", mse)
    # print("STD = ", std)
    return model, mse, std


def full_pipeline(filename, out_dir):
    X, y = read_csv(filename)

    # split into test and traing sets
    # the test set will NEVER be trained on
    # k folds cross validation is done on training set only for hyperparameter selction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    plot_3d_scatter(X_train[:,0], X_train[:,1], y_train, out_dir)

    # Use cross validation to select polynomial order
    poly_orders = [1, 2, 3, 4, 5]
    mse_list = []
    std_list = []
    for order in poly_orders:
        # add polynomial features up to degree n
        poly = PolynomialFeatures(order)
        X_poly = poly.fit_transform(X_train)
        model, mse, std = do_kfold(X_poly, y_train, method="logistic", order=order, C=1)
        mse_list.append(mse)
        std_list.append(std)

        # evaluate on test set
        X_test_poly = poly.fit_transform(X_test)
        y_pred = model.predict(X_test_poly)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_poly[:,1], X_poly[:,2], y_train, marker="o", s=1, label="training data")
        pos = y_test==1
        neg = y_test==-1
        ax.scatter(X_test_poly[pos,1], X_test_poly[pos,2], y_pred[pos], marker="o", color="g", s=40, label="test (+ve)")
        ax.scatter(X_test_poly[neg,1], X_test_poly[neg,2], y_pred[neg], marker="o",color="r", s=40, label="test (-ve)")
        ax.set_xlabel('input $x_1$')
        ax.set_ylabel('input $x_2$')
        ax.set_zlabel('output $y$')
        plt.legend(loc="upper left")
        plt.savefig(out_dir +  "/logistic_N_" + str(order) + ".png")
    plt.figure()
    plt.xlabel("Polynomial Order ($N$)")
    plt.ylabel("Mean squared error")
    plt.errorbar(poly_orders, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/logistic_mse_vs_N.png")


    # Use cross validation to select C regularisatoin in logistic regression
    c_vals = [0.01, 0.1, 1, 10]
    mse_list = []
    std_list = []
    for c in c_vals:
        # add polynomial features up to degree 2
        poly = PolynomialFeatures(2)
        X_poly = poly.fit_transform(X_train)
        model, mse, std = do_kfold(X_poly, y_train, method="logistic", order=2, C=c)
        mse_list.append(mse)
        std_list.append(std)

        # evaluate on test set
        X_test_poly = poly.fit_transform(X_test)
        y_pred = model.predict(X_test_poly)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_poly[:,1], X_poly[:,2], y_train, marker="o", s=1, label="training data")
        pos = y_test==1
        neg = y_test==-1
        ax.scatter(X_test_poly[pos,1], X_test_poly[pos,2], y_pred[pos], marker="o", color="g", s=40, label="test (+ve)")
        ax.scatter(X_test_poly[neg,1], X_test_poly[neg,2], y_pred[neg], marker="o",color="r", s=40, label="test (-ve)")
        ax.set_xlabel('input $x_1$')
        ax.set_ylabel('input $x_2$')
        ax.set_zlabel('output $y$')
        plt.legend(loc="upper left")
        plt.savefig(out_dir +  "/logistic_N_2" +  "_C_" + str(c) + ".png")
    plt.figure()
    plt.xlabel("L2 Regularisatoin $C$")
    plt.ylabel("Mean squared error")
    plt.errorbar(c_vals, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/logistic_mse_vs_C.png")

    # Use cross validation to select k in knn
    knn_vals = [1, 3, 5, 7, 9]
    mse_list = []
    std_list = []
    for k in knn_vals:
        # # add polynomial features up to degree 1
        # # this has no effect on the features but is necessary to keep consitency
        # poly = PolynomialFeatures(1)
        # X_poly = poly.fit_transform(X_train)
        model, mse, std = do_kfold(X_train, y_train, method="knn", knn=k)
        mse_list.append(mse)
        std_list.append(std)

        # evaluate on test set
        y_pred = model.predict(X_test)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_train[:,0], X_train[:,1], y_train, marker="o", s=1, label="training data")
        pos = y_test==1
        neg = y_test==-1
        ax.scatter(X_test[pos,0], X_test[pos,1], y_pred[pos], marker="o", color="g", s=40, label="test (+ve)")
        ax.scatter(X_test[neg,0], X_test[neg,1], y_pred[neg], marker="o",color="r", s=40, label="test (-ve)")
        ax.set_xlabel('input $x_1$')
        ax.set_ylabel('input $x_2$')
        ax.set_zlabel('output $y$')
        plt.legend(loc="upper left")
        plt.savefig(out_dir +  "/knn_k_" + str(k) + ".png")
    plt.figure()
    plt.xlabel("K nearest Neighbours $k$")
    plt.ylabel("Mean squared error")
    plt.errorbar(knn_vals, mse_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig(out_dir + "/knn_mse_vs_k.png")
    

    # best logisitc model
    poly = PolynomialFeatures(1)
    X_poly = poly.fit_transform(X_train)
    model = train_logistic_with_l2(X_poly, y_train, C=0.1)
    # evaluate on test set
    X_test_poly = poly.fit_transform(X_test)
    y_pred = model.predict(X_test_poly)
    log_fpr, log_tpr, _ = roc_curve(y_test, y_pred)
    print("Best Logistic Model")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # best knn model
    model = train_knn_model(X_train, y_train, k=3)
    y_pred = model.predict(X_test)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, y_pred)
    print("Best Knn Model")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Dummy classifier 
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, y_dummy)
    print("Dummy Model")
    print(confusion_matrix(y_test, y_dummy))
    print(classification_report(y_test, y_dummy))

    # plot ROC curve
    plt.figure()
    plt.plot(log_fpr, log_tpr, label="logistic model")
    plt.plot(knn_fpr, knn_tpr, label="knn model")
    plt.plot(dummy_fpr, dummy_tpr, label="dummy model")
    plt.plot([0, 1], [0, 1], color='k',linestyle='--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(out_dir + "/ROC.png")


if __name__ == "__main__":

    # Just to keep stuff tidy
    # Output directory for each dataset
    out_path_1 = "dataset_1"
    out_path_2 = "dataset_2"
    if not os.path.exists(out_path_1):
        os.makedirs(out_path_1)
    if not os.path.exists(out_path_2):
        os.makedirs(out_path_2)

    full_pipeline("week4-b.csv", out_path_2 )
    full_pipeline("week4-a.csv", out_path_1 )

