import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def read_csv(filename):
    df = pd.read_csv(filename, comment="#", header=None)
    X1 = np.array(df.iloc[:,0])
    X2 = np.array(df.iloc[:,1])
    X = np.column_stack((X1, X2))
    print(X)
    y = np.array(df.iloc[:,2])
    return X, y


def visualise_data(X, y):
    pos_classes = X[y == 1]
    neg_classes = X[y == -1]
    plt.figure(1)
    plt.scatter(pos_classes[:,0], pos_classes[:,1], marker="+")
    plt.scatter(neg_classes[:,0], neg_classes[:,1], marker="o")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.title("Training data")
    plt.savefig("training_data.png")

def logistic_classifier(X, y):
    # Train a logistic classifier on data
    clf = LogisticRegression(random_state=0)
    clf.fit(X, y)
    print("classes = ", clf.classes_)
    print("intercept = ", clf.intercept_)
    print("coefficients = ", clf.coef_)

    # Use classifier to predict targets on training data
    preds = clf.predict(X)
    pred_pos = X[preds==1]
    pred_neg = X[preds==-1]
    print("# pred_pos ", len(pred_pos))
    print("# pred_neg ", len(pred_neg))

    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    # Calculate the intercept and slope of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot predictions along with decision boundary
    decision_bnd = m * X[:,0] + c
    plt.figure(1)
    plt.plot(X[:,0], decision_bnd)
    plt.scatter(pred_pos[:,0], pred_pos[:,1], marker="+")
    plt.scatter(pred_neg[:,0], pred_neg[:,1], marker="o")
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.title("Logistic Regression predictions")
    plt.legend(["Decision Boundary", "+ class", "- class", "Predicted +", "Predicted -"])
    plt.savefig("logistic_train_and_predicted.png")


def linear_svm(X, y):

    for i, C in enumerate([0.001, 1, 10, 100, 1000]):
        print("C =" , C)
        clf = LinearSVC(random_state=0, C=C, max_iter=90000)
        clf.fit(X, y)

        print("intercept = ", clf.intercept_)
        print("coefficients = ", clf.coef_)

        # Retrieve the model parameters.
        b = clf.intercept_[0]
        w1, w2 = clf.coef_.T
        # Calculate the intercept and slope of the decision boundary.
        c = -b/w2
        m = -w1/w2

        # Use classifier to predict targets on training data
        preds = clf.predict(X)
        pred_pos = X[preds==1]
        pred_neg = X[preds==-1]
        print("# pred_pos ", len(pred_pos))
        print("# pred_neg ", len(pred_neg))

        # Plot training data
        pos_classes = X[y == 1]
        neg_classes = X[y == -1]
        plt.figure()
        plt.scatter(pos_classes[:,0], pos_classes[:,1], marker="+")
        plt.scatter(neg_classes[:,0], neg_classes[:,1], marker="o")
        # plot predictions
        plt.scatter(pred_pos[:,0], pred_pos[:,1], marker="+")
        plt.scatter(pred_neg[:,0], pred_neg[:,1], marker="o")
        # plot decision boundary 
        decision_bnd = m * X[:,0] + c
        plt.plot(X[:,0], decision_bnd)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.title("Linear SVM predictions with C = " + str(C))
        plt.legend(["Decision Boundary", "+ class", "- class", "Predicted +", "Predicted -"])
        plt.savefig("svm_train_and_predicted_c_" + str(C) + ".png")



    


if __name__ == "__main__":

    X, y = read_csv("week2.csv")

    visualise_data(X, y)

    logistic_classifier(X,y)

    linear_svm(X, y)
