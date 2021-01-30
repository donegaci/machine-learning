import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib.ticker import MaxNLocator

def text_pre_processing(X, min_df, max_df, n_gram=1):
    vectorizer = TfidfVectorizer(norm=None, stop_words="english",  min_df=min_df, max_df=max_df, ngram_range=(1,n_gram))
    X_vectors = vectorizer.fit_transform(X)
    print("vectorizer.stop_words_")
    print(len(vectorizer.stop_words_))
    print("vectorizer.get_feature_names()")
    print(len(vectorizer.get_feature_names()))
    return vectorizer, X_vectors.toarray()


def do_kfold(X, y,  method, order=None, C=None, kfolds=5):
    valid_methods = ["logistic", "svm", "bayes"]
    if method not in valid_methods:
        raise ValueError("Invalid type")

    kf = KFold(n_splits=kfolds)
    mean_error = []
    for train, test in kf.split(X):
        if method == "logistic":
            model = train_logistic_with_l2(X[train], y[train], C)
        elif method =="bayes":
            model = train_naive_bayes(X, y)
        else:
            model = train_svm(X, y, C)
        y_pred = model.predict(X[test])
        mean_error.append(accuracy_score(y[test], y_pred))
    acc = np.array(mean_error).mean()
    std = np.array(mean_error).std()
    return model, acc, std

def train_logistic_with_l2(X, y, C):
    clf = LogisticRegression(C=C, random_state=0)
    clf.fit(X,y)
    return clf

def train_svm(X, y, C):
    clf = SVC(kernel="linear", random_state=0, C=C)
    clf.fit(X, y)
    return clf

def train_naive_bayes(X, y):
    clf = GaussianNB()
    clf.fit(X, y)
    return clf


# Define some useful functions
class PlotLossAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(int(self.i))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        plt.figure()
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Model Loss')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("training_curve.eps", format="eps")


def full_pipeline(filename):

    with open(filename, "rb") as f:
        X,y,z = pickle.load(f)

    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)

    print(sum(z==True))

    # Split into 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.1, random_state=10)

    # Use cross val to select preprocessing parameters
    # Select min_df (very rare terms)
    min_df_list = [0.0025, 0.005, 0.01, 0.015] 
    acc_list = []
    std_list = []
    for min_df in min_df_list:
        _, X_vectors= text_pre_processing(X_train, min_df, max_df=1.0)
        model, acc, std = do_kfold(X_vectors, y_train, method="svm", order=1, C=1)
        acc_list.append(acc)
        std_list.append(std)
    plt.figure()
    plt.xlabel("Min_df")
    plt.ylabel("Accuracy")
    plt.errorbar(min_df_list, acc_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.show()

    # Select max_df (very common terms)
    max_df_list = [0.09, 0.1, 0.11, 1.0] 
    acc_list = []
    std_list = []
    for max_df in max_df_list:
        _, X_vectors= text_pre_processing(X_train, min_df=0.0025, max_df=max_df)
        model, acc, std = do_kfold(X_vectors, y_train, method="svm", order=1, C=1)
        acc_list.append(acc)
        std_list.append(std)
    plt.figure()
    plt.xlabel("Max_df")
    plt.ylabel("Accuracy")
    plt.errorbar(max_df_list, acc_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.show()

    # Use cross val to select ngram parameter
    n_gram_list = [1,2] 
    acc_list = []
    std_list = []
    for n in n_gram_list:
        _, X_vectors= text_pre_processing(X_train, 0.001, max_df=1.0, n_gram=n)
        model, acc, std = do_kfold(X_vectors, y_train, method="svm", order=1, C=1)
        acc_list.append(acc)
        std_list.append(std)
    plt.figure()
    plt.xlabel("n_gram range")
    plt.ylabel("Accuracy")
    plt.errorbar(n_gram_list, acc_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.savefig("n_gram_cross_val.eps", format='eps')

    # Use cross validation to select C regularisatoin in logistic regression
    c_vals = [ 0.001, 0.01, 0.1]
    acc_list = []
    std_list = []
    _, X_vectors_train = text_pre_processing(X_train, min_df=0.01, max_df=1.0)
    for c in c_vals:

        model, acc, std = do_kfold(X_vectors_train, y_train, method="logistic", order=2, C=c)
        acc_list.append(acc)
        std_list.append(std)

    plt.figure()
    plt.xlabel("L2 Regularisatoin $C$")
    plt.ylabel("Accuracy")
    plt.errorbar(c_vals, acc_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    plt.show()

    # Use cross validation to select C in SVM
    c_vals = [0.01, 0.1, 1]
    acc_list = []
    std_list = []
    _, X_vectors_train = text_pre_processing(X_train, min_df=0.0025, max_df=1.0)
    for c in c_vals:

        model, acc, std = do_kfold(X_vectors_train, y_train, method="svm", C=c)
        print(acc )
        acc_list.append(acc)
        std_list.append(std)

    plt.figure()
    plt.xlabel("Linear SVM Regularisatoin $C$")
    plt.ylabel("Accuracy")
    plt.errorbar(c_vals, acc_list, yerr=std_list, fmt="-o", ecolor="r", capsize=5)
    # plt.show()

    vectorizer, X_vectors_train = text_pre_processing(X_train, min_df=0.001, max_df=1.0, n_gram=2)
    X_vectors_test = vectorizer.transform(X_test).toarray()
    print(X_vectors_train.shape)
    print(X_vectors_test.shape)

    # Logistic model classifier 
    model = train_logistic_with_l2(X_vectors_train, y_train, C=0.01)
    y_pred = model.predict(X_vectors_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    log_fpr, log_tpr, _ = roc_curve(y_test, y_pred)


    # SVM model classifier 
    model = train_svm(X_vectors_train, y_train, C=1)
    y_pred = model.predict(X_vectors_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    svm_fpr, svm_tpr, _ = roc_curve(y_test, y_pred)

    # fully connected NN
    inputs = keras.layers.Input(shape=len(vectorizer.get_feature_names()))
    x = inputs
    x = Dense(500, activation='relu', )(x)
    x = Dense(50, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    # we create the model 
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # we create a callback function to plot our loss function and accuracy
    pltCallBack = PlotLossAccuracy()
    # and train
    model.fit(X_vectors_train, y_train,
            batch_size=256, epochs=40,  
            callbacks=[pltCallBack])
    y_pred = (model.predict(X_vectors_test) > 0.5).astype("int32")
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    nn_fpr, nn_tpr, _ = roc_curve(y_test, y_pred)

    # Dummy classifier 
    dummy = DummyClassifier(strategy="most_frequent").fit(X_vectors_train, y_train)
    y_dummy = dummy.predict(X_vectors_test)
    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, y_dummy)
    print("Dummy Model")
    print(confusion_matrix(y_test, y_dummy))
    print(classification_report(y_test, y_dummy))
    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, y_dummy)

    # plot ROC curve
    plt.figure()
    plt.plot(log_fpr, log_tpr, label="logistic model")
    plt.plot(svm_fpr, svm_tpr, label="SVM model")
    plt.plot(nn_fpr, nn_tpr, label="NN model")
    plt.plot(dummy_fpr, dummy_tpr, label="dummy model")
    plt.plot([0, 1], [0, 1], color='k',linestyle='--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig("ROC.eps", format="eps")


if __name__ == "__main__":

    full_pipeline("cleaned.pickle")

