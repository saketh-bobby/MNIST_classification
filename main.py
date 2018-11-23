import logistic_model
import deep_neural_network
import random_forest
import ensemble
import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import pickle
import gzip
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import math
from PIL import Image
import os
from sklearn.metrics import confusion_matrix

def get_mnist_usps():
    # load mnist
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()


    X_train = training_data[0]
    y_train = training_data[1]

    enc = OneHotEncoder()
    enc.fit(y_train.reshape(-1, 1))
    y_train_onehot = enc.transform(y_train.reshape(-1, 1)).toarray()

    X_val = validation_data[0]
    y_val = validation_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)


    # load usps
    
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)


    print(np.array(USPSMat).shape)
    print(np.array(USPSTar).shape)


    return X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, USPSMat, USPSTar

def get_data_split(data, split=0.2):
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data

    indices = np.random.choice(range(X_train.shape[0]), size=int(X_train.shape[0]*split))

    X_train = X_train[indices]
    y_train = y_train[indices]
    y_train_onehot = y_train_onehot[indices]

    return X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS


def ensemble_classifier(data):
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data
    print("Performing 20% random sampling for ensemble...")
    get_data_split(data, split=0.2)

    print("Performing logistic regression for ensemble...(300 epochs) ~5 mins")
    y_pred_test_log, y_pred_USPS_log, loss, losses, log_test_acc, log_USPS_acc = logistic_model.logistic_model(data, learning_rate=1, num_epochs=300)

    print("Performing 20% random sampling for ensemble...")
    get_data_split(data, split=0.2)
    print("Running DNN for ensemble...")
    y_pred_test_dnn, y_pred_USPS_dnn, test_acc_dnn, USPS_acc_dnn = deep_neural_network.deep_neural_network(data, learning_rate=0.005, num_epochs=10)

    print("Performing 20% random sampling for ensemble...")
    get_data_split(data, split=0.2)
    print("Running Random Forest for ensemble...")
    y_pred_test_rf, y_pred_USPS_rf, USPS_acc, test_acc = random_forest.random_forest(data, n_estimators=977)

    print("Performing 20% random sampling for ensemble...")
    get_data_split(data, split=0.2)
    print("Running linear SVM for ensemble...")
    y_pred_test_svm_linear, y_pred_USPS_svm_linear = svm.svm_model(data)

    print("Performing 20% random sampling for ensemble...")
    get_data_split(data, split=0.2)
    print("Running SVM with RBF kernel for ensemble...")
    y_pred_test_svm_rbf, y_pred_USPS_svm_rbf = svm.svm_model(data, kernel="rbf", gamma=1)


    np.savetxt("./logistic_mnist_test.csv", y_pred_test_log, delimiter=",", fmt="%u")
    np.savetxt("./logistic_usps_test.csv", y_pred_USPS_log, delimiter=",", fmt="%u")

    np.savetxt("./dnn_mnist_test.csv", np.array(y_pred_test_dnn).T, delimiter=",", fmt="%u")
    np.savetxt("./dnn_usps_test.csv", np.array(y_pred_USPS_dnn).T, delimiter=",", fmt="%u")
    
    np.savetxt("./random_forest_mnist_test.csv", y_pred_test_rf, delimiter=",", fmt="%u")
    np.savetxt("./random_forest_usps_test.csv", y_pred_USPS_rf, delimiter=",", fmt="%u")

    np.savetxt("./svm_linear_mnist_test.csv", y_pred_test_svm_linear, delimiter=",", fmt="%u")
    np.savetxt("./svm_linear_usps_test.csv", y_pred_USPS_svm_linear, delimiter=",", fmt="%u")

    np.savetxt("./svm_rbf_1_mnist_test.csv", y_pred_test_svm_rbf, delimiter=",", fmt="%u")
    np.savetxt("./svm_rbf_1_usps_test.csv", y_pred_USPS_svm_rbf, delimiter=",", fmt="%u")


    y_pred_test_log, y_pred_test_rf, y_pred_test_dnn, y_pred_test_svm_linear, y_pred_test_svm_rbf = [np.array(arr).flatten().tolist() for arr in [y_pred_test_log, y_pred_test_rf, y_pred_test_dnn, y_pred_test_svm_linear, y_pred_test_svm_rbf]]


    y_pred_USPS_log, y_pred_USPS_dnn, y_pred_USPS_rf, y_pred_USPS_svm_linear, y_pred_USPS_svm_rbf = [np.array(arr).flatten().tolist() for arr in [y_pred_USPS_log, y_pred_USPS_dnn, y_pred_USPS_rf, y_pred_USPS_svm_linear, y_pred_USPS_svm_rbf]]

    ensemble_pred_test = ensemble.majority_voting(y_pred_test_log, y_pred_test_dnn, y_pred_test_rf, y_pred_test_svm_linear, y_pred_test_svm_rbf)


    print("Accuracy of ensemble classifier on MNIST testing set is:", ensemble.getAccuracy(np.array(ensemble_pred_test, dtype="int64"), y_test))

    ensemble_pred_USPS = ensemble.majority_voting(y_pred_USPS_log, y_pred_USPS_dnn, y_pred_USPS_rf, y_pred_USPS_svm_linear, y_pred_USPS_svm_rbf)

    print("Accuracy of ensemble classifier on USPS dataset is:", ensemble.getAccuracy(np.array(ensemble_pred_USPS, dtype="int64"), y_USPS))

    # Saving predictions from ensemble classifier
    np.savetxt("./ensemble_mnist_test.csv", ensemble_pred_test, delimiter=",", fmt="%u")
    np.savetxt("./ensemble_mnist_USPS.csv", ensemble_pred_USPS, delimiter=",", fmt="%u")

    # cm = confusion_matrix(ensemble_pred_test, y_test)
    # print(cm)

    # cm = confusion_matrix(ensemble_pred_USPS, y_USPS)
    # print(cm)


def tune_hyperparams_logistic(data, learning_rates, lambdas):

    loss_grid = []
    losses_grid = []
    log_test_acc_grid = []
    log_USPS_acc_grid = []


    for lambd in lambdas:
        for learning_rate in learning_rates:
            print("Logistic regression for lr = %s and lambd = %s" % (learning_rate, lambd))
            _, _, loss, losses, log_test_acc, log_USPS_acc = logistic_model.logistic_model(data, learning_rate=learning_rate, num_epochs=300, lambd=lambd)
            loss_grid.append(loss)
            losses_grid.append(losses)
            log_test_acc_grid.append(log_test_acc)
            log_USPS_acc_grid.append(log_USPS_acc)

    return loss_grid, log_test_acc_grid, log_USPS_acc_grid

def plot_lrs_accuracies_dnn(values, test_accs_grid_dnn, usps_accs_grid_dnn, title, xlabel):
    n_groups = len(values)
    index = np.arange(n_groups)
    bar_width = 0.20
    xticklabels = values
    
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(index, test_accs_grid_dnn, bar_width,
                    color='g',
                    label='Test Accuracy')
    
    rects2 = ax.bar(index + bar_width, usps_accs_grid_dnn, bar_width,
                    color='r',
                    label='USPS Accuracy')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(index + bar_width / 2)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(xticklabels)
    ax.legend()

    fig.tight_layout()
    plt.show()
    

def tune_hyperparams_dnn(data):
    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
    num_neurons = [100, 100, 100]
    
    test_accs_grid_dnn = []
    USPS_accs_grid_dnn = []

    for lr in learning_rates:
        print("Running DNN with learning rate = %s" % lr)
        _, _, test_acc_dnn, USPS_acc_dnn = deep_neural_network.deep_neural_network(data, learning_rate=lr, num_epochs=10, num_neurons=num_neurons)

        test_accs_grid_dnn.append(test_acc_dnn)
        USPS_accs_grid_dnn.append(USPS_acc_dnn)

    plot_lrs_accuracies_dnn(learning_rates, test_accs_grid_dnn, USPS_accs_grid_dnn, title="Number of neurons and Accuracy", xlabel="Number of neurons")

    num_neurons = [[50, 50, 50], [100, 50, 50], [50, 100, 100], [100, 100, 100]]

    for neurons in num_neurons:
        print("Running DNN with num neurons = %s" % neurons)
        _, _, test_acc_dnn, USPS_acc_dnn = deep_neural_network.deep_neural_network(data, learning_rate=0.005, num_epochs=10, num_neurons=neurons)

        test_accs_grid_dnn.append(test_acc_dnn)
        USPS_accs_grid_dnn.append(USPS_acc_dnn)
    
    plot_lrs_accuracies_dnn(num_neurons, test_accs_grid_dnn, USPS_accs_grid_dnn, title="Learning rates and Accuracy", xlabel="Learning rate")

def tune_hyperparams_rf(data):
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 10)]

    max_features = ["auto", "sqrt"]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features
    }

    clf = RandomForestClassifier()
    
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    return rf_random

def plot_losses_accuracies_logistic(learning_rates, lambdas, loss_grid, acc1_grid, acc2_grid):
    n_groups = len(loss_grid)
    index = np.arange(n_groups)
    bar_width = 0.5
    yticklabels = [str((x, y)) for x in learning_rates for y in lambdas]
    
    fig, ax = plt.subplots()
    
    rects1 = ax.barh(index, loss_grid, bar_width,
                    color='b',
                    label='Loss')

    ax.set_title('Loss using Grid Search')
    ax.set_ylabel('(Learning rate, Regularization)')
    ax.set_xlabel('Loss')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(yticklabels)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
    
    fig, ax = plt.subplots()
    
    rects2 = ax.barh(index, acc1_grid, bar_width,
                    color='g',
                    label='Test Accuracy')
    
    rects3 = ax.barh(index + bar_width, acc2_grid, bar_width,
                    color='r',
                    label='USPS Accuracy')

    
    
    ax.set_title('Test and USPS Accuracies using Grid Search')
    ax.set_ylabel('(Learning rate, Regularization)')
    ax.set_xlabel('Accuracy')
    ax.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(yticklabels)
    ax.legend()

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    data = get_mnist_usps()

    # ------------- TUNING HYPERPARAMETERS ------------------------
    
    # Logistic Regression    
    learning_rates = [1, 3, 5, 7]
    lambdas = [0, 0.0005, 0.001]

    loss_grid, log_test_acc_grid, log_USPS_acc_grid = tune_hyperparams_logistic(data, learning_rates, lambdas)

    plot_losses_accuracies_logistic(learning_rates, lambdas, loss_grid, log_test_acc_grid, log_USPS_acc_grid)

    # DNN

    tune_hyperparams_dnn(data)

    # Random Forest

    rf = tune_hyperparams_rf(data)
    print("Best parameters are:", rf.best_params_)

    # ------------------ END OF TUNING ------------------


    # --------------------- VARIOUS ALGORITHMS -------------------------------

    print("Logistic regression...")
    y_pred_test_log, y_pred_USPS_log, loss, losses, log_test_acc, log_USPS_acc = logistic_model.logistic_model(data, learning_rate=1, num_epochs=300)

    print("Deep neural network...")
    y_pred_test_dnn, y_pred_USPS_dnn, test_acc_dnn, USPS_acc_dnn = deep_neural_network.deep_neural_network(data, learning_rate=0.005, num_epochs=10)

    print("Random Forest...")
    y_pred_test_rf, y_pred_USPS_rf, USPS_acc, test_acc = random_forest.random_forest(data)


    print("SVM Linear...")
    y_pred_test_svm, y_pred_USPS_svm = svm.svm_model(data, kernel="linear")

    print("SVM RBF...")
    y_pred_test_svm, y_pred_USPS_svm = svm.svm_model(data, kernel="rbf", gamma=1)

    # -------------------------- END OF ALGORITHMS ---------------------------------


    # -------------------------- ENSEMBLE LEARNING ---------------------------------
    
    ensemble_classifier(data)