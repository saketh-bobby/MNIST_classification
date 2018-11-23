import pickle
import gzip
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import math
from PIL import Image
import os

def softmax(h_x):
    exps = np.exp(h_x - np.max(h_x))
    return exps / np.sum(exps, axis=1).reshape((exps.shape[0], 1))

def calculate_loss(X, y, W, lambd):
    h_x = np.array(np.dot(X, W), dtype=np.float32)
    Z = softmax(h_x)

    m = X.shape[0]
    loss = (-1/m) * (np.sum(y * np.log(Z))) + (lambd/2)*np.sum(W*W)
    
    gradient = (-1 / m)* np.dot(X.T, (y - Z)) + lambd * W
    
    return loss, gradient

def random_mini_batches(X, Y, mini_batch_size = 256, seed = 42):
    
    np.random.seed(seed)            
    m = X.shape[0]           
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def mini_batch_gradient_descent(mini_batches, W, num_epochs, learning_rate, lambd, verbose=False):
    losses = []
    for epoch in range(num_epochs):
        if verbose and (epoch + 1) % 5 == 0:
            print("Epoch %d" % (epoch+1))
        for batch in mini_batches:
            X_batch = batch[0]
            y_batch = batch[1]
            n = X_batch.shape[0]

            W_batch = np.random.uniform(-0.2, 0.2, size=(X_batch.shape[1], 10))

            h_x = np.array(np.dot(X_batch, W_batch), dtype=np.float32)
            Z = softmax(h_x)

            loss, gradient = calculate_loss(X_batch, y_batch, W, lambd=lambd)

            W = W - learning_rate * gradient

        loss, gradient = calculate_loss(X_batch, y_batch, W, lambd=lambd)
        losses.append(loss)

    return W, losses

def getAccuracy(prede,someY):
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

def get_prediction(X, W):
    h_x = np.array(np.dot(X, W), dtype=np.float32)
    Z = softmax(h_x)
    y_pred = np.argmax(Z, axis=1)

    return y_pred

def normalize(X):
    max = np.max(X, axis = 1)
    max = max.reshape((X.shape[0], 1))

    min = np.min(X, axis = 1)
    min = min.reshape((X.shape[0], 1))

    return (X - min) / (max - min)


def logistic_model(data,learning_rate=0.1, num_epochs=50, lambd=0):
    
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data

    X_train = normalize(X_train)
    W = np.random.uniform(-0.2, 0.2, size=(X_train.shape[1], 10))

    mini_batches = random_mini_batches(X_train, y_train_onehot)

    W, losses = mini_batch_gradient_descent(mini_batches=mini_batches, W=W, num_epochs=num_epochs, learning_rate=0.05, lambd=lambd)

    loss, _ = calculate_loss(X_train, y_train_onehot, W, lambd=lambd)
    print("Loss =", loss)

    y_pred_train = get_prediction(X_train, W)
    print("MNIST training accuracy is:", getAccuracy(y_pred_train, y_train) * 100)

    y_pred_val = get_prediction(X_val, W)
    print("MNIST validation accuracy is:", getAccuracy(y_pred_val, y_val) * 100)


    y_pred_test = get_prediction(X_test, W)

    log_test_acc = getAccuracy(y_pred_test, y_test)
    print("MNIST testing accuracy is:", log_test_acc * 100)

    y_pred_USPS = get_prediction(X_USPS, W)

    log_USPS_acc = getAccuracy(y_pred_USPS, y_USPS)
    print("USPS testing accuracy is:", log_USPS_acc * 100)

    # cm = confusion_matrix(y_pred_test, y_test)
    # print(cm)

    # cm = confusion_matrix(y_pred_USPS, y_USPS)
    # print(cm)


    return y_pred_test, y_pred_USPS, loss, losses, log_test_acc, log_USPS_acc