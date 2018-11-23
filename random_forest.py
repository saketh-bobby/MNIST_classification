import pickle
import gzip
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
from PIL import Image
import numpy as np


def random_forest(data, n_estimators=100):
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data

    

    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    y_train_pred = clf.predict(X_train)

    y_val_pred = clf.predict(X_val)

    y_test_pred = clf.predict(X_test)

    y_pred_USPS = clf.predict(X_USPS)


    print("Training Accuracy is:",metrics.accuracy_score(y_train, y_train_pred) * 100)
    print("Validation Accuracy is:",metrics.accuracy_score(y_val, y_val_pred) * 100)

    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy is:", test_acc * 100)

    USPS_acc = metrics.accuracy_score(y_USPS, y_pred_USPS)
    print("USPS Numerals Accuracy is:", USPS_acc * 100)

    # cm = confusion_matrix(y_test_pred, y_test)
    # print(cm)

    # cm = confusion_matrix(y_pred_USPS, y_USPS)
    # print(cm)


    return y_test_pred, y_pred_USPS, USPS_acc, test_acc