import pickle
import gzip
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os
import pandas as pd
from collections import Counter

def majority_voting(logistic, dnn, rf, svm_linear, svm_rbf):
    new_pred = []
    for pred_vector in zip(logistic, dnn, rf, svm_linear, svm_rbf):
        a = Counter(pred_vector).most_common(1)[0]
        new_pred.append(a[0])

    return new_pred

def getAccuracy(pred,labels):
    a = (pred == labels)

    accuracy = np.sum(a)/(float(len(labels)))
    return accuracy * 100
