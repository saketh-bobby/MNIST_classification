import pickle
import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier



def svm_model(data, kernel="linear", gamma="auto", max_iter=-1):
    X_train, y_train, y_train_onehot, X_val, y_val, X_test, y_test, X_USPS, y_USPS = data

    model = svm.SVC(kernel=kernel, gamma=gamma, max_iter=max_iter)

    model.fit(X_train, y_train)

    # y_pred_train = model.predict(X_train)
    # y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    y_pred_USPS = model.predict(X_USPS)

    # print("Training Accuracy is:", accuracy_score(y_train, y_pred_train))
    # print("Validation Accuracy is:", accuracy_score(y_val, y_pred_val))

    print("Testing Accuracy is:", accuracy_score(y_test, y_pred_test) * 100)

    print("USPS Accuracy is:", accuracy_score(y_USPS, y_pred_USPS) * 100)

    # cm = confusion_matrix(y_pred_test, y_test)
    # print(cm)

    # cm = confusion_matrix(y_pred_USPS, y_USPS)
    # print(cm)


    return y_pred_test, y_pred_USPS

