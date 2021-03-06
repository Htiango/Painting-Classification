import time
from sklearn import svm
import numpy as np
import sklearn

def predict_from_features(X, y, kernel='linear'):
    """ Given labels and VGG features, predict the breed of the testing set. 
    Args: 
        X (numpy ndarray) : 2D array of VGG features, each row is a set of features for a single example
        y (numpy ndarray) : 1D array of labels corresponding to the features in X
    Returns: 
        (numpy ndarray) 1D array of predicted labels for the unlabeled examples in X_te
    """
    clf = svm.SVC(kernel=kernel)
    clf.fit(X, y)
    return clf
    # return clf.predict(X_te)

X_tr = np.loadtxt("../data/features/VGG_X_tr.txt")
X_va = np.loadtxt("../data/features/VGG_X_va.txt")
X_te = np.loadtxt("../data/features/VGG_X_te.txt")

y_tr = np.loadtxt("../data/features/VGG_Y_tr.txt", dtype=int)
y_va = np.loadtxt("../data/features/VGG_Y_va.txt", dtype=int)
y_te = np.loadtxt("../data/features/VGG_Y_te.txt", dtype=int)

print("data loaded")

# choosing realism and abstract
X_tr = X_tr[(y_tr==5) | (y_tr==6)]
y_tr = y_tr[(y_tr==5) | (y_tr==6)]
y_tr[(y_tr==5)] = 0
y_tr[(y_tr==6)] = 1
print(X_tr.shape[1])
print("Training number is: " + str(X_tr.shape[0]))
print(y_tr.shape[0])

X_va = X_va[(y_va==5) | (y_va==6)]
y_va = y_va[(y_va==5) | (y_va==6)]
y_va[(y_va==5)] = 0
y_va[(y_va==6)] = 1
print("Validation number is: " + str(X_va.shape[0]))
print(y_va.shape[0])

X_te = X_te[(y_te==5) | (y_te==6)]
y_te = y_te[(y_te==5) | (y_te==6)]
y_te[(y_te==5)] = 0
y_te[(y_te==6)] = 1
print("Testing number is: " + str(X_te.shape[0]))
print(y_te.shape[0])


# processing data: transform it into western and eastern styles
# X_tr = X_tr[(y_tr==0) | (y_tr==2) | (y_tr==5) | (y_tr==8)]
# y_tr = y_tr[(y_tr==0) | (y_tr==2) | (y_tr==5) | (y_tr==8)]
# y_tr[(y_tr==2)] = 0
# y_tr[(y_tr==5) | (y_tr==8)] = 1
# print("Training number is: " + str(X_tr.shape[0]))

# X_va = X_va[(y_va==0) | (y_va==2) | (y_va==5) | (y_va==8)]
# y_va = y_va[(y_va==0) | (y_va==2) | (y_va==5) | (y_va==8)]
# y_va[(y_va==2)] = 0
# y_va[(y_va==5) | (y_va==8)] = 1
# print("Validation number is: " + str(X_va.shape[0]))

# X_te = X_te[(y_te==0) | (y_te==2) | (y_te==5) | (y_te==8)]
# y_te = y_te[(y_te==0) | (y_te==2) | (y_te==5) | (y_te==8)]
# y_te[(y_te==2)] = 0
# y_te[(y_te==5) | (y_te==8)] = 1
# print("Testing number is: " + str(X_te.shape[0]))

print("Finish selecting")



best_model = None
acc = 0

for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    start = time.time()
    model = predict_from_features(X_tr, y_tr, kernel)
    y_p = model.predict(X_va)
    acc_model = np.mean(y_p==y_va)
    end = time.time()
    print("kernel: " + kernel)
    print("SVM Training accuracy: {}".format(np.mean(model.predict(X_tr)==y_tr)))
    print("SVM Validation accuracy: {} in {} seconds".format(acc_model, end-start))
    print("SVM Testing accuracy: {}".format(np.mean(model.predict(X_te)==y_te)))
    print()
    if acc < acc_model:
    	acc = acc_model
    	best_model = model

y_p_te = best_model.predict(X_te)
print("SVM Testing accuracy: {}".format(np.mean(y_p_te==y_te)))

logreg = sklearn.linear_model.LogisticRegression(C=1e5)
logreg.fit(X_tr, y_tr)
y_p = logreg.predict(X_va)
acc_model = np.mean(y_p==y_va)
print("SVM Training accuracy: {}".format(np.mean(logreg.predict(X_tr)==y_tr)))
print("Logistic Validation accuracy: {}".format(acc_model))
y_p_te = logreg.predict(X_te)
acc_model_te = np.mean(y_p_te==y_te)
print("Logistic Testing accuracy: {}".format(acc_model_te))

