from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.features_selection import f_classif
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
import csv
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    dividend = np.sum(np.sum(C, axis=1))
    return np.true_divide(np.sum(np.diag(C)), dividend, where=dividend!=0)

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    dividend = np.sum(C, axis=1)
    return np.true_divide(np.diag(C), dividend, where=dividend!=0).tolist()

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    dividend = np.sum(C, axis=0)
    return np.true_divide(np.diag(C), dividend, where=dividend!=0).tolist()

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    iBest = 0
    confMtrx = []
    # load data
    data = np.load(filename)
    data = data['arr_0']

    # getting y value
    X = data[:,:-1]
    y = data[:,-1]

    # splitting data into test and training 20%,80% 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    
    # support vector machine with linear kernel
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    confMtrx.append(confusion_matrix(predict, y_test))

    # radial basis func=2 kernel
    model = SVC(max_iter=10000, gamma=2)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    confMtrx.append(confusion_matrix(predict, y_test))

    # RandForest with a maximum depth of 5, and 10 estimators
    model = RandomForestClassifier(max_depth=5, n_estimators=10)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    confMtrx.append(confusion_matrix(predict, y_test))

    # MLP Classifier A feed-forward neural network, with α = 0.05
    model = MLPClassifier(alpha=0.05)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    confMtrx.append(confusion_matrix(predict, y_test))

    # AdaBoost default
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    confMtrx.append(confusion_matrix(predict, y_test))

    bestAccuracy = -1

    with open('a1_3.1.csv', 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',')
        for i in range(len(confMtrx)):
            C = confMtrx[i]
            accuracyI = accuracy(C)
            if accuracyI > bestAccuracy:
                bestAccuracy = accuracyI
                iBest = i+1
            csvWriter.writerow([i+1] + [accuracyI] + recall(C) + np.ravel(C).tolist())
    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    dataSize = [1000, 5000, 10000, 15000, 20000]
    # choose best model
    if iBest == 1:
        model = LinearSVC(max_iter=10000)
    elif iBest == 2:
        model = SVC(max_iter=10000, gamma=2)
    elif iBest == 3:
        model = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif iBest == 4:
        model = MLPClassifier(alpha=0.05)
    else:
        model = AdaBoostClassifier()

    # accuracy
    accur = []
    for size in dataSize:
        model.fit(X_train[:size], y_train[:size])
        C = confusion_matrix(model.predict(X_test), y_test)
        accur.append(accuracy(C))
    with open('a1_3.2.csv', "w", newline='') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',')
        csvWriter.writerow(accur)

    X_1k = np.array(X_train[:dataSize[0]]) 
    y_1k = np.array(y_train[:dataSize[0]])
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    k = {5, 10, 20, 30, 40, 50}
    
    X_new = selector.fit_transform(X_train, y_train)
    pp = selector.pvalues_
    for i in k:
        selector = SelectKBest(f_classif, k) 

    # print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)



