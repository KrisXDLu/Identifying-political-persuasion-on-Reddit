from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv
import random

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    accurate = 0
    total = 0
    it = np.nditer(C, flags=['multi_index'])
    while not it.finished:
      if it.multi_index[0] == it.multi_index[1]:
        accurate += it[0]
      total += it[0]
      it.iternext()
    if total == 0:
      return 0
    else:
      return accurate/total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_list = []
    for i in range(np.shape(C)[0]):
      if np.sum(C[i]) == 0:
        recall_list.append(0)
      else:
        recall_list.append(C[i][i]/np.sum(C[i]))
    return recall_list

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_list = []
    sum_list = np.sum(C, axis=0)
    for j in range(np.shape(C)[1]):
      if sum_list[j] == 0:
        precision_list.append(0)
      else:
        precision_list.append(C[j][j]/sum_list[j])
    return precision_list

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
    # split data
    data = np.load(filename)
    x = data['arr_0'][:, :173]
    y = data['arr_0'][:, 173:]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    # svc with linear kernal
    linear_clf = LinearSVC(max_iter=10000)
    linear_clf.fit(X_train, y_train)
    linear_test = linear_clf.predict(X_test)
    linear_cm = confusion_matrix(y_test, linear_test, labels=[0, 1, 2, 3])
    # linear_cm = np.array([[443,24,60,1459],[188,53,160,1566],[179,48,338,1482],[181,41,164,1614]])
    # print(linear_cm)
    # print(accuracy(linear_cm))
    # svc with rbf kernal
    radial_clf = SVC(gamma = 'auto')
    radial_clf.fit(X_train, y_train)
    radial_test = radial_clf.predict(X_test)
    radial_cm = confusion_matrix(y_test, radial_test, labels=[0, 1, 2, 3])
    # radial_cm = np.array([[649,269,104,964],[305,416,217,1029],[279,318,450,1000],[284,345,216,1155]])
    # print(radial_cm)
    # print(accuracy(radial_cm))
    # random forest classifier
    rfc_clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    rfc_clf.fit(X_train, y_train)
    rfc_test = rfc_clf.predict(X_test)
    rfc_cm = confusion_matrix(y_test, rfc_test, labels=[0, 1, 2, 3])
    # rfc_cm = np.array([[1172,139,272,403],[315,461,627,564],[262,280,1068,437],[297,351,583,129]])
    # print(rfc_cm)
    # print(accuracy(rfc_cm))
    # neural network classifier
    nn_clf = MLPClassifier(alpha=0.05)
    nn_clf.fit(X_train, y_train)
    nn_test = nn_clf.predict(X_test)
    nn_cm = confusion_matrix(y_test, nn_test, labels=[0, 1, 2, 3])
    # nn_cm = np.array([[1330,380,239,37],[438,1070,395,64],[429,817,712,89],[503,929,439,129]])
    # print(nn_cm)
    # print(accuracy(nn_cm))
    # AdaBoostClassifier
    abc_clf = AdaBoostClassifier()
    abc_clf.fit(X_train, y_train)
    abc_test = abc_clf.predict(X_test)
    abc_cm = confusion_matrix(y_test, abc_test, labels=[0, 1, 2, 3])
    # abc_cm = np.array([[1220,186,217,363],[323,609,508,527],[266,361,985,437],[338,402,488,772]])
    # print(abc_cm)
    # print(accuracy(abc_cm))
    classifier_list = [linear_cm, radial_cm, rfc_cm, nn_cm, abc_cm]
    with open('a1_3.1.csv', 'w', newline='') as csvfile:
      spamwriter = csv.writer(csvfile)
      for i in range(5):
        c_classifier =classifier_list[i]
        row = [i+1]+[accuracy(c_classifier)]+recall(c_classifier)+precision(c_classifier)+np.ravel(c_classifier).tolist()
        spamwriter.writerow(row)
    best_accurancy = 0
    for i in range(len(classifier_list)):
      c_clf = classifier_list[i]
      if accuracy(c_clf) > best_accurancy:
        best_accurancy = accuracy(c_clf)
        iBest = i+1
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
    # find the best accuracy classifier
    if iBest == 1:
      clf = LinearSVC(max_iter=10000)
    elif iBest == 2:
      clf = SVC(gamma = 'auto')
    elif iBest == 3:
      clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif iBest == 4:
      clf = MLPClassifier(alpha=0.05)
    else:
      clf = AdaBoostClassifier()

    random.seed(1)
    # find the train list of random 1k, 5k, 10k, 15k, 20k
    train_size = {1000, 5000, 10000, 15000, 20000}
    accuracy_list = []
    for c_size in train_size:
      index = random.sample(range(np.shape(X_train)[0]), k=c_size)
      X_index = np.array([X_train[x] for x in index])
      y_index = np.array([y_train[x] for x in index])
      clf.fit(X_index, y_index)
      y_pred = clf.predict(X_test)
      cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
      accuracy_list.append(accuracy(cm))

    with open('a1_3.2.csv', 'w', newline='') as csvfile:
      spamwriter = csv.writer(csvfile)
      spamwriter.writerow(accuracy_list)

    # seed?
    index_1k = random.sample(range(np.shape(X_train)[0]), k=1000)
    X_1k = np.array([X_train[x] for x in index_1k])
    y_1k = np.array([y_train[x] for x in index_1k])

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
    # 3.3.1
    k_list = {5, 10, 20, 30, 40, 50}
    best_features_1k = {}
    p_value_1k = {}
    for c_k in k_list:
      c_feature = []
      selector = SelectKBest(f_classif, k=c_k)
      X_1k_new = selector.fit_transform(X_1k, y_1k)
      pp_1k = selector.pvalues_
      mask_1k = selector.get_support()
      for i in range(len(mask_1k)):
        if mask_1k[i]:
          c_feature.append(i)
      best_features_1k[c_k] = c_feature
      sorted_pp = sorted(selector.pvalues_)
      p_value_1k[c_k] = sorted_pp[:v_k]

    best_features_32k = {}
    rows = []
    p_value_32k = {}
    for c_k in k_list:
      c_feature = []
      selector = SelectKBest(f_classif, k=c_k)
      X_new = selector.fit_transform(X_train, y_train)
      pp = selector.pvalues_
      mask = selector.get_support()
      for i in range(len(mask)):
        if mask[i]:
          c_feature.append(i)
      best_features_32k[c_k] = c_feature
      sorted_pp = sorted(selector.pvalues_)
      p_value_32k[c_k] = sorted_pp[:v_k]
      rows.append([c_k] + sorted_pp[:v_k])

    # 3.3.2
    # feature_1k = np.zeros(X_1k[1])
    # for i in range(5):
    #   feature_1k[best_features_1k[5][i]][i] = 1
    # x_feature_1k = np.dot(X_1k, feature_1k)
    selector = SelectKBest(f_classif, k=5)
    x_feature_1k = selector.fit_transform(X_1k, y_1k)

    # feature_32k = np.zeros(X_32k[1])
    # for i in range(5):
    #   feature_32k[best_features_32k[5][i]][i] = 1
    # x_feature_32k = np.dot(X_train, feature_32k)
    selector = SelectKBest(f_classif, k=5)
    x_feature_32k = selector.fit_transform(X_train, y_train)

    if iBest == 1:
      clf = LinearSVC(max_iter=10000)
    elif iBest == 2:
      clf = SVC(gamma = 'auto')
    elif iBest == 3:
      clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif iBest == 4:
      clf = MLPClassifier(alpha=0.05)
    else:
      clf = AdaBoostClassifier()

    accuracy_list = []
    clf.fit(x_feature_1k, y_train)
    y_feature_1k = clf.predict(X_test)
    cm_feature_1k = confusion_matrix(y_test, y_feature_1k, labels=[0, 1, 2, 3])
    accuracy_list.append(accuracy(cm_feature_1k))

    clf.fit(x_feature_32k, y_train)
    y_feature_32k = clf.predict(X_test)
    cm_feature_32k = confusion_matrix(y_test, y_feature_32k, labels=[0, 1, 2, 3])
    accuracy_list.append(accuracy(cm_feature_32k))

    rows.append(accuracy_list)

    # 3.3.3
    print(x_feature_1k)
    print(x_feature_32k)
    print(p_value_1k)
    print(p_value_32k)
    with open('a1_3.3.csv', 'w', newline='') as csvfile:
      spamwriter = csv.writer(csvfile)
      spamwriter.writerows(rows)


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    
if __name__ == "__main__":
    # parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    # args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    (X_train, X_test, y_train, y_test, iBest) = class31('part2_result.npz')
    (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)