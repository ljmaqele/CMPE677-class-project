# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:23:17 2020

@author: Lefu
"""
import numpy as np
import pandas as pd
import sklearn.linear_model as sl
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from imblearn.over_sampling import SMOTE


class classify:
    
    def __init__(self, X, y, options):
        self.X = X
        self.y = y
        self.options = options
        
    def metrics(self):
        X_balance, Y_balance = SMOTE().fit_sample(self.X, self.y) # balance the data
        n = len(Y_balance)
        cv = self.options['CV']
        np.random.seed(self.options['seed'])
        num_rep = n//cv
        identifiers = list(np.repeat(np.arange(1, cv+1, 1), num_rep)) + list(np.arange(1, n%cv+1, 1))
        np.random.shuffle(identifiers) # shuffle the identifiers
        
        predictedClasses = np.zeros(n) # initiate predicted classes
        train_accuracy = []
        for i in np.arange(1, cv+1, 1):
            train_index = np.where(identifiers != i)
            test_index = np.where(identifiers == i)
            Xtrain = X_balance[train_index, :][0]
            ytrain = Y_balance[train_index]
            Xtest = X_balance[test_index, :][0]
            ytest = Y_balance[test_index]
            
            if self.options['method'] == 'LogisticRegression':
                clf = sl.LogisticRegression(solver='lbfgs').fit(Xtrain, ytrain)
                yt = clf.predict(Xtrain)
                ac = accuracy_score(ytrain, yt)
                predictions = clf.predict(Xtest)
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'SVM':
                clf = SVC(gamma='auto').fit(Xtrain, ytrain)
                yt = clf.predict(Xtrain)
                ac = accuracy_score(ytrain, yt)
                predictions = clf.predict(Xtest)
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'kNN':
                neigh = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, ytrain)
                yt = neigh.predict(Xtrain)
                ac = accuracy_score(ytrain, yt)
                predictions = neigh.predict(Xtest)
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'RegressionTree':
                clf = tree.DecisionTreeClassifier().fit(Xtrain, ytrain)
                yt = clf.predict(Xtrain)
                ac = accuracy_score(ytrain, yt)
                predictions = clf.predict(Xtest)
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'Hybrid1': # Logistic followed by kNN
                clf1 = sl.LogisticRegression(solver='lbfgs').fit(Xtrain, ytrain)
                predictions = clf1.predict(Xtest)
                misclassified = np.abs(ytest - predictions)
                idx = np.where(misclassified == 1)
                test = Xtest[idx, :][0]
                neigh = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, ytrain)
                predictions2 = neigh.predict(test)
                predictions[idx] = predictions2
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'Hybrid2':
                clf = tree.DecisionTreeClassifier().fit(Xtrain, ytrain)
                predictions = clf.predict(Xtest)
                misclassified = np.abs(ytest - predictions)
                idx = np.where(misclassified == 1)
                test = Xtest[idx, :][0]
                neigh = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, ytrain)
                predictions2 = neigh.predict(test)
                predictions[idx] = predictions2
                predictedClasses[test_index] = predictions
                predictedClasses[test_index] = predictions
                
            if self.options['method'] == 'Hybrid3':
                clf = SVC(gamma='auto').fit(Xtrain, ytrain)
                predictions = clf.predict(Xtest)
                misclassified = np.abs(ytest - predictions)
                idx = np.where(misclassified == 1)
                test = Xtest[idx, :][0]
                neigh = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, ytrain)
                predictions2 = neigh.predict(test)
                predictions[idx] = predictions2
                predictedClasses[test_index] = predictions
            train_accuracy.append(ac)   
            
        cm = confusion_matrix(Y_balance, predictedClasses)
        train_accuracy = np.mean(train_accuracy)
        test_accuracy = accuracy_score(Y_balance, predictedClasses)
        p_score = precision_score(Y_balance, predictedClasses)
        r_score = recall_score(Y_balance, predictedClasses)
        
        return cm, train_accuracy, test_accuracy, p_score, r_score


    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        #plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('CM_'+self.options['method']+'.png')
        
