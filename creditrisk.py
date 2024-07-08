# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 03:11:06 2020

@author: Lefu
"""

import pandas as pd
import numpy as np
import classify
import matplotlib.pyplot as plt


credit_record = pd.read_csv("credit_record.csv")
application_record = pd.read_csv("application_record.csv")

credit_record = credit_record.filter(['ID', "STATUS"])
credit_record = credit_record.loc[credit_record['STATUS'] != 'X']
credit_record = credit_record.loc[credit_record['STATUS'] != '0']

def risk_class(rep):
    if rep == '0':
        return 1
    if rep == '1':
        return 1
    if rep == '2':
        return 0
    if rep == '3':
        return 0
    if rep == '4':
        return 0
    if rep == '5':
        return 0
    else:
        return 1
        
RiskClass = np.vectorize(risk_class)

credit_record['Class'] = RiskClass(credit_record['STATUS'].values)
IDs = credit_record['ID'].unique()
aggregated_score = []

ID1 = set(application_record['ID'].values)
ID2 = set(IDs)

ID_use = ID1.intersection(ID2)

aggr_df = pd.DataFrame()
for idx, id in enumerate(ID_use):
    if idx == 0:
        aggr_df = application_record.loc[application_record['ID'] == id]
    else:
        aggr_df = pd.concat([aggr_df, application_record.loc[application_record['ID'] == id]])
    temp = np.min(credit_record.loc[credit_record['ID'] == id]['Class'].values)
    aggregated_score.append(temp)

aggr_df['Status'] = aggregated_score

aggr_df.loc[aggr_df['CNT_CHILDREN'] <= 1, 'CNT_CHILDREN'] = 0
aggr_df.loc[aggr_df['CNT_CHILDREN'] > 1, 'CNT_CHILDREN'] = 1

aggr_df['AMT_INCOME_TOTAL'] = np.log(aggr_df['AMT_INCOME_TOTAL'].values)
aggr_df.loc[aggr_df['AMT_INCOME_TOTAL'] >= np.mean(aggr_df['AMT_INCOME_TOTAL'].values), 'AMT_INCOME_TOTAL'] = 1
aggr_df.loc[aggr_df['AMT_INCOME_TOTAL'] > 1, 'AMT_INCOME_TOTAL'] = 0


aggr_df['DAYS_BIRTH'] = np.log(aggr_df['DAYS_BIRTH'].values*-1)
aggr_df.loc[aggr_df['DAYS_BIRTH'] >= np.mean(aggr_df['DAYS_BIRTH'].values), 'DAYS_BIRTH'] = 1
aggr_df.loc[aggr_df['DAYS_BIRTH'] != 1, 'DAYS_BIRTH'] = 0

aggr_df['DAYS_EMPLOYED'] = aggr_df['DAYS_EMPLOYED'].values/365
aggr_df.loc[aggr_df['DAYS_EMPLOYED'] < -10, 'DAYS_EMPLOYED'] = 0 
aggr_df.loc[aggr_df['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED'] = 1 
aggr_df.loc[aggr_df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 2


aggr_df = aggr_df.drop(columns=['ID'])
   
to_replace = {'F':0, 'M':1, 'N':0, 'Y':1, 'Working':1, 'State servant':2,
              'Commercial associate':3, 'Secondary / secondary special':0,
              'Higher education':1, 'Married':1, 'Civil marriage':2,
              'Single / not married':0, 'Pensioner':4, 'With parents':0,
              'House / apartment':1, 'Managers':1, 'Core staff':2, 
              'Cooking staff':3, 'High skill tech staff':4, np.nan:0,
              'Incomplete higher':2, 'Separated':3, 'Rented apartment':2,
              'Municipal apartment':3, 'Academic degree':3, 'Widow':4, 
              'Lower secondary':4, 'Sales staff':5, 'Laborers':6,
              'Secretaries':7, 'Cleaning staff':8, 'Drivers':9,
              'Accountants':10, 'Security staff':11, 'Waiters/barmen staff':12,
              'Low-skill Laborers':13, 'Medicine staff':14, 'Private service staff':15,
              'Realty agents':16, 'IT staff':17, 'HR staff':18, 'Co-op apartment':4,
              'Office apartment':5, 'Student':5}

for key in to_replace.keys():
    aggr_df = aggr_df.replace(key, to_replace[key])

m = np.array(aggr_df)
X = m[:, 0:17]
y = m[:, 17]

options = [{'CV':5, 'seed':32, 'method':'LogisticRegression'},
           {'CV':5, 'seed':32, 'method':'kNN'},
           {'CV':5, 'seed':32, 'method':'SVM'},
           {'CV':5, 'seed':32, 'method':'RegressionTree'}]#,
           #{'CV':5, 'seed':32, 'method':'Hybrid1'},
           #{'CV':5, 'seed':32, 'method':'Hybrid2'},
           #{'CV':5, 'seed':32, 'method':'Hybrid3'}]
class_labels = ['High Risk', 'Low Risk']

algorithms = []
train_accuracy = []
test_accuracy = []
precision_score = []
recall_score = []

for option in options:
    algorithms.append(classify.classify(X, y, option))

for i in range(4):
    ac = algorithms[i].metrics()
    algorithms[i].plot_confusion_matrix(ac[0],
                      classes= class_labels, normalize = False, 
                      title='Confusion Matrix: ' + algorithms[i].options['method'])
    train_accuracy.append(ac[1])
    test_accuracy.append(ac[2])
    precision_score.append(ac[3])
    recall_score.append(ac[4])
    print('For method %s the train accuracy is %f and the test accuracy is %f' %(algorithms[i].options['method'], ac[1], ac[2]))
    
methods = ['Logistic', 'kNN', 'SVM', 'CART']#, 'Hybrid1', 'Hybrid2', 'Hybrid3']
width = 0.2

pos1 = np.arange(4)
pos2 = [x + width for x in pos1]
pos3 = [x + width for x in pos2]
pos4 = [x + width for x in pos3]

plt.figure()
plt.bar(pos1, train_accuracy, color = 'black', width=width, edgecolor = 'white', label='Mean Train Accuracy')
plt.bar(pos2, test_accuracy, color = 'blue', width=width, edgecolor = 'white', label='Test Accuracy')
plt.bar(pos3, precision_score, color = 'red', width=width, edgecolor = 'white', label='Precision')
plt.bar(pos4, recall_score, color = 'green', width=width, edgecolor = 'white', label='Recall')
plt.grid()
plt.ylabel('Score')
plt.xlabel('ML Algorithm')
plt.xticks([r + width + 0.05 for r in range(4)], methods)
plt.show()
#plt.savefig('performance.png')
