# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 01:40:32 1019

@author: Siva
"""


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score as score
from sklearn.tree import DecisionTreeClassifier

dataset_train = pd.read_csv('train_encoded3.csv')
X_train = dataset_train.iloc[:,0:10].values
y_train = dataset_train.iloc[:, 10].values
dataset_test = pd.read_csv('test_encoded3.csv')
X_test = dataset_test.iloc[:, 0:10].values
y_test = dataset_test.iloc[:, 10].values
"""
dataset_train = pd.read_csv('train_encoded2.csv')
X_train = dataset_train.iloc[:,0:10].values
y_train = dataset_train.iloc[:, 10].values
dataset_test = pd.read_csv('test_encoded2.csv')
X_test = dataset_test.iloc[:, 0:10].values
y_test = dataset_test.iloc[:, 10].values
"""

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
"Accuracy"
accuracy=accuracy_score(y_test,y_pred)

"Recall"
recall=np.diag(cm)/np.sum(cm,axis=0)

"Precision"
precision = np.diag(cm)/np.sum(cm,axis=1)

"""F1 Score Calculation"""
f1_score=2*((recall*precision)/(recall+precision))


"""Overall Metrics"""
overall_accuracy=np.round(accuracy,4)*100
overall_recall=np.round(np.mean(recall),4)*100
overall_precision=np.round(np.mean(precision),4)*100
overall_f1_score=np.round(np.mean(f1_score),4)*100

#Calculating Individual Accuracy from Confusion Matrix

recall=np.round(recall,4)*100
precision=np.round(precision,4)*100
f1_score=np.round(f1_score,4)*100

    




print("---------------------------------------------------------")
"""OverAll Metrics"""
print("OveraAll Metrics")
print("Overall_Accuracy:"+str(overall_accuracy)+"%")
print("Overall_Recall:"+str(overall_recall)+"%")
print("Overall_Precision:"+str(overall_precision)+"%")
print("Overall f1Score:"+str(overall_f1_score)+"%")
print("---------------------------------------------------------")
print("Precision")
print("NORMAL:"+str(precision[0])+"%")
print("DOS:"+str(precision[1])+"%")
print("PROBE:"+str(precision[2])+"%")
print("R2L:"+str(precision[3])+"%")
print("U2R:"+str(precision[4])+"%")
print("---------------------------------------------------------")
print("Recall")
print("NORMAL:"+str(recall[0])+"%")
print("DOS:"+str(recall[1])+"%")
print("PROBE:"+str(recall[2])+"%")
print("R2L:"+str(recall[3])+"%")
print("U2R:"+str(recall[4])+"%")
print("---------------------------------------------------------")
print("F1_Score")
print("NORMAL:"+str(f1_score[0])+"%")
print("DOS:"+str(f1_score[1])+"%")
print("PROBE:"+str(f1_score[2])+"%")
print("R2L:"+str(f1_score[3])+"%")
print("U2R:"+str(f1_score[4])+"%")
print("---------------------------------------------------------")

n=p=d=r=u=0

for i in range(22543):
    if(y_test[i] == 0):
        n = n + 1
    if(y_test[i] == 1):
        d = d + 1
    if(y_test[i] == 2):
        p = p + 1
    if(y_test[i] == 3):
        r = r + 1
    if(y_test[i] == 4):
        u = u + 1
    
print("Normal:"+str(n))
print("Dos:"+str(d))
print("Probe:"+str(p))
print("R2L:"+str(r))
print("U2R:"+str(u))