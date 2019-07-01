# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:47:31 2019

@author: Deepak sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()

X=dataset.data
y=dataset.target

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

from sklearn.svm import SVC
svm=SVC()


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.ensemble import VotingClassifier
vot=VotingClassifier([('LR',log_reg),('KNN',knn),('DT',dtf),('SVM',svm),('NB',n_b)])



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
knn.fit(X_train,y_train)

knn.score(X_train,y_train)
vot.fit(X,y)
vot.score(X,y)

from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(log_reg,n_estimators= 5)
bag.fit(X,y)
bag.score(X,y)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=5)
rf.fit(X,y)
rf.score(X,y)
