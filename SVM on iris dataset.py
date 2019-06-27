import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset=load_iris()

X=dataset.data
y=dataset.target

from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)
svm.score(X,y)

from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(X,y)
l.score(X,y)