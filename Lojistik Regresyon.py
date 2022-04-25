# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:53:16 2021

@author: MONSTER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

veriler = pd.read_csv("veriler.csv")

        #Bağımsız ve bağımlı değişkenleri belirleme
x = veriler.iloc[:,1:3].values #bağımsız değişkenler
y = veriler.iloc[:,-1:].values #bağımlı değişkenler

        #verilerin eğitim ve test içi bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

        #LOJİSTİK REGRESYON
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)

tahmin = pd.DataFrame(y_pred)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

