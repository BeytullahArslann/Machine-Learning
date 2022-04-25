# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:06:05 2021

@author: 90545
"""

###LOJISTIK REGRESYON

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:3].values #bagimsiz degiskenler (girdiler)
y=veriler.iloc[:,-1:].values #bagimli degisken (cikti)



#VERİLERİN EGİTİM VE TEST KÜMESİ İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred=logr.predict(x_test) #tahmin
print(y_pred)
tahmin=pd.DataFrame(y_pred)


from sklearn.metrics import confusion_matrix,classification_report
cm= confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

"""
classification_report: sınıflandırma raporu
accuary 1'e ne kadar yakınsa model o kadar başarılı
"""