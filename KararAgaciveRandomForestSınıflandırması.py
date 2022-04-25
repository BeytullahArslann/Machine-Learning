# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 00:08:06 2021

@author: 90545
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report

veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

#Girdi olarak boy,kilo,yaş alınsın.Çıktı olarak cinsiyet alınsın:

x=veriler.iloc[:,1:4].values #bagimsiz degiskenler (girdiler)
y=veriler.iloc[:,4:].values #bagimli degisken (cikti)

### iloc kullanımı = virgulun sag tarafi satir,solu sutun. 
### : verilerin hepsini al demektir. 1:4 ise 1'den 3e kadar olanlari alir.4'u almaz


#VERİLERİN EGİTİM VE TEST KÜMESİ İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


##KARAR AGACININ SINIFLANDIRMADA KULLANILMASI:
"""
from sklearn.tree import DecisionTreeClassifier
#dtc= DecisionTreeClassifier(criterion= 'gini') defaultu gini
dtc= DecisionTreeClassifier(criterion= 'entropy')

dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
"""



#RANDOMFOREST SINIFLANDIRMADA KULLANILMASI:
"""
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=5,criterion='entropy')
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
"""


