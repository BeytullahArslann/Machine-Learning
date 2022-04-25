# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:39:19 2021

@author: 90545
"""


#KNN ALGORITMASI#

#1. Kutuphaneler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report

"""
confusion_matrix : Sınıflandirma algoritmalarinin model basarilarini
ölcmek icin kullanilir.
"""

#2. Veri Onizleme

#2.1. Veri Yukleme
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

"""
#VERILERIN OLCEKLENMESI
from sklearn.preprocessing import StandardScaler
X_train=sc.fit_transform(x_train) #Eğit ve Dönüştür. x_traine gore standartize eder.
X_test=sc.transform(x_test)
"""


"""
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=1, metric="chebyshev")
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train) #fit : modeli eğit
y_pred=knn.predict(x_test) #predict : tahmin fonks.


cm=confusion_matrix(y_test,y_pred) #modelin kaç tane dogru bildigine bakiyor
print(cm)
print(classification_report(y_test,y_pred))
"""

"""
#En iyi k degerini bulmak için
#!!sınavda gelebilirmis
from sklearn.model_selection import GridSearchCV
par={"n_neighbors":np.arange(1,5)} #sözlük
#arange: k'yı 1'den 5'e kadar dene.5'i almaz.
#sözlük 1den çok parametre alabilir.

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn_cv=GridSearchCV(knn,par,cv=3)
knn_cv.fit(x,y)
print("En İyi Parametreler")
print(knn_cv.best_params_)
print(knn_cv.best_score_)
"""




























