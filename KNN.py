# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("veriler.csv")


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişkenler

        #verilerin eğitim ve test içi bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=1)


        #verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric="chebyshev")
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
"""

        # Grid search (Tune işlemi)

from sklearn.model_selection import GridSearchCV
par = {"n_neighbors":np.arange(1,7),"metric":["manhattan","minkowski"]}

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,par,cv=3)
knn_cv.fit(x,y)
print("En İyi Parametreler")
print(knn_cv.best_params_)


        # Çapraz Dogrulama

from sklearn.model_selection import cross_val_score
CD = cross_val_score(knn,x,y,cv=5)
print(CD)
print(np.mean(CD))

        # Karar Ağacı Yapısı

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion= 'entropy') 
# gini , entropy

dtc.fit(x_train,y_train)
y_pred4 = dtc.predict(x_test)
cm4 = confusion_matrix(y_test,y_pred4)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, y_pred4))


            # Random Forest 
            
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5 , criterion='entropy')
rfc.fit(x_train,y_train)

y_pred5 = rfc.predict(x_test)
cm5 = confusion_matrix(y_test,y_pred5)
print(classification_report(y_test,y_pred5))






