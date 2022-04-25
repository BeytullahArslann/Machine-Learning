# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:32:13 2021

@author: MONSTER
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:]


from sklearn.cluster import KMeans

model = KMeans(n_clusters=3 , init= "k-means++" , random_state=0)
model.fit(X)

gruplar = model.predict(X)

"""x = X[:,0]
y = X[:,1]
plt.scatter(x,y)"""

merkezler = model.cluster_centers_

x = X[:,0]
y = X[:,1]

merkezler_x = merkezler[:,0]
merkezler_y = merkezler[:,1]

plt.scatter(x,y, c=gruplar)

plt.scatter(merkezler_x,merkezler_y,marker = "P", s=100 ,c="r")

gruplardt = pd.DataFrame(gruplar,columns=["Grup Numarası"])


grup1 = gruplardt[(gruplardt["Grup Numarası"] == 0)]
grup2 = gruplardt[(gruplardt["Grup Numarası"] == 1)]
grup3 = gruplardt[(gruplardt["Grup Numarası"] == 2)]


plt.figure()
sonuclar = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init='k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11),sonuclar)



