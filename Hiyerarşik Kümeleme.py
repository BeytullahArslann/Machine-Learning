# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:21:25 2021

@author: MONSTER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values #Hacim ve maaşa göre kümeleme

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4 , affinity='euclidean', linkage = 'ward')
tahmin = ac.fit_predict(X)

gruplarh = pd.DataFrame(tahmin,columns = ['Grup Numarası'])

gruph1=gruplarh[(gruplarh["Grup Numarası"]==0)]
gruph2=gruplarh[(gruplarh["Grup Numarası"]==1)]
gruph3=gruplarh[(gruplarh["Grup Numarası"]==2)]
gruph4=gruplarh[(gruplarh["Grup Numarası"]==3)]


plt.figure()

plt.scatter(X[tahmin == 0,0], X[tahmin == 0,1], s=40, c ='red')
plt.scatter(X[tahmin == 1,0], X[tahmin == 1,1], s=40, c ='blue')
plt.scatter(X[tahmin == 2,0], X[tahmin == 2,1], s=40, c ='green')
plt.scatter(X[tahmin == 3,0], X[tahmin == 3,1], s=40, c ='yellow')
plt.title('Hiyerarşik Bölütleme')
plt.show()

