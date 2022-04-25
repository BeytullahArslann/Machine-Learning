# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:06:18 2021

@author: MONSTER
"""

import pandas as pd 
import numpy as np 
import math

"""
        # Manuel Kod 
c1 = [45,50]
ort1 = np.mean(c1)
sapma1 = np.std(c1)
sapma1 = sapma1*np.sqrt(len(c1)/(len(c1)-1))

c2 = [39,22,35,28,22]
ort2 = np.mean(c2)
sapma2 = np.std(c2)
sapma2 = sapma2*np.sqrt(len(c2)/(len(c2)-1))

c3 = [25,18,40]
ort3 = np.mean(c3)
sapma3 = np.std(c3)
sapma3 = sapma3*np.sqrt(len(c3)/(len(c3)-1))

g1 = (1/(sapma1*np.sqrt(2*math.pi)))*math.exp(-0.5*((42-ort1)/sapma1)**2)
g2 = (1/(sapma2*np.sqrt(2*math.pi)))*math.exp(-0.5*((42-ort2)/sapma2)**2)
g3 = (1/(sapma3*np.sqrt(2*math.pi)))*math.exp(-0.5*((42-ort3)/sapma3)**2)

g1 = g1*0.5*0.5*0.2
g2 = g2*0.6*0.2*0.5
g3 = g3*(2/3)*(1/3)*0.3

S = g1+g2+g3

P1 = g1/S
P2 = g2/S
P3 = g3/S
"""

a1 = pd.DataFrame(["Evet","Hayır","Evet","Hayır","Evet","Hayır","Evet","Hayır","Hayır","Hayır"],columns=["Kaza"])
a2 = pd.DataFrame([45,50,39,22,35,28,22,25,18,40],columns=["Yaş"])
a3 = pd.DataFrame(["Erkek","Kadın","Kadın","Kadın","Erkek","Erkek","Erkek","Erkek","Kadın","Erkek"],columns=["Cinsiyet"])
# a4 = pd.DataFrame(["Yüksek","Yüksek","Orta","Düşük","Düşük","Orta","Orta","Orta","Orta","Yüksek"])
a4 = pd.DataFrame([2,2,1,0,0,1,1,1,1,0])
m1 = pd.concat([a1,a2,a3],axis=1)

        # Etiketleme Kısmı
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
  
# Kaza sütunu etiketlendi
for i in range(0, 1):
    m1.iloc[:,i:i+1] = lb.fit_transform(m1.iloc[:,i:i+1])

# Cinsiyet sütunu etiketlendi
for i in range(2, 3):
    m1.iloc[:,i:i+1] = lb.fit_transform(m1.iloc[:,i:i+1])
    
    
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(m1, a4)

y_pred = gnb.predict(np.array([1,42,1]).reshape(1,3)) # 1 42 1 olmasının sebebi bizim tahmin ettiğimiz kişi 42 yaşında kadın ve hiç kaza yapmamış ( Kaza yapmamış = 1 , Kadın = 1)
                                                      # reshape kullanma amacımız 1 42 1 i 1 satıwr 3 sütunluk bir matris haline getirmek 
print(y_pred)
print(gnb.predict_proba(np.array([1,42,1]).reshape(1,3)))    





















