# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:49:13 2021

@author: 90545
"""

#BAYES ALGORITMASI

import pandas as pd
import numpy as np
import math

#Hazır Kod
a1=pd.DataFrame(["Evet","Hayır","Evet","Hayır","Evet","Hayır","Evet","Hayır","Hayır","Hayır"],columns=["Kaza"])
a2=pd.DataFrame([25,18,39,50,45,22,35,28,22,40],columns=["Yaş"])
a3=pd.DataFrame(["Erkek","Kadın","Kadın","Kadın","Erkek","Erkek","Erkek","Erkek","Kadın","Erkek"],columns=["Cinsiyet"])
#a4=pd.DataFrame
a4=pd.DataFrame([2,2,1,0,0,1,1,1,1,0])
#2:Yuksek risk , 1:Orta ,0:Düşük
m1=pd.concat([a1,a2,a3],axis=1)

##Kaza ve Cinsiyet kategorik degerler bunları numerik yapmamız gerekiyor.
#Bunun için LabelEncoder kullanilir.Etiketleme. 
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for i in range(0,1):
    m1.iloc[:,i:i+1]=lb.fit_transform(m1.iloc[:,i:i+1])

for i in range(2,3):
    m1.iloc[:,i:i+1]=lb.fit_transform(m1.iloc[:,i:i+1])
    

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(m1,a4)

y_pred= gnb.predict(np.array([1,42,1]).reshape(1,3))
#1,42,1:Kaza yapmamış, 42 yaşında ,Kadın
print(y_pred)
print(gnb.predict_proba(np.array([1,42,1]).reshape(1,3)))


