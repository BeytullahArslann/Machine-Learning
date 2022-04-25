from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

from sklearn import datasets

veri=datasets.load_boston()


girdiler=pd.DataFrame(veri.data)
girdiler.columns=veri.feature_names

çıktı=pd.DataFrame(veri.target)
çıktı.columns=["Fiyat"]

data=pd.concat([girdiler,çıktı],axis=1)

#VERİLERİN EGİTİM VE TEST KÜMESİ İÇİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(girdiler,çıktı,test_size=0.33, random_state=0)

#POLİNOM REGRESSİON

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 2) #degree degeri polinomun kaçıncı dereceden olacagını belirler 
x_poly = poly_reg.fit_transform(x_train)
poli = LinearRegression()
poli.fit(x_poly,y_train)

tahmin = poli.predict(poly_reg.fit_transform(x_test))
print("POLİNOM REGRESYON R2 DEGERİ")
print(r2_score(y_test,poli.predict(poly_reg.fit_transform(x_test))))

#ÇAPRAZ DOGRULAMA 

from sklearn.model_selection import cross_val_score

kvs =cross_val_score(poli, x_poly, y_train,cv=5)
print("ÇAPRAZ DOGRULAMA R2 SCORE")
print(kvs)
print(np.mean(kvs))


