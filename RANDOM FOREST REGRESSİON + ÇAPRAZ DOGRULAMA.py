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

#RANDOM FOREST REGRESSİON

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x_train, y_train)
tahmin= rf_reg.predict(x_test)
print("RASSAL ORMAN R2 DEGERİ")
print(r2_score(y_test,tahmin))

#ÇAPRAZ DOGRULAMA 

from sklearn.model_selection import cross_val_score

kvs =cross_val_score(rf_reg, x_train, y_train,cv=5)
print("ÇAPRAZ DOGRULAMA R2 SCORE")
print(kvs)
print(np.mean(kvs))

