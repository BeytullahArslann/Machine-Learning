import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

çıktı = data
çıktı=data["K"]

girdi = data.drop(["K","Type"],axis=1)


#RANDOM FOREST REGRESSİON

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(girdi, çıktı)
tahmin= rf_reg.predict(girdi)

#ÇAPRAZ DOGRULAMA 

from sklearn.model_selection import cross_val_score

kvs =cross_val_score(rf_reg, girdi, çıktı,cv=5)
print("ÇAPRAZ DOGRULAMA R2 SCORE")
print(kvs)
print(np.mean(kvs))



