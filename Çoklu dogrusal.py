import pandas as pd
import numpy as np


data = pd.read_csv("data.csv")

çıktı = data
çıktı = data["K"]

girdi = data.drop(["K","Type"],axis=1)


# Çoklu Doğrusal Regresyon Model İnşaası
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(girdi,çıktı)

# Çoklu Doğrusal Regresyon Çapraz Doğrulaması
from sklearn.model_selection import cross_val_score

kvs =cross_val_score(lr,girdi,çıktı,cv=5)
print("\nÇoklu Doğrusal Regresyon ÇAPRAZ DOGRULAMA")
print(kvs)
print(np.mean(kvs))