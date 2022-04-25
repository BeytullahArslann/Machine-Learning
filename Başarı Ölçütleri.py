from sklearn.metrics import mean_square_error
import pandas as pd
import numpy as np

# Kod Manuel Dogrusal Regresyonun devamıdır.
#
# Manuel R2 Score
#
x1 = x_train.values
x2 = x_test.values
y1 = y_train.values
y2 = y_test.values

RSS = sum((y2-tahmin)**2)
TSS = sum((y2-np.mean(y2))**2)
R2 = 1-(RSS/TSS)

# Manuel Düzeltilmiş R2 Score

n = len(y2)  # Gözlem sayısı
d = x2.shape[1]  # Bagımsız degişken sayısı (Sütun sayısı)

adj_R2 = 1-(RSS/(n-d-1))/(TSS/(n-1))

# Manuel RMSE

RMSE = np.sqrt(RSS/n)
print(RMSE)

# Kod ile RMSE

rmse = np.sqrt(mean_square_error(tahmin, y_test))
print(rmse)

# Mallow CP

var = np.var(y2-tahmin)
cp = (RSS+2*d*var)/n
print(cp)


# AIC

aic = (RSS+2*d*var)/(n*var)
print(aic)


# BIC

bic = (RSS+np.log(n)*d*var)/(n*var)
print(bic)
