import pandas as pd
import numpy as np

#
# Manuel Dogrusal Regresyon 
#

x1=x_train.values
x2=x_test.values
y1=y_train.values
y2=y_test.values

a1= sum((x1-np.mean(x1))*(y1-np.mean(y1)))
a2= ((x1-np.mean(x1))**2)

beta1 = a1/a2

print(beta1)

beta0 = np.mean(y1)-beta1*np.mean(x1)

print(beta0)

# Fonksiyon
# y=beta0+beta1*x

tahmin_manuel=beta0+beta1*x_test

# Manuel Çizim

Fiyat = data["Fiyat"].values
LSTAT = data["LSTAT"].values

import matplotlib.pyplot as plt

plt.scatter(LSTAT,Fiyat, color="blue")
plt.xlabel("Alım Gücü Nüfus Oranı")
plt.ylabel("Ev Fiyatları")
plt.plot(x2,tahmin_manuel,color="red")

# Kod ile EKK Grafik Çizimi
# Durum 1 

import seaborn as sns

sns.lmplot(x="LSTAT",y="Fiyat",data=data)

# Durum 2 

iris=load_dataset("iris")
sns.lmplot(x="sepal_lenght",y="sepal_width",hue="species",data=iris)

