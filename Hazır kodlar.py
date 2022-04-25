# Hazır kodlar 

# KARAR AGACI YAPISI 

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(egitimx,egitimy)

tahmin = dt .predict(testx)

print("KARAR AGACI R2 DEGERİ")
print(r2_score(testy , tahmin))



#RANDOM FOREST REGRESSİON

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, y)
tahmin= rf_reg.predict(X)
print("RASSAL ORMAN R2 DEGERİ")
print(r2_score(testy,tahmin))

#POLİNOM REGRESYON

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) #degree degeri polinomun kaçıncı dereceden olacagını belirler 
x_poly = poly_reg.fit(trainx)
poli = LinearRegression()
poli.fit(x_poly,trainy)
tahmin = poli.predict(poly_reg.fit_transform(testx))
print("POLİNOM REGRESYON R2 DEGERİ")
print(r2_score(testy,poli.predict(poly_reg.fit_transform(testx))))


#ÇAPRAZ DOGRULAMA 

from sklearn.model_selection import cross_val_score

kvs =cross_val_score(dt, trainx, trainy,cv=5)
print("ÇAPRAZ DOGRULAMA R2 SCORE")
print(kvs)
print(np.mean(kvs))
