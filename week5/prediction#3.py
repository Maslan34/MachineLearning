###POLINOMAL REGRESSION

#kütüphane yükleme
import pandas as pd #veriler için (data frame)
import numpy as np  # büyük sayıların hesabı için
import matplotlib.pyplot as plt #çizimler için



#veri yükleme

veriler=pd.read_csv("maaslar.csv") #csv:comma seperated value

#print(veriler.columns)




x=veriler.iloc[:,1:2] #eğiitim düzyi

y=veriler.iloc[:,2:] #maas


X=x.values
Y=y.values




#Burada  farkı görmek için lin tahmini kullandık
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x, y)

plt.scatter(x, y)
plt.plot(x,lin_reg.predict(x) ,color="yellow")


#Burada  farkı görmek için lin tahmini kullandık






from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(X)



lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(X, Y, color="purple")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)) , color="red")

plt.show()


#tahminler

print("-------------")
print(lin_reg.predict([[12]]))
print(lin_reg.predict([[6.7]]))
print(lin_reg.predict([[20]]))

print("-------------")
print(lin_reg2.predict(poly_reg.fit_transform([[12]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.7]])))
print(lin_reg2.predict(poly_reg.fit_transform([[20]])))

#eğitim ve test

#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(birlestirme2,birlestirme3,test_size=0.33,random_state=0)

#eğitim ve test








###POLINOMAL REGRESSION

