import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("linear_regression_dataset.csv",sep=";")

data.head()


data.info()



plt.scatter(df.deneyim,df.maas) #x ve y koordinatlarına hangi sütunların geleceği belirlenir.
plt.xlabel("Deneyim") #x ekseninin adı atanır
plt.ylabel("Maas")    #y ekseninin adı atanır.
plt.show()


x = data.deneyim.values 

x.shape #bu feature 14 satır 1 sütundan oluşmaktadır.

x = data.deneyim.values.reshape(-1,1)
y = data.maas.values.reshape(-1,1)






from sklearn.linear_model import LinearRegression
#sklearn kütüphanesinin içinde machine learning modelleri bulunur.

linear_reg = LinearRegression() #LinearRegression modeli linear_reg adlı variable'a eşitlenir.
linear_reg.fit(x,y)             #line fit edilir.




#prediction
import numpy as np

b0 = linear_reg.predict([[0]]) #fit edilen line'ın b0 değişkenine yani y eksenini kestiği noktaya bakılır.
print("b0: ",b0)               #y eksenine kestiği noktada x değeri 0 olacağından y=b0'dır.


b0_ = linear_reg.intercept_    #ayrıca b0 değeri değeri intercept methoduyla da bulunur.
print("b0: ",b0_)   



b1 = linear_reg.coef_  #b1'in diğer adı coefficient'tır. coef methoduyla b1 değeri bulunur.
print("b1: ",b1)   



new_salary = 1663 + 1138*11 #11 yıllık deneyimi olan birinin maaşı linear regression denklemine göre hesaplanmıştır.
print(new_salary)



b11 = linear_reg.predict([[11]])  #11 yıllık deneyimi olan birinin maaşı predict methoduyla bulunur.
print("b11: ",b11)



y_head = linear_reg.predict(x)



plt.plot(x, y_head, color="red")
plt.scatter(x,y)
plt.show()



from sklearn.metrics import r2_score
print("R Square Score: ",r2_score(y,y_head))



from sklearn.metrics import mean_squared_error
print("Mean Squared Error: ",mean_squared_error(y,y_head))



MSE = np.square(np.subtract(y,y_head)).mean()
print("Mean Squared Error: ",MSE)
