import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x=np.array([0,1,2,3,4,5,6])

y=np.array([1,6,17,34,57,86,121])

poly=PolynomialFeatures(degree=2)

x_poly =poly.fit_transform(x.reshape(-1,1))

model = LinearRegression()

model.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x_poly),color="red")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("") #column2C

df.head()
df.tail()


abnormaldata1 = df[df["class"] == "Abnormal"]
x2 = np.array(abnormaldata1.loc[:,"pelvic_incidence"]).reshape(-1,1)
y2 = np.array(abnormaldata1.loc[:,"sacral_slope"]).reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()



x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print("r2 score: ",r2_score(y_test,y_pred))

polyReg =PolynomialFeatures(degree=4,include_bias=True)
x_train_trans = poly.fit(x_train)
x_test_trans = poly.fit(x_test)

lr =LinearRegression()

lr.fit(x_train_trans,y_train)

y_poly_pred = lr.predict(x_test_trans)
print("r2score with poly: ",r2_score(y_test,y_poly_pred))
      
X_new =np.linspace(0,132,200).reshape(200,1)

X_new_poly = polyReg.transform(X_new)
y_new =lr.predict(X_new_poly)

plt.figure(figsize=[10,10])
plt.plot(X_new, y_new,"r-",linewidth=2,label="Prediction")
plt.plot(x_train,y_train,"b.",label="Traning Points")
plt.plot(x_test,y_test,"g",label="Testing Points")
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.legend()
plt.show()










