# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:57:08 2023

@author: Coppe
"""


#kütüphane yükleme
import pandas as pd #veriler için (data frame)
import numpy as np  # büyük sayıların hesabı için
import matplotlib.pyplot as plt #çizimler için

#veri yükleme

veriler=pd.read_csv("satislar.csv") #csv:comma seperated value



#eğitim ve test

from sklearn.model_selection import train_test_split


aylar=veriler[["Aylar"]]

satislar=veriler[["Satislar"]]

#ilk parametre bağımsız ikinci parametre bağımlı değişken aylar->> bağımsız satislar ->>bağımlı

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)


#eğitim ve test




#öznitelik

from sklearn.preprocessing import StandardScaler

standart_scaler=StandardScaler()



# !!!!!!!! bu standartlaştırma işleminde dataFrame olarak yapınca farklı array olarak yapınca bu işlemleri (iloc yüzünden) sonuçlar farklı çıkıyor.
"""
X_train =standart_scaler.fit_transform(x_train)
X_test =standart_scaler.fit_transform(x_test)


Y_train =standart_scaler.fit_transform(y_train)
Y_test =standart_scaler.fit_transform(y_test)
"""
#öznitelik



#Model inşası
    #linear Regression


from sklearn.linear_model import LinearRegression

linear_regression=LinearRegression()

linear_regression.fit(x_train,y_train)



tahmin=linear_regression.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test,linear_regression.predict(x_test))



    #linear Regression
#Model inşası

