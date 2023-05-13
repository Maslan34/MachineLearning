import matplotlib.pyplot as plt 


veriler=pd.read_csv("human_size_and_gender.csv")
x=veriler.iloc[:,1:4].values #bağımsız değişkenler
y=veriler.iloc[:,4].values #bağımlı değişkenler


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

standart_scaler=StandardScaler()


x_train =standart_scaler.fit_transform(x_train)
x_test =standart_scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)


logr.fit(x_train,y_train)


y_pred=logr.predict(x_test)
print(y_pred)


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)