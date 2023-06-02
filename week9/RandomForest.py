import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("/kaggle/input/tumor-data/tumor-data.csv")
df.head()



df.drop(["id","Unnamed: 32"],axis=1,inplace=True)


df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df.diagnosis.values
x_data = df.drop(["diagnosis"],axis=1)



x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=42)



from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()



dt.fit(x_train,y_train)



print("decisiontree score: ",dt.score(x_test,y_test)) 



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, random_state=1)



rf.fit(x_train,y_train)




print("random forest algo result",rf.score(x_test,y_test)) 



import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce 





df2= pd.read_csv("/kaggle/input/car-evaluation-data-set/car_evaluation.csv", header=None)
df2.head()




df2.shape



col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df2.columns = col_names
col_names



df2.head()





df2.info()





for col in col_names:
    print(df2[col].value_counts())





df2["class"].value_counts()




df2.isnull().sum()




x = df2.drop(["class"], axis=1)
y= df2["class"]




x_train.head()






x_test.head()





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)




encoder = ce.OrdinalEncoder(cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)




x_train.head()







x_test.head()







rfc = RandomForestClassifier(n_estimators=10, random_state=0)







rfc.fit(x_train,y_train)






y_pred= rfc.predict(x_test)
y_pred





print("model accuracy score with 10 decision-trees : {0:0.4f}".format(accuracy_score(y_test, y_pred)))




rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0) 
rfc_100.fit(x_train,y_train)
y_pred_100= rfc_100.predict(x_test)

print("model accuracy score with 100 decision-trees : {0:0.4f}".format(accuracy_score(y_test, y_pred_100)))




df3=pd.read_csv("/kaggle/input/descending-numbers/Descending_Numbers.csv")
df3




x_numbers=df3.iloc[:,1].values.reshape(-1,1)
y_numbers=df3.iloc[:,2].values.reshape(-1,1)




rfRegressor=RandomForestRegressor(n_estimators=100,random_state=42)




rfRegressor.fit(x_numbers,y_numbers)




rfRegressor.predict([[4.3]])





x_numbers_all=np.arange(min(x_numbers),max(x_numbers),0.01).reshape(-1,1)




y_pred=rfRegressor.predict(x_numbers_all)





plt.scatter(x_numbers,y_numbers,color="red")
plt.plot(x_numbers_all,y_pred,color="green")
plt.xlabel("feature")
plt.ylabel("label")
plt.show()







