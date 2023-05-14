from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# Data Seti Yükleme
iris = load_iris()
iris_data = iris.data
iris_columns = iris.feature_names

# Data Seti DataFrame Çevirme
iris_df = pd.DataFrame(data=iris_data, columns=iris_columns)
iris_df



iris_df.info()


iris_df.head(20)


scaler = MinMaxScaler()
iris_df_scaled = scaler.fit_transform(iris_df)

iris_df_normalized = pd.DataFrame(data=iris_df_scaled, columns=iris_columns)
print(iris_df_normalized.head(30))


sns.pairplot(iris_df_normalized)





import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv("tumor-data.csv")
data.head()


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()



M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]



plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="ıyı",alpha=0.3)
plt.xlabel("radius_mean(tümör yarıçapı)")
plt.ylabel("texture_mean(tümör dokusu)")
plt.legend()
plt.show()



data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]

x_data= data.drop(["diagnosis"],axis=1)
y= data.diagnosis.values



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)



nb = GaussianNB() #Naive-Bayes Lib
nb.fit(x_train,y_train)


print("Doğruluk : ",nb.score(x_test,y_test))







import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC




data2= pd.read_csv("tumor-data.csv")



data2.head()



data2.info()



data2.drop(["Unnamed: 32","id"],axis=1, inplace=True)



M_data2 = data2[data2.diagnosis=="M"]
B_data2 = data2[data2.diagnosis=="B"]



plt.scatter(M_data2.radius_mean,M_data2.texture_mean,color="red",label="malignant") 
plt.scatter(B_data2.radius_mean,B_data2.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()





data2.diagnosis = [1 if each=="M" else 0 for each in data2.diagnosis] 

y_data2 = data2.diagnosis.values 
x_data2 = data2.drop(["diagnosis"],axis=1) 





x_data2= (x_data2 - np.min(x_data2))/(np.max(x_data2)-np.min(x_data2))




x_data2.head()



from sklearn.model_selection import train_test_split
x_train_data2, x_test_data2, y_train_data2, y_test_data2 = train_test_split(x_data2,y_data2,test_size=0.3,random_state=1)




svc= SVC(random_state=42)
svc.fit(x_train_data2,y_train_data2)



svc.score(x_test_data2,y_test_data2)



train_accuracy = []
test_accuracy = []
for i in range(1,100):
    svm = SVC(C=i)
    svm.fit(x_train_data2,y_train_data2)
    train_accuracy.append(svm.score(x_train_data2,y_train_data2))
    test_accuracy.append(svm.score(x_test_data2,y_test_data2))
    
plt.plot(range(1,100),train_accuracy,label="training accuracy")
plt.plot(range(1,100),test_accuracy,label="testing accuracy")
plt.legend()
plt.xlabel("C values")
plt.ylabel("Accuracy")
plt.grid()
plt.show()




print(" En iyi doğruluk {} oranı C = {} olduğunda  ölçülür.".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))