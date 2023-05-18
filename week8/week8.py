import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

dataset = pd.read_csv("mushrooms.csv")


dataset.head()


dataset.info()


dataset.isnull().sum()


dataset["class"].unique()


dataset.shape


sns.histplot(dataset["class"])



x = dataset.drop(["class"],axis=1)
y  = dataset["class"]
x.head()



x = pd.get_dummies(x)
x.head()



encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)



from sklearn.model_selection  import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)



x_test.shape, x_train.shape


y_test.shape, y_train.shape



clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)



clf_gini.fit(x_train, y_train)


plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(x_train,y_train))



y_pred_gini = clf_gini.predict(x_test)
y_pred_gini



y_pred_train_gini = clf_gini.predict(x_train)
y_pred_train_gini




print("model accuracy giniye göre: {0:0.4f}".format(accuracy_score(y_test,y_pred_gini)))
print("training set accuracy score: {0:0.4f}".format(accuracy_score(y_train,y_pred_train_gini)))



cm = confusion_matrix(y_test, y_pred_gini)
print("confusion matrix\n\n", cm)




f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.show()
plt.savefig("confusion matrix")




print(classification_report(y_test, y_pred_gini))




f1_score= f1_score(y_test, y_pred_gini)
print("f1 score", f1_score)




data = pd.read_csv("tumor-data.csv")



data.head()


data.info()




data.drop(["Unnamed: 32","id"],axis=1, inplace=True)



M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]




plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant") 
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()





data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis] 





y = data.diagnosis.values 




x_data= data.iloc[:,1:3].values 




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size=0.3,random_state=1)




from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
print(x_test,y_train)




from sklearn.tree import DecisionTreeClassifier




tree_classification = DecisionTreeClassifier(random_state=1, criterion='entropy')





tree_classification.fit(x_train,y_train)




y_head = tree_classification.predict(x_test)
y_head



from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_head)
accuracy




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_head)




f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,fmt= '.0f', linewidths = 0.5, linecolor = "red", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.show()






from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step =0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, tree_classification.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple','green' )))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)
plt.title('Decision Tree Algorithm (Training set)')
plt.xlabel('iyi_huylu_tümör')
plt.ylabel('kötü_huylu_tümör')
plt.legend()
plt.show()






from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step =0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, tree_classification.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple','green' )))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('purple', 'green'))(i), label = j)
plt.title('Decision Tree Algorithm(Test set)')
plt.xlabel('iyi_huylu_tümör')
plt.ylabel('kötü_huylu_tümör')
plt.legend()
plt.show()






from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,r2_score
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')








df= pd.read_csv("bank-additional-full.csv",sep=';')
df.head()


df.info()




for col in df.columns:
    print()
    if df[col].dtype == 'object':
        print(f'Sütun Adı: {col} ve eşsiz değerler: {df[col].unique()}')



def return_categorical(df):
    
  categorical_columns = [column_name for column_name in df if df[column_name].dtype == 'O']
  return categorical_columns

def return_numerical(df):

  return list(set(df.columns) - set(return_categorical(df)))


def check_normal(df):
  fig, axes = plt.subplots(1,len(return_numerical(df)), figsize =(70, 10))

  for i,numeric_column_name in enumerate(list(set(df.columns) -set(return_categorical(df)))):

    sns.distplot(df[numeric_column_name], ax=axes[i]);
    plt.title(f'Distribution of {numeric_column_name}');
    
def classifier(clf, x_train,x_test,y_train,y_test):
    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)

    accuracy_test = accuracy_score(y_test,y_test_pred)
    accuracy_train =  accuracy_score(y_train,y_train_pred)
    
    roc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovr')
    roc_train = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
    
    print('Train accuracy is:',accuracy_train )
    print('Test accuracy is:',accuracy_test )
    print()
    print('Train ROC is:', roc_train)
    print('Test ROC is:',roc_test )
    
    f1 = f1_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred) 
    print()
    print("F score is:",f1 )
    print("Precision is:",precision)
    print("Recall is:", recall)
  

def random_search(clf,params, x_train,x_test,y_train,y_test):
    
    random_search = RandomizedSearchCV(estimator= clf, param_distributions=params, scoring='roc_auc', cv=5)
    random_search.fit(x_train, y_train)
    optimal_model = random_search.best_estimator_

    print("Best parameters are: ", random_search.best_params_)
    print()
    print("Best estimator is: ", random_search.best_estimator_)
    print()
    print('Scores and accuracies are:')
    print()
    classifier(optimal_model, x_train,x_test,y_train,y_test)


check_normal(df)



for col in return_categorical(df):
    counts = df[col].value_counts().sort_index()
    if len(counts) > 10:
      fig = plt.figure(figsize=(30, 10))
    else:
      fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()




corr = df.corr()
corr_greater_than_75 = corr[corr>=.75]
corr_greater_than_75



plt.figure(figsize=(12,8))
sns.heatmap(corr_greater_than_75, cmap="Reds", annot = True);



df['pdays'].unique()



df['job'] = df['job'].apply(lambda x: -1 if x=='unknown' or x=='unemployed' else (15 if x=='entrepreneur' else (8 if x == 'blue-collar' else ( 6 if x=='technician' or x=='services' or  x=='admin.' or x=='management' else (4 if x== 'self-employed' or x=='student' else (2 if x=='housemaid' or x=='retired' else None) )))))
df['housing'] = df['housing'].apply(lambda x: 0 if x=='no' else (1 if x=='yes' else -1))
df['loan'] = df['loan'].apply(lambda x: 0 if x=='no' else (1 if x=='yes' else -1))
df['y'] = df['y'].apply(lambda x: 0 if x=='no' else (1 if x=='yes' else -1))
df['default'] = df['default'].apply(lambda x: 0 if x=='no' else (1 if x=='yes' else -1))
df['poutcome'] = df['poutcome'].apply(lambda x: 0 if x=='failure' else (2 if x=='failure' else -1))
df['pdays'] = df['pdays'].apply(lambda x: 0 if x==999 else(20 if x<=10 else(6 if x<=20 else 3)))



df.drop(['day_of_week', 'contact', 'month'], axis=1, inplace = True)


df  = pd.get_dummies(df, drop_first = True)


x = df.drop("y", axis=1)
y = df['y']
x.sample()

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=42)





smote = SMOTE()

x_smote, y_smote = smote.fit_resample(x_train, y_train)

print('Original dataset shape', len(x_train))





s = StandardScaler()



knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit( s.fit_transform(x_train), y_train)

classifier(knn, s.fit_transform(x_smote),s.transform(x_test), y_smote,y_test)






