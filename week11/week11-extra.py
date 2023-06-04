from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')





data = pd.read_csv('WineQT.csv')#https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
data.head(30)





print(data.dtypes)




data.info()





elbow = [] 
for k in range(1,10):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(data)
    elbow.append(km.inertia_)







sns.set_style("whitegrid")
g=sns.lineplot(x=range(1,10), y=elbow)
  
g.set(xlabel ="Number of cluster (k)", 
      ylabel = "Sum Squared Error", 
      title ='Elbow Method')
  
plt.show()








kmeans = KMeans(n_clusters = 3, random_state = 2)
kmeans.fit(data)








kmeans.cluster_centers_








pred = kmeans.fit_predict(data)
pred








plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(data.iloc[:,0],data.iloc[:,1],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("Ph")
plt.ylabel("Fixed Aciditiy")
      
plt.subplot(1,2,2)   
plt.scatter(data.iloc[:,2],data.iloc[:,3],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[2:4]
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("citric acid")
plt.ylabel("chlorides")
plt.show()