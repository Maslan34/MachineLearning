from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np



iris = load_iris()



X, y = iris.data, iris.target



kf = KFold(n_splits=5, shuffle=True, random_state=42)



model = LinearRegression()



scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Test seti üzerinde tahmin yap ve hata hesapla
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    scores.append(score)


mean_score = np.mean(scores)


print("K-fold cross validation result: ", scores)
print("Mean Score: ", mean_score)