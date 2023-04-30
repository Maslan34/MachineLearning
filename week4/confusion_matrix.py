import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix



# Veri yükleme ve bölme
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Model eğitimi
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Tahminler
y_pred = clf.predict(X_test)

# Confusion matrix oluşturma
cm = confusion_matrix(y_test, y_pred)

# Görselleştirme
labels = ['Setosa', 'Versicolor', 'Virginica']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues) # annot=True parametresi sayesinde hücrelerin içindeki sayıları da gösteriyoruz.
plt.xlabel('Tahmin edilen')
plt.ylabel('Gerçek')
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Veri oluşturma
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

# Model eğitimi
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Tahminler
y_pred = clf.predict(X_test)

# Confusion matrix oluşturma
cm = confusion_matrix(y_test, y_pred)

# Görselleştirme
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negatif', 'Pozitif'], rotation=45)
plt.yticks(tick_marks, ['Negatif', 'Pozitif'])
plt.xlabel('Tahmin edilen etiket')
plt.ylabel('Gerçek etiket')
plt.text(0, 0, f"True Negative: {tn}", ha="center", va="center", color="white", fontsize=12)
plt.text(0, 1, f"False Positive: {fp}", ha="center", va="center", color="black", fontsize=12)
plt.text(1, 0, f"False Negative: {fn}", ha="center", va="center", color="black", fontsize=12)
plt.text(1, 1, f"True Positive: {tp}", ha="center", va="center", color="white", fontsize=12)
plt.text(2.5, 0, f"Precision: {precision:.2f}", ha="center", va="center", color="black", fontsize=12)
plt.text(2.5, -0.2, f"Recall: {recall:.2f}", ha="center", va="center", color="black", fontsize=12)
plt.text(2.5, -0.4, f"F1 Score: {f1_score:.2f}", ha="center", va="center", color="black", fontsize=12)
plt.show()


