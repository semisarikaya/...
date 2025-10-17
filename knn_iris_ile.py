import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ----------------- KNN SINIFI -----------------
class MyKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ----------------- VERİYİ YÜKLE -----------------
iris = load_iris()
X = iris.data  # 4 özellik: sepal + petal
y = iris.target
class_names = iris.target_names

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

# ----------------- EN İYİ K DEĞERİNİ BUL -----------------
best_k = 1
best_acc = 0
for k in range(1, 11):
    knn = MyKNN(k=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"K={k}, Doğruluk={acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nEn iyi K: {best_k}, Doğruluk: {best_acc*100:.2f}%")

# ----------------- EN İYİ K İLE MODEL -----------------
best_knn = MyKNN(k=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

# ----------------- GRAFİK (İKİ ÖZELLİK İLE) -----------------
# 2 boyutlu görselleştirme için ilk 2 özelliği kullanıyoruz
X_plot = X[:, :2]
y_plot = y
best_knn.fit(X_plot, y_plot)

# Grid oluştur
x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(9,7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)

# Eğitim verisi gri
plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', label='Eğitim verisi', alpha=0.5)

# Test verisini renkli göster
colors = ['red','green','blue']
for i, name in enumerate(class_names):
    plt.scatter(
        X_test[y_pred==i, 0],
        X_test[y_pred==i, 1],
        color=colors[i],
        label=f"{name} (Test, {np.sum(y_pred==i)} örnek)",
        edgecolor='k',
        s=50
    )

# Legend (doğruluk ve sınıf sayısı)
plt.legend(title=f"Doğruluk: %{best_acc*100:.2f}", loc='lower right', fontsize=9, title_fontsize=10)
plt.title("KNN Sınıflandırma - Iris Veri Seti")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid(alpha=0.3)
plt.show()
