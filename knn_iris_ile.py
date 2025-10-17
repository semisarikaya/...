import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

# ----------------- VERİ -----------------
iris = load_iris()
X = iris.data  # 4 özellik
y = iris.target
class_names = iris.target_names

# Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# ----------------- EN İYİ K BUL -----------------
best_k = 1
best_acc = 0
print("K değerleri test ediliyor:")
for k in range(1, 11):
    knn = MyKNN(k=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"K={k}, Doğruluk={acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nEn iyi K: {best_k}, Doğruluk: {best_acc*100:.2f}%")

# ----------------- PCA İLE 2D -----------------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# En iyi K ile model
best_knn = MyKNN(k=best_k)
best_knn.fit(X_train, y_train)

# ----------------- KARAR SINIRI -----------------
# Grid oluştur
x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# PCA grid noktalarını orijinal 4D uzaya dönüştür
grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
grid_points_4d = pca.inverse_transform(grid_points_2d)

# Tahmin yap
Z = best_knn.predict(grid_points_4d)
Z = Z.reshape(xx.shape)

# ----------------- GRAFİK -----------------
plt.figure(figsize=(12,8))

# Arka plan contour
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)

# Eğitim verisi gri
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c='gray', label='Eğitim verisi',
            alpha=0.5, s=50, edgecolor='k')

# Test verisi renkli
colors = ['red','green','blue']
for i, name in enumerate(class_names):
    plt.scatter(X_test_pca[y_test==i,0], X_test_pca[y_test==i,1],
                color=colors[i],
                label=f"{name} (Test, {np.sum(y_test==i)} örnek)",
                edgecolor='k', s=80)

plt.title("KNN - Iris (4 Özellik, PCA ile 2D + Decision Boundary + Legend)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title=f"Doğruluk: %{best_acc*100:.2f}", loc='lower right',
           fontsize=9, title_fontsize=10)
plt.grid(alpha=0.3)
plt.show()
