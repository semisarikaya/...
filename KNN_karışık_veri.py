import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ==================== 1️⃣ GELİŞMİŞ KNN SINIFI ====================
class MyKNN:
    def __init__(self, k=5, distance_metric='euclidean', weighted=True, p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.p = p

    def fit(self, X_train, y_train):
        self.min = X_train.min(axis=0)
        self.max = X_train.max(axis=0)
        self.X_train = (X_train - self.min) / (self.max - self.min + 1e-8)
        self.y_train = y_train

    def _compute_distances(self, X_test):
        X_test = (X_test - self.min) / (self.max - self.min + 1e-8)
        if self.distance_metric == 'euclidean':
            X_test_sq = np.sum(X_test**2, axis=1).reshape(-1,1)
            X_train_sq = np.sum(self.X_train**2, axis=1).reshape(1,-1)
            cross = X_test @ self.X_train.T
            dists = np.sqrt(X_test_sq + X_train_sq - 2*cross)
        elif self.distance_metric == 'manhattan':
            dists = np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        elif self.distance_metric == 'minkowski':
            dists = np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])**self.p, axis=2)**(1/self.p)
        else:
            raise ValueError("distance_metric must be 'euclidean', 'manhattan', or 'minkowski'")
        return dists

    def predict(self, X_test):
        dists = self._compute_distances(X_test)
        predictions = []
        for i in range(X_test.shape[0]):
            k_indices = np.argsort(dists[i])[:self.k]
            k_labels = [self.y_train[j] for j in k_indices]
            if self.weighted:
                k_distances = dists[i][k_indices]
                weights = 1 / (k_distances + 1e-8)
                vote = {}
                for label, w in zip(k_labels, weights):
                    vote[label] = vote.get(label, 0) + w
                predicted_label = max(vote, key=vote.get)
            else:
                predicted_label = Counter(k_labels).most_common(1)[0][0]
            predictions.append(predicted_label)
        return np.array(predictions)

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)

# ==================== 2️⃣ KARMAŞIK RASTGELE VERİ ====================
np.random.seed(42)
n_samples = 7000
n_features = 4
n_classes = 3

centers = np.array([[2,2,2,2],[3,3,3,3],[4,4,4,4]])
X = np.zeros((n_samples, n_features))
y = np.zeros(n_samples, dtype=int)

for i in range(n_classes):
    start = i * (n_samples // n_classes)
    end = (i+1) * (n_samples // n_classes) if i < n_classes-1 else n_samples
    X[start:end] = centers[i] + np.random.randn(end-start, n_features) * 1.2
    y[start:end] = i

class_names = [f"Class {i}" for i in range(n_classes)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print(f"Eğitim veri sayısı: {len(X_train)}, Test veri sayısı: {len(X_test)}")

# ==================== 3️⃣ K OPTİMİZASYONU ====================
best_k = 1
best_acc = 0
k_range = range(1,11)
accuracy_list = []

for k in k_range:
    knn = MyKNN(k=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracy_list.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"En iyi K: {best_k}, Doğruluk: {best_acc*100:.2f}%")

# ==================== 4️⃣ PCA 2D ====================
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

best_knn = MyKNN(k=best_k)
best_knn.fit(X_train, y_train)

# ==================== 5️⃣ KARAR SINIRI ====================
x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min,x_max,50), np.linspace(y_min,y_max,50))

grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
grid_points_4d = pca.inverse_transform(grid_points_2d)
Z = best_knn.predict(grid_points_4d)
Z = Z.reshape(xx.shape)

# ==================== 6️⃣ GÖRSELLEŞTİRME ====================
plt.figure(figsize=(12,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c='gray', label='Eğitim verisi', alpha=0.5, s=50)

colors = ['red','green','blue']
for i, name in enumerate(class_names):
    plt.scatter(X_test_pca[y_test==i,0], X_test_pca[y_test==i,1],
                color=colors[i],
                label=f"{name} (Test, {np.sum(y_test==i)} örnek)",
                s=80)

plt.title("KNN - Hafifletilmiş Karışık Veri (PCA 2D + Decision Boundary)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title=f"Doğruluk: %{best_acc*100:.2f}", loc='lower right', fontsize=9, title_fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# ==================== 7️⃣ K DEĞERİNE GÖRE DOĞRULUK ====================
plt.figure(figsize=(8,5))
plt.plot(k_range, [a*100 for a in accuracy_list], marker='o')
plt.xticks(k_range)
plt.xlabel("K Değeri")
plt.ylabel("Doğruluk (%)")
plt.title("K Değerine Göre Doğruluk")
plt.grid(alpha=0.3)
plt.show()
