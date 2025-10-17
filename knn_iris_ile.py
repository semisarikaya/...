import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time

# ==================== 1️⃣ KNN SINIFI ====================
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
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)


# ==================== 2️⃣ GERÇEK VERİ: IRIS ====================
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print(f"Eğitim veri sayısı: {len(X_train)}, Test veri sayısı: {len(X_test)}")

# ==================== 3️⃣ MODEL EĞİTİMİ ====================
knn = MyKNN(k=5, distance_metric='euclidean', weighted=True)
knn.fit(X_train, y_train)

start_time = time.time()
accuracy = knn.score(X_test, y_test)
end_time = time.time()

print(f"Tahmin süresi: {end_time - start_time:.4f} saniye")
print(f"Doğruluk oranı: {accuracy*100:.2f}%")

# ==================== 4️⃣ GÖRSELLEŞTİRME ====================
# Sadece 2 özelliği alalım (örneğin: sepal length ve sepal width)
X_plot = X[:, :2]
y_plot = y

# Eğitim verisini yeniden fit et (görselleştirme için)
knn.fit(X_plot, y_plot)

# Grafik için grid alanı oluştur
x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Griddeki her nokta için sınıf tahmini yap
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ==================== 5️⃣ GRAFİK ====================
plt.figure(figsize=(9,7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)
scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.Set1, edgecolor='k', s=50)

plt.title(f"Iris Verisi - KNN Sınıflandırma (Doğruluk: {accuracy*100:.2f}%)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

# ==================== 6️⃣ AÇIKLAMA (LEGEND) ====================
handles, _ = scatter.legend_elements()
labels = [f"{i}: {name}" for i, name in enumerate(target_names)]
legend = plt.legend(handles, labels, title="Sınıflar", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.gcf().subplots_adjust(bottom=0.25)  # Alt boşluk aç

plt.show()
