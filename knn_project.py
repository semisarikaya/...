import numpy as np
from collections import Counter
import time

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

#  Zor Sentetik Veri
np.random.seed(42)
num_train_per_class = 500
num_test_per_class = 100
num_features = 5

# Sınıf merkezleri birbirine yakın
centers = {
    'A': np.array([10, 10, 10, 10, 10]),
    'B': np.array([20, 20, 20, 20, 20]),
    'C': np.array([30, 30, 30, 30, 30]),
    'D': np.array([40, 40, 40, 40, 40])
}

# Eğitim verisi
X_train, y_train = [], []
for label, center in centers.items():
    X_train.append(center + np.random.randn(num_train_per_class, num_features)*8)  # daha fazla gürültü
    y_train += [label]*num_train_per_class
X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Test verisi
X_test, y_test = [], []
for label, center in centers.items():
    X_test.append(center + np.random.randn(num_test_per_class, num_features)*8)
    y_test += [label]*num_test_per_class
X_test = np.vstack(X_test)
y_test = np.array(y_test)

#  KNN Test
knn = MyKNN(k=5, distance_metric='euclidean', weighted=True)
knn.fit(X_train, y_train)

start_time = time.time()
predictions = knn.predict(X_test)
end_time = time.time()

accuracy = knn.score(X_test, y_test)
print(f"Tahmin süresi: {end_time - start_time:.2f} saniye")
print("Doğruluk:", accuracy)
print("Tahminler (ilk 10):", predictions[:10])
