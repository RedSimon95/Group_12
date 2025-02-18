import numpy as np

class KNNClassifier:
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Memorizza il dataset di training."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        """Calcola la distanza euclidea tra due punti."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """Predice le etichette per i dati di test."""
        return np.array([self._predict_instance(x) for x in np.array(X_test)])

    def _predict_instance(self, x):
        """Predice la classe di un singolo campione."""
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        max_count = max(label_counts.values())
        candidates = [label for label, count in label_counts.items() if count == max_count]
        return np.random.choice(candidates) if len(candidates) > 1 else candidates[0]
