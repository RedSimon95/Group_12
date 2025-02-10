import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        """
        Inizializza il classificatore k-NN con il numero di vicini k pre-impostato a 3.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Memorizza i dati di training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        """
        Calcola la distanza euclidea tra due punti.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """
        Classifica ogni campione nel set di test.
        """
        predictions = []
        for x in X_test:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]  # Indici dei k vicini più vicini
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Conta le occorrenze delle classi 
            label_counts = {}
            for label in k_nearest_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            
            # Trova la classe con il maggior numero di occorrenze
            max_count = max(label_counts.values())
            most_common = [key for key, value in label_counts.items() if value == max_count]
            
            # Se c'è un pareggio, scegli casualmente
            prediction = np.random.choice(most_common) if len(most_common) > 1 else most_common[0]
            predictions.append(prediction)
        return np.array(predictions)

# Esempio di utilizzo
if __name__ == "__main__":
    # Genera dati di esempio (da sostituire con dataset reale)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[3, 3], [4, 5]])
    
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print("Predizioni:", predictions)
