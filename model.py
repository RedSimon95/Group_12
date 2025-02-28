import numpy as np
from abc import ABC, abstractmethod

# Classe base per i classificatori
class Classifier(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# Implementazione KNN
class KNNClassifier(Classifier):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # Inizializza un array per le predizioni
        predictions = []
        # itera su ogni punto x nel dataset di test X.
        for x in X_test:
            # Calcola le distanze tra x e tutti i punti di training
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Trova gli indici dei k punti più vicini
            nearest_indices = np.argsort(distances)[:self.k]
            # Prende le etichette dei k punti più vicini
            nearest_labels = self.y_train[nearest_indices]
            # Prende l'etichetta più comune tra i k punti più vicini
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)

# Factory per creare il classificatore
class ClassifierFactory:
    @staticmethod
    def create_classifier(method, param):
        if method == "k-NN":
            return KNNClassifier(k=param)
        else:
            raise ValueError("Classifier not supported")

# Blocco di test per il KNN
def test_knn_classifier():
    print("\n Eseguendo i test su KNNClassifier...\n")

    # Dataset di test
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8]])
    y_train = np.array([0, 0, 1, 1, 1])

    X_test = np.array([[2, 2], [4, 5], [6, 7]])
    y_test = np.array([0, 1, 1])  # Valori attesi

    # Creazione del modello KNN con k=3
    knn = ClassifierFactory.create_classifier("k-NN", 3)
    knn.train(X_train, y_train)
    predictions = knn.predict(X_test)

    print(f"🔹 Predizioni: {predictions.tolist()}")
    print(f"🔹 Valori attesi: {y_test.tolist()}")

    # Controlliamo se le predizioni corrispondono ai valori attesi
    assert len(predictions) == len(y_test), "Errore: Numero di predizioni errato!"
    assert np.array_equal(predictions, y_test), "Errore: Le predizioni non corrispondono ai valori attesi!"
    print("KNNClassifier ha passato tutti i test con successo!")

# Esegue i test solo se il file viene eseguito direttamente
if __name__ == "__main__":
    test_knn_classifier()
    print("\n Tutti i test su KNN sono stati superati con successo! \n")