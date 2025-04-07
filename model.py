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
        # Calcola le predizioni e le probabilità per X_test
        # input:    X_test
        # output:   np.array(predictions)       array delle predizioni del classificatore su X_test
        #           np.array(probabilities)     array delle probabilità di assegnazione ad ogni classe

        # Inizializza un array per le predizioni
        predictions = []
        # Inizializza un array per le probabilità
        probabilities = []
        unique_classes = np.unique(self.y_train)
        # itera su ogni punto x nel dataset di test X.
        for x in X_test:
            # Calcola le distanze tra x e tutti i punti di training
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Trova gli indici dei k punti più vicini
            nearest_indices = np.argsort(distances)[:self.k]
            # Prende le etichette dei k punti più vicini
            nearest_labels = self.y_train[nearest_indices].astype(int)  # Converte le etichette in int
            # Prende l'etichetta più comune tra i k punti più vicini
            predictions.append(np.bincount(nearest_labels).argmax())
            # Calcola le probabilità per ciascuna classe (per ogni classe, somma le occorrenze di quella classe in nearest_lables e le divide per k)
            prob = [np.sum(nearest_labels == cls) / self.k for cls in unique_classes]
            # Concatena il risultato precedente a un array di probabilità
            probabilities.append(prob)
        return np.array(predictions), np.array(probabilities)
    

# ------ BLOCCO DI TEST PER IL KNN ------
def test_knn_classifier():
    print("\n Eseguendo i test su KNNClassifier...\n")

    # Dataset di test
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8]])
    y_train = np.array([0, 0, 1, 1, 1])

    X_test = np.array([[2, 2], [4, 5], [6, 7]])
    y_test = np.array([0, 1, 1])  # Valori attesi di predizioni

    prob_attese=np.array([[0.6666666666666666, 0.3333333333333333], [0.3333333333333333, 0.6666666666666666], [0.0, 1.0]]) # Valori attesi di probabilità

    # Creazione del modello KNN con k=3
    knn = KNNClassifier(3)
    knn.train(X_train, y_train)

    # Predizione delle classi e delle probabilità associate
    predictions, probabilities = knn.predict(X_test)

   
    # Verifica se i test sulle predizioni sono steti eseguiti con successo e stampa il risultato
    if np.array_equal(predictions, y_test):
        print("SUCCESSO: Test sulle predizioni eseguito con successo!\n")
    else:
        print("ERRORE: Test sulle predizioni fallito!\n")

    print(f" > Predizioni: {predictions.tolist()}")
    print(f" > Valori attesi: {y_test.tolist()}")

    # Verifica se i test sulle probabilità sono steti eseguiti con successo e stampa il risultato
    if np.array_equal(probabilities, prob_attese):
        print("\nSUCCESSO: Test sulle probabilità eseguito con successo!\n")
    else:
        print("\nERRORE: Test sulle probabilità fallito!\n")

    print(f" > Probabilità: {probabilities.tolist()}")
    print(f" > Valori attesi: {prob_attese.tolist()}")


# Esegue i test solo se il file viene eseguito direttamente
if __name__ == "__main__":
    test_knn_classifier()
