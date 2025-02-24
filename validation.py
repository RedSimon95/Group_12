from abc import ABC, abstractmethod
import numpy as np

# Interfaccia comune per tutte le strategie di validazione
class ValidationStrategy(ABC):
    @abstractmethod
    def split_data(self, X, y):
        pass

# Strategia Holdout
class HoldoutValidation(ValidationStrategy):
    def __init__(self, train_size=0.8):
        self.train_size = train_size

    def split_data(self, X, y):
        # Se train_size = 0.8 e len(X) = 100, allora split = int(100 * 0.8) = 80.
        split = int(len(X) * self.train_size)
        # In ordine restituisce X_train, X_test, y_train, y_test
        return X[:split], X[split:], y[:split], y[split:] 

# Strategia K-Fold
class KFoldValidation(ValidationStrategy):
    def __init__(self, k=5):
        self.k = k

    def split_data(self, X, y):
        # Inizializza un array di indici da 0 a len(X) e li mescola
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        # Dividi gli indici in k parti
        folds = np.array_split(indices, self.k)
        return folds

# Factory per selezionare la strategia appropriata
class ValidationFactory:
    @staticmethod
    def get_strategy(method, param=None):
        if method == "Holdout":
            return HoldoutValidation(train_size=param)
        elif method == "K-Fold":
            return KFoldValidation(k=param)
        else:
            raise ValueError("Validation method not supported")
# Test delle ValidationStrategy
def test_validation_strategy():
    print("\n Eseguendo i test sulle ValidationStrategy...\n")

    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Campioni: {X.tolist()}")
    y = np.array(["A", "A", "B", "B", "A", "B", "B", "A", "A", "B"])
    print(f"Etichette: {y.tolist()}")

    # Test Holdout (train_size=0.6) usando la Factory
    holdout = ValidationFactory.get_strategy("Holdout", 0.4)
    X_train, X_test, y_train, y_test = holdout.split_data(X, y)

    assert len(X_train) == 4, "Errore: Holdout non divide correttamente il dataset"
    assert len(X_test) == 6, "Errore: Holdout non divide correttamente il dataset"
    print(f"Training set: {X_train.tolist()} | Test set: {X_test.tolist()} | Training labels: {y_train.tolist()} | Test labels: {y_test.tolist()}")
    print("HoldoutValidation funziona correttamente.")

    # Test K-Fold (imposta k in base al numero di fold desiderati)
    kfold = ValidationFactory.get_strategy("K-Fold", 5)
    folds = kfold.split_data(X, y)

    print("\n Suddivisione K-Fold dei dati:")
    for i, fold in enumerate(folds):
        valori_fold = [int(X[j]) for j in fold]  # Converte np.int64 in int
        etichette_fold = [str(y[j]) for j in fold]  # Converte np.str_ in str
        print(f"Fold {i+1}: Indici {fold.tolist()}, Valori {valori_fold}, Etichette {etichette_fold}")

# Esegue i test solo se il file viene eseguito direttamente
if __name__ == "__main__":
    test_validation_strategy()
    print("\n Tutti i test sono stati superati con successo! \n")