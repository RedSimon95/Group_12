from abc import ABC, abstractmethod
import numpy as np 

# Interfaccia comune per tutte le strategie di validazione
class ValidationStrategy(ABC):
    """
    Classe astratta che definisce un'interfaccia comune per tutte le strategie di validazione.
    Ogni strategia specifica deve implementare il metodo 'split_data' che suddivide i dati in set di addestramento e di test.
    """
    @abstractmethod
    def split_data(self, X, y):
        pass

    def compute_confusion_matrix(self, y_true, y_pred):
        """Calcola la matrice di confusione."""
        # Calcolo dei veri positivi (TP): casi in cui il valore reale e la predizione sono entrambi 4
        tp = np.sum((y_true == 4) & (y_pred == 4))
        # Calcolo dei veri negativi (TN): casi in cui il valore reale e la predizione sono entrambi 2
        tn = np.sum((y_true == 2) & (y_pred == 2))
        # Calcolo dei falsi positivi (FP): casi in cui il valore reale è 2 ma la predizione è 4
        fp = np.sum((y_true == 2) & (y_pred == 4))
        # Calcolo dei falsi negativi (FN): casi in cui il valore reale è 4 ma la predizione è 2
        fn = np.sum((y_true == 4) & (y_pred == 2))
        
        # Restituisce la matrice di confusione
        return np.array([[tn, fp], [fn, tp]])

    def compute_metrics(self, y_true, y_pred):
        """Calcola le metriche di valutazione."""
        # Calcola la matrice di confusione
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.ravel()
        
        # Accuratezza: proporzione di previsioni corrette rispetto al totale dei campioni
        accuracy = (tp + tn) / len(y_true)
        # Tasso di errore: complemento dell'accuratezza
        error_rate = 1 - accuracy
        # Sensibilità (Recall o True Positive Rate): capacità del modello di identificare correttamente i positivi
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificità (True Negative Rate): capacità del modello di identificare correttamente i negativi
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Media geometrica della sensibilità e specificità, utile per dataset sbilanciati
        geometric_mean = np.sqrt(sensitivity * specificity)
        
        # Restituisce un dizionario con tutte le metriche calcolate
        return {
            "Confusion Matrix": confusion_matrix,
            "Accuracy": f"{accuracy:.4f}",
            "Error Rate": f"{error_rate:.4f}",
            "Sensitivity (Recall)": f"{sensitivity:.4f}",
            "Specificity": f"{specificity:.4f}",
            "Geometric Mean": f"{geometric_mean:.4f}"
        }

# Strategia Holdout
class HoldoutValidation(ValidationStrategy):
    """
    Questa strategia suddivide il dataset in due set (training e test) utilizzando una percentuale di divisione
    specificata dall'utente. I dati vengono prima mescolati per garantire una distribuzione casuale.
    """
    def __init__(self, train_size=0.8):
        # Imposta la percentuale di dati da utilizzare per l'addestramento
        self.train_size = train_size

    def split_data(self, X, y):
        # Mescola gli indici dei dati per garantire la casualità
        indices = np.random.permutation(len(X))  
        
        # Calcola la posizione dove i dati saranno divisi in training e test set
        split = int(len(X) * self.train_size) 
        
        # Suddivide gli indici dei dati in due gruppi: uno per il training e uno per il test
        train_indices, test_indices = indices[:split], indices[split:]  
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Strategia K-Fold
class KFoldValidation(ValidationStrategy):
    """
    Questa strategia suddivide il dataset in k sottoinsiemi, chiamati "folds", 
    e usa ogni sottoinsieme come test set mentre gli altri fold vengono usati per l'addestramento. 
    Il processo viene ripetuto per tutti i fold.
    """
    def __init__(self, k=5):
        # Imposta il numero di fold per la validazione incrociata
        self.k = k

    def split_data(self, X, y):

        # Mescola gli indici dei dati per garantire la casualità prima di suddividere in fold
        indices = np.random.permutation(len(X))  
        
        # Suddivide gli indici in k parti (fold)
        folds = np.array_split(indices, self.k)  
        return folds

# Strategia Stratified Shuffle Split
class StratifiedShuffleSplitValidation(ValidationStrategy):
    """
    Questa strategia suddivide i dati in training e test set, cercando di mantenere la stessa distribuzione delle classi
    sia nel training set che nel test set. È utile quando le classi sono sbilanciate.
    """
    def __init__(self, test_size=0.2, n_splits=5):
        # Imposta la dimensione del test set e il numero di suddivisioni (splits) da eseguire
        self.test_size = test_size
        self.n_splits = n_splits

    def split_data(self, X, y):
        # Calcola la distribuzione delle classi nel dataset
        unique_classes, class_counts = np.unique(y, return_counts=True)
        splits = []
        
        for _ in range(self.n_splits):
            # Per ogni split, prepariamo gli indici per il training e il test set
            train_indices = []
            test_indices = []
            
            # Per ogni classe, mescoliamo i campioni e li dividiamo tra training e test set
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]  # Ottieni gli indici di una specifica classe
                np.random.shuffle(cls_indices)  # Mescola gli indici di quella classe
                
                # Calcoliamo quanti campioni di questa classe devono andare nel test set
                n_test = int(len(cls_indices) * self.test_size)  
                
                # Selezioniamo i primi indici per il test set e gli altri per il training set
                test_indices.extend(cls_indices[:n_test])  
                train_indices.extend(cls_indices[n_test:])  
            
            # Salviamo le suddivisioni (training e test) per ciascun split
            splits.append((np.array(train_indices), np.array(test_indices)))
        
        return splits

# Factory per la creazione dinamica delle strategie di validazione
class ValidationFactory:
    """
    Questa factory crea le diverse strategie di validazione in base al metodo scelto.
    È utile per centralizzare la logica di creazione delle strategie senza dover ripetere codice.
    """
    @staticmethod
    def get_strategy(method, param=None):
        if method == "Holdout":
            # Crea una strategia Holdout con la dimensione del training set specificata
            return HoldoutValidation(train_size=param)
        elif method == "K-Fold":
            # Crea una strategia K-Fold con il numero di fold specificato
            return KFoldValidation(k=param)
        elif method == "Stratified Shuffle":
            # Crea una strategia Stratified Shuffle Split con la dimensione del test set specificata
            return StratifiedShuffleSplitValidation(test_size=param)
        else:
            raise ValueError("Validation method not supported")
        
# Funzione di test per verificare la correttezza delle strategie di validazione
def test_validation_strategy():
    """
    Testa tutte le strategie di validazione (Holdout, K-Fold, Stratified Shuffle Split) su un piccolo dataset.
    La funzione stampa i risultati per ogni strategia, verificando se i dati sono suddivisi correttamente.
    """
    print("\nEseguendo i test sulle ValidationStrategy...\n")
    
    # Creazione di un piccolo dataset di esempio per testare le strategie
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Campioni: {X.tolist()}")
    y = np.array(["A", "A", "B", "B", "A", "B", "B", "A", "A", "B"])
    print(f"Etichette: {y.tolist()}")

    # Test per la strategia Holdout
    holdout = ValidationFactory.get_strategy("Holdout", 0.4)
    X_train, X_test, y_train, y_test = holdout.split_data(X, y)
    
    
    assert len(X_train) == 4, "Errore: Holdout non divide correttamente il dataset"
    assert len(X_test) == 6, "Errore: Holdout non divide correttamente il dataset"
    print(f"Holdout - Training set: {X_train.tolist()}, Test set: {X_test.tolist()}, Training labels: {y_train.tolist()}, Test labels: {y_test.tolist()}")
    
    # Test per la strategia K-Fold
    kfold = ValidationFactory.get_strategy("K-Fold", 5)
    folds = kfold.split_data(X, y)

    print("\n Suddivisione K-Fold dei dati:")
    for i, fold in enumerate(folds):
        X_fold_train = X[fold]
        y_fold_train = y[fold]
        print(f"Fold {i+1}: Training set: {X_fold_train.tolist()}, Training labels: {y_fold_train.tolist()}")
    
    # Test per la strategia Stratified Shuffle Split
    stratified = ValidationFactory.get_strategy("Stratified Shuffle", 0.4)
    splits = stratified.split_data(X, y)
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train_split = X[train_idx]
        X_test_split = X[test_idx]
        y_train_split = y[train_idx]
        y_test_split = y[test_idx]
        print(f"Stratified Shuffle {i+1}: Training set: {X_train_split.tolist()}, Test set: {X_test_split.tolist()}, Training labels: {y_train_split.tolist()}, Test labels: {y_test_split.tolist()}")

# Eseguiamo i test se il file viene eseguito direttamente
if __name__ == "__main__":
    test_validation_strategy()
