import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ModelEvaluator:
    """
    Classe per la valutazione del modello KNN utilizzando Holdout, K-Fold Cross Validation e Stratified Shuffle Split.
    """
    def __init__(self, model, X, y):
        """
        Inizializza l'evaluator con il modello e i dati.
        :param model: Modello KNN addestrabile.
        :param X: Features.
        :param y: Target.
        """
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.logs = []  # Creo struttura per il logging , utile per i successivi metodi che verranno
        
    def holdout(self, test_size=0.2): 
        """
        Divide i dati in training e test set secondo la percentuale specificata e valuta il modello.
        :param test_size: La frazione dei dati da utilizzare come test set. Default è 0.2 (20% per il test, 80% per il training).
        :return: I risultati della valutazione del modello (metriche).
        """
        
        # Calcola la dimensione del test set in base alla percentuale fornita
        n_test = int(len(self.X) * test_size)
        
        # Crea un array di indici casuali 
        indices = np.random.permutation(len(self.X))
        
        # Separa gli indici in test e training se
        test_indices, train_indices = indices[:n_test], indices[n_test:]
        
        # Utilizza gli indici per dividere X (features) e y (target) in training e test set
        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        
        # Allena il modello sui dati di training
        self.model.fit(X_train, y_train)
        
        # Predice i valori sul test set
        y_pred = self.model.predict(X_test)
        
        # Calcola le metriche di performance confrontando i valori predetti con quelli reali
        results = self.compute_metrics(y_test, y_pred)
        
        # Registra i risultati nel log, associando il metodo di valutazione
        self.logs.append({"Method": "Holdout", **results})
        
        # Aggiorna la matrice di confusione con i risultati attuali
        self._update_confusion_matrix(y_test, y_pred)
        
        # Plotta la matrice di confusione
        self._plot_confusion_matrix()
        
        # Restituisce i risultati ottenuti dalle metriche
        return results
        
    def compute_metrics(self, y_true, y_pred):
        """Calcola le metriche di valutazione."""
        
        # Calcolo dei veri positivi (TP): casi in cui il valore reale e la predizione sono entrambi 4
        tp = np.sum((y_true == 4) & (y_pred == 4))
        # Calcolo dei veri negativi (TN): casi in cui il valore reale e la predizione sono entrambi 2
        tn = np.sum((y_true == 2) & (y_pred == 2))
        # Calcolo dei falsi positivi (FP): casi in cui il valore reale è 2 ma la predizione è 4
        fp = np.sum((y_true == 2) & (y_pred == 4))
        # Calcolo dei falsi negativi (FN): casi in cui il valore reale è 4 ma la predizione è 2
        fn = np.sum((y_true == 4) & (y_pred == 2))
        
        # Accuratezza: proporzione di previsioni corrette rispetto al totale dei campioni
        accuracy = (tp + tn) / len(y_true)
        # Tasso di errore: complemento dell'accuratezza
        error_rate = 1 - accuracy
        # Sensibilità (Recall o True Positive Rate): capacità del modello di identificare correttamente i positivi
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificità (True Negative Rate): capacità del modello di identificare correttamente i negativi
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Media geometrica della sensibilità e specificità, utile per dataset sbilanciati
        geometric_mean = np.sqrt(sensitivity * specificity)
        
        # Restituisce un dizionario con tutte le metriche calcolate
        return {
            "Accuracy": accuracy,
            "Error Rate": error_rate,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Geometric Mean": geometric_mean
        }