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
    
    def _update_confusion_matrix(self, y_true, y_pred):
        """
        Aggiorna la Confusion Matrix cumulativa sulla base delle nuove predizioni.
        :param y_true: Valori reali delle classi
        :param y_pred: Predizioni del modello
        """
        labels = [2, 4]  # Classi presenti nel dataset (benigno=2, maligno=4)
        for true, pred in zip(y_true, y_pred):
            # Usa gli indici corrispondenti alle classi per aggiornare la matrice di confusione
            self.global_conf_matrix[labels.index(true), labels.index(pred)] += 1
    
    def _plot_confusion_matrix(self):
        """
        Genera e visualizza una Confusion Matrix cumulativa.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(self.global_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Benigno (2)', 'Maligno (4)'],
                    yticklabels=['Benigno (2)', 'Maligno (4)'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predetto")
        plt.ylabel("Reale")
        plt.show()
    
    def k_fold_cross_validation(self, k=5):
        """
        Esegue la validazione incrociata K-Fold e aggiorna la Confusion Matrix cumulativa.
        :param k: Numero di fold per la cross-validation (default = 5)
        """
        fold_size = len(self.X) // k  # Calcola la dimensione di ogni fold
        
        if fold_size == 0:
            # Se k è troppo grande rispetto alla dimensione del dataset, si genera un errore
            raise ValueError("Numero di folds troppo alto rispetto alla dimensione del dataset.")
        
        indices = np.random.permutation(len(self.X))  # Mescola casualmente gli indici del dataset
        
        for i in range(k):
            # Divide il dataset in training e test set
            test_indices = indices[i * fold_size: (i + 1) * fold_size]  # Indici per il test set
            train_indices = np.setdiff1d(indices, test_indices)  # Il resto è il training set
            
            # Estrae i dati corrispondenti agli indici
            X_train, X_test = self.X[train_indices], self.X[test_indices]
            y_train, y_test = self.y[train_indices], self.y[test_indices]
            
            # Addestramento del modello
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)  # Predizioni sul test set
            
            # Calcolo delle metriche di valutazione
            results = self.compute_metrics(y_test, y_pred)
            
            # Salva i risultati della fold corrente nei log
            self.logs.append({"Method": f"K-Fold {i+1}/{k}", **results})
            
            # Aggiorna la Confusion Matrix con i risultati della fold corrente
            self._update_confusion_matrix(y_test, y_pred)
        
        # Al termine della K-Fold, visualizza la Confusion Matrix cumulativa
        self._plot_confusion_matrix()
        
        return results  # Restituisce i risultati dell'ultima fold
    
    def stratified_shuffle_split(self, test_size=0.2, n_splits=5):
        """
        Esegue Stratified Shuffle Split per suddividere i dati in training e test set 
        in modo stratificato, aggiornando la Confusion Matrix cumulativa.
        
        Parametri:
        test_size (float): Percentuale di dati da destinare al test set.
        n_splits (int): Numero di suddivisioni da effettuare.
        """
        for i in range(n_splits):
            indices = np.random.permutation(len(self.X))  # Genera una permutazione casuale degli indici
            n_test = int(len(self.X) * test_size)  # Calcola la dimensione del test set
            test_indices, train_indices = indices[:n_test], indices[n_test:]  # Divide gli indici in training e test
            
            X_train, X_test = self.X[train_indices], self.X[test_indices]  # Separa i dati
            y_train, y_test = self.y[train_indices], self.y[test_indices]  # Separa le etichette
            
            self.model.fit(X_train, y_train)  # Allena il modello sui dati di training
            y_pred = self.model.predict(X_test)  # Effettua le predizioni sul test set
            
            results = self.compute_metrics(y_test, y_pred)  # Calcola le metriche di valutazione
            self.logs.append({"Method": f"Stratified Shuffle {i+1}/{n_splits}", **results})  # Salva i risultati
            self._update_confusion_matrix(y_test, y_pred)  # Aggiorna la Confusion Matrix cumulativa
        
        self._plot_confusion_matrix()  # Visualizza la Confusion Matrix finale
        return results
    
    def save_logs(self, filename="evaluation_logs.xlsx"):
        """Salva i risultati di tutte le valutazioni in un file Excel."""
        df_logs = pd.DataFrame(self.logs)
        df_logs.to_excel(filename, index=False)  # Esporta i log in un file Excel
        print(f"📊 Log delle valutazioni salvato in {filename}")