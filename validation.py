import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataprocessing import DataProcessor, CSVLoadStrategy  # Import per il preprocessing
from model import ClassifierFactory  # Import per il modello KNN

class ModelEvaluator:
    """
    Classe per la valutazione del modello KNN utilizzando Holdout, K-Fold Cross Validation e Stratified Shuffle Split.
    """
    def _init_(self, model, X, y):
        """
        Inizializza l'evaluator con il modello e i dati.
        :param model: Modello KNN addestrabile.
        :param X: Features.
        :param y: Target.
        """
        # Inizializza le variabili per il modello, le features, il target e per il logging
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y).flatten() 
        self.logs = []  # Struttura per tenere traccia dei risultati
        self.global_conf_matrix = np.zeros((2, 2), dtype=int)  # Matrice di confusione globale

    def holdout(self, test_size=0.2):
        """
        Suddivide il dataset in training e test set secondo il metodo Holdout.
        :param test_size: Percentuale di dati da destinare al test set (default = 0.2).
        """
        # Esegue la divisione casuale tra training e test set
        indices = np.random.permutation(len(self.X))  # Indici casuali per mescolare i dati
        n_test = int(len(self.X) * test_size)  # Numero di campioni nel test set
        test_indices, train_indices = indices[:n_test], indices[n_test:]  # Suddivisione dei dati

        X_train, X_test = self.X[train_indices], self.X[test_indices]  # Features di train e test
        y_train, y_test = self.y[train_indices], self.y[test_indices]  # Target di train e test

        # Allena il modello e predice i risultati sui dati di test
        self.model.train(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Calcola le metriche di valutazione e aggiorna la matrice di confusione
        results = self.compute_metrics(y_test, y_pred)
        self.logs.append({"Method": "Holdout", **results})  # Aggiunge i risultati ai log
        self._update_confusion_matrix(y_test, y_pred)  # Aggiorna la matrice di confusione
        self._plot_confusion_matrix()  # Visualizza la matrice di confusione
        return results  # Restituisce i risultati

    def k_fold_cross_validation(self, k=5):
        """
        Esegue la validazione incrociata K-Fold.
        :param k: Numero di fold per la cross-validation (default = 5).
        """
        # Mescola i dati per assicurarsi che ogni fold sia casuale
        indices = np.random.permutation(len(self.X))
        fold_size = len(self.X) // k  # Calcola la dimensione di ogni fold

        # Esegui la cross-validation per ogni fold
        for i in range(k):
            # Separa i dati in un fold di test e il resto come train
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = self.X[train_indices], self.X[test_indices]  # Features per il train e il test
            y_train, y_test = self.y[train_indices], self.y[test_indices]  # Target per il train e il test

            # Allena il modello sui dati di addestramento e predice sui dati di test
            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Calcola le metriche per il fold corrente e aggiorna la matrice di confusione
            results = self.compute_metrics(y_test, y_pred)
            self.logs.append({"Method": f"K-Fold {i+1}/{k}", **results})  # Aggiunge i risultati ai log
            self._update_confusion_matrix(y_test, y_pred)

        # Dopo tutti i fold, visualizza la matrice di confusione aggregata
        self._plot_confusion_matrix()
        return results  # Restituisce gli ultimi risultati

    def stratified_shuffle_split(self, test_size=0.2, n_splits=5):
        """
        Implementazione manuale di Stratified Shuffle Split usando solo numpy e pandas.
        """
        unique_classes, class_counts = np.unique(self.y, return_counts=True)  # Ottiene le classi uniche

        # Esegui la suddivisione stratificata per un numero definito di split
        for i in range(n_splits):
            train_indices = []
            test_indices = []

            # Per ogni classe, suddivide i dati mantenendo la distribuzione originale
            for cls, count in zip(unique_classes, class_counts):
                cls_indices = np.where(self.y == cls)[0]  # Ottiene gli indici per ciascuna classe
                np.random.shuffle(cls_indices)  # Mescola gli indici della classe

                n_test = int(len(cls_indices) * test_size)  # Numero di dati da destinare al test
                test_indices.extend(cls_indices[:n_test])  # Aggiungi gli indici al test set
                train_indices.extend(cls_indices[n_test:])  # Aggiungi gli altri indici al train set

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            X_train, X_test = self.X[train_indices], self.X[test_indices]  # Features di train e test
            y_train, y_test = self.y[train_indices], self.y[test_indices]  # Target di train e test

            # Allena il modello e predice sui dati di test
            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Calcola le metriche per lo split corrente e aggiorna la matrice di confusione
            results = self.compute_metrics(y_test, y_pred)
            self.logs.append({"Method": f"Stratified Shuffle {i+1}/{n_splits}", **results})
            self._update_confusion_matrix(y_test, y_pred)

        # Dopo tutti gli split, visualizza la matrice di confusione aggregata
        self._plot_confusion_matrix()
        return results  # Restituisce gli ultimi risultati

# Caricamento e preprocessing del dataset
# Viene caricato il dataset tramite la classe DataProcessor che gestisce il caricamento e la preparazione dei dati
data_processor = DataProcessor("dataset.csv", CSVLoadStrategy())
features, target = data_processor.process()  # Estrae le features e il target dal dataset

# Creazione del modello KNN
# Utilizza il factory pattern per creare un classificatore KNN con k=3
knn_model = ClassifierFactory.create_classifier("k-NN", 3)

# Valutazione del modello
# Viene inizializzata la classe ModelEvaluator per eseguire la valutazione del modello con i vari metodi
model_evaluator = ModelEvaluator(knn_model, features, target)
model_evaluator.holdout()  # Esegui la valutazione tramite il metodo Holdout
model_evaluator.k_fold_cross_validation()  # Esegui la valutazione tramite K-Fold Cross Validation
model_evaluator.stratified_shuffle_split()  # Esegui la valutazione tramite Stratified Shuffle Split