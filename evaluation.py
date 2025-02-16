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
        """
        n_test = int(len(self.X) * test_size)
        indices = np.random.permutation(len(self.X))
        test_indices, train_indices = indices[:n_test], indices[n_test:]
        
        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        results = self.compute_metrics(y_test, y_pred)
        self.logs.append({"Method": "Holdout", **results})
        self._update_confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix()
        return results
