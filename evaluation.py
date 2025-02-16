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
      