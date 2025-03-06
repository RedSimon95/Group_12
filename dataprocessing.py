import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
from load_strategy import LoadStrategy, CSVLoadStrategy, JSONLoadStrategy, TextLoadStrategy, ExcelLoadStrategy

# Classe principale per il preprocessing del dataset
class DataProcessor:
    """
    Classe per il preprocessing del dataset: gestione dei valori mancanti, normalizzazione 
    e salvataggio del file elaborato.
    """
    
    def __init__(self, file_path, load_strategy, output_features="processed_features.csv", output_target="processed_target.csv"):
        """
        Inizializza la classe DataProcessor.
        
        Input:
            file_path (str): indica il percorso del file che contiene il dataset da elaborare
            load_strategy (LoadStrategy): strategia di caricamento dei dati
            output_features (str): nome del file CSV dove vengono salvate le features preprocessate
            output_target (str): nome del file CSV dove viene salvato il target preprocessato
        Output:
            None
        """
        self.file_path = file_path
        self.output_features = output_features
        self.output_target = output_target
        self.data = None
        self.features = None
        self.target = None
        self.load_strategy = load_strategy

    def process(self):
        """
        Processa i dati del file fornito in ingresso.

        Input:
            None
        Output:
            self.features (array bidimensionale di float): dataset ripulito e normalizzato senza la colonna delle classi 
            self.target (array di float): colonna delle classi
        """

        """Carica il dataset utilizzando la strategia specificata."""
        self.data = self.load_strategy.load(self.file_path)

   
        """Gestisce i valori mancanti: elimina righe con target NaN e riempie le feature con la media."""
        self.data.dropna(subset=['classtype_v1'], inplace=True)  # Rimuove righe con target NaN
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0 and column != 'classtype_v1':
                self.data[column].fillna(self.data[column].mean(), inplace=True)  # Sostituisce con la media

        # Rimuove le colonne 1, 3 e 9
        self.data.drop(self.data.columns[[0, 2, 8]], axis=1, inplace=True)
    
        """Separa le features dal target e rimuove colonne non necessarie."""
        self.features = self.data.drop(columns=['Sample code number', 'classtype_v1'], errors='ignore')
        self.target = self.data[['classtype_v1']]

    
        """Normalizza le features tra 0 e 1."""
        self.features = (self.features - self.features.min()) / (self.features.max() - self.features.min())

    
        """Salva le features e il target in file CSV separati."""
        self.features.to_csv(self.output_features, index=False)
        self.target.to_csv(self.output_target, index=False)
        print(f"Features salvate in {self.output_features}")
        print(f"Target salvato in {self.output_target}")

    
        return self.features, self.target


