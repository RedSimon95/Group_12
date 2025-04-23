import pandas as pd
import numpy as np
import os
from load_strategy import LoadStrategy, CSVLoadStrategy, JSONLoadStrategy, TextLoadStrategy, ExcelLoadStrategy

# Classe principale per il preprocessing del dataset
class DataProcessor:
    """
    Classe che si occupa del preprocessing del dataset, inclusa la gestione dei valori mancanti,
    la normalizzazione delle feature numeriche e il salvataggio del dataset trasformato.
    Utilizza il pattern Strategy per il caricamento flessibile da diversi formati (CSV, JSON, TXT, XLSX).
    """
    
    def __init__(self, file_path, load_strategy, output_features="processed_features.csv", output_target="processed_target.csv"):
        """
        Inizializza l'istanza del DataProcessor con il percorso del file da caricare, la strategia di caricamento 
        dei dati e i nomi dei file di output.
        
        Args:
            file_path (str): Percorso del dataset originale.
            load_strategy (LoadStrategy): Strategia per il caricamento (design pattern Strategy).
            output_features (str): Nome del file per salvare le feature preprocessate.
            output_target (str): Nome del file per salvare il target preprocessato.
        """
        self.file_path = file_path
        self.load_strategy = load_strategy
        self.output_features = output_features
        self.output_target = output_target
        self.data = None
        self.features = None
        self.target = None

    def process(self):
        """
        Esegue l'intero processo di preprocessing:
        - Caricamento del dataset con la strategia scelta.
        - Gestione dei valori mancanti.
        - Pulizia e separazione tra feature e target.
        - Normalizzazione delle feature numeriche.
        - Salvataggio dei dati preprocessati su file CSV.

        Returns:
            tuple: (features, target) entrambi come DataFrame pandas
        """
        
        # Carica il dataset utilizzando l'oggetto load_strategy (design pattern per supportare diversi formati di file)
        self.data = self.load_strategy.load(self.file_path)

        # Rimuove le righe in cui il target (classtype_v1) è mancante, poiché non possono essere utilizzate per il training
        self.data.dropna(subset=['classtype_v1'], inplace=True)

        # Per ogni colonna (eccetto il target), sostituisce i valori NaN con la media della colonna
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0 and column != 'classtype_v1':
                self.data[column].fillna(self.data[column].mean(), inplace=True)

        # Rimuove le colonne di indice 0, 2 e 8 (basato su indice, non nome)
        # Spesso utile per eliminare colonne identificative o irrilevanti
        self.data.drop(self.data.columns[[0, 2, 8]], axis=1, inplace=True)
    
        # Separa il target (classtype_v1) dalle feature
        # Tenta anche di rimuovere 'Sample code number' se presente, in quanto probabilmente identificativo
        self.features = self.data.drop(columns=['Sample code number', 'classtype_v1'], errors='ignore')
        self.target = self.data[['classtype_v1']]

        # Normalizza tutte le feature in un range [0, 1] 
        self.features = (self.features - self.features.min()) / (self.features.max() - self.features.min())

        # Salva il dataset trasformato in due file CSV: uno per le feature e uno per il target
        self.features.to_csv(self.output_features, index=False)
        self.target.to_csv(self.output_target, index=False)
        print(f"Features salvate in {self.output_features}")
        print(f"Target salvato in {self.output_target}")

        return self.features, self.target
