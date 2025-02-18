import pandas as pd
import numpy as np

class DataProcessor:
    """
    Classe per il preprocessing del dataset: gestione dei valori mancanti, normalizzazione 
    e salvataggio del file elaborato.
    """
    
    def __init__(self, file_path, output_features="processed_features.csv", output_target="processed_target.csv"):
        """
        Inizializza la classe DataProcessor.
        
        Parametri:
        1. file_path (str): indica il percorso del file CSV che contiene il dataset da elaborare
        2. output_features (str): nome del file CSV dove vengono salvate le features preprocessate
        3. output_target (str): nome del file CSV dove viene salvato il target preprocessato
        """
        self.file_path = file_path
        self.output_features = output_features
        self.output_target = output_target
        self.data = None
        self.features = None
        self.target = None

    def load_data(self):
        """Carica il dataset e converte i valori numerici."""
        self.data = pd.read_csv(self.file_path)
        self.data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        self.data = self.data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici

    def handle_missing_values(self):
        """Gestisce i valori mancanti: elimina righe con target NaN e riempie le feature con la media."""
        self.data.dropna(subset=['classtype_v1'], inplace=True)  # Rimuove righe con target NaN
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0 and column != 'classtype_v1':
                self.data[column].fillna(self.data[column].mean(), inplace=True)  # Sostituisce con la media

    def separate_features_and_target(self):
        """Separa le features dal target e rimuove colonne non necessarie."""
        self.features = self.data.drop(columns=['Sample code number', 'classtype_v1'], errors='ignore')
        self.target = self.data[['classtype_v1']]

    def normalize_features(self):
        """Normalizza le features tra 0 e 1."""
        self.features = (self.features - self.features.min()) / (self.features.max() - self.features.min())

    def save_processed_data(self):
        """Salva le features e il target in file CSV separati."""
        self.features.to_csv(self.output_features, index=False)
        self.target.to_csv(self.output_target, index=False)
        print(f"Features salvate in {self.output_features}")
        print(f"Target salvato in {self.output_target}")

    def process(self):
        """Esegue l'intero processo di preprocessing."""
        self.load_data()
        self.handle_missing_values()
        self.separate_features_and_target()
        self.normalize_features()
        self.save_processed_data()
        return self.features, self.target
