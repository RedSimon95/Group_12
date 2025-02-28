import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os

# Definizione dell'interfaccia della strategia di caricamento
class LoadStrategy(ABC):
    @abstractmethod
    def load(self, file_path):
        """
        Metodo astratto per caricare i dati da un file.
        
        Parametri:
        file_path (str): Il percorso del file da cui caricare i dati.
        
        Restituisce:
        pd.DataFrame: Il dataset caricato.
        """
        pass

# Implementazione della strategia di caricamento da file CSV
class CSVLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file CSV.
        
        Parametri:
        file_path (str): Il percorso del file CSV da cui caricare i dati.
        
        Restituisce:
        pd.DataFrame: Il dataset caricato.
        """
        #Verifica l'esistenza della directory
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici
        return data

# Implementazione della strategia di caricamento da file JSON
class JSONLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file JSON.
        
        Parametri:
        file_path (str): Il percorso del file JSON da cui caricare i dati.
        
        Restituisce:
        pd.DataFrame: Il dataset caricato.
        """
        #Verifica l'esistenza della directory
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_json(file_path)
        data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici
        return data

# Implementazione della strategia di caricamento da file di testo
class TextLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file di testo con delimitatore di tabulazione.
        
        Parametri:
        file_path (str): Il percorso del file di testo da cui caricare i dati.
        
        Restituisce:
        pd.DataFrame: Il dataset caricato.
        """
        #Verifica l'esistenza della directory
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_csv(file_path, delimiter='\t')
        data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici
        return data

# Implementazione della strategia di caricamento da file Excel
class ExcelLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file Excel.
        
        Parametri:
        file_path (str): Il percorso del file Excel da cui caricare i dati.
        
        Restituisce:
        pd.DataFrame: Il dataset caricato.
        """
        #Verifica l'esistenza della directory
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_excel(file_path)
        data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici
        return data

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

# Funzione per selezionare automaticamente la strategia di caricamento
def select_load_strategy():
    if os.path.exists('dataset.csv'):
        return CSVLoadStrategy(), 'dataset.csv'
    elif os.path.exists('dataset.json'):
        return JSONLoadStrategy(), 'dataset.json'
    elif os.path.exists('dataset.txt'):
        return TextLoadStrategy(), 'dataset.txt'
    elif os.path.exists('dataset.xlsx'):
        return ExcelLoadStrategy(), 'dataset.xlsx'
    else:
        raise FileNotFoundError("Nessun file di dataset trovato (CSV, JSON, TXT, XLSX).")

# Esempio di utilizzo con selezione automatica della strategia di caricamento
if __name__ == "__main__":
    load_strategy, file_path = select_load_strategy()
    processor = DataProcessor(file_path, load_strategy)
    processor.process()