import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod

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
        #if not os.path.exists(file_path):
        #    raise FileNotFoundError(f"File not found: {file_path}")
        
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
        #if not os.path.exists(file_path):
        #    raise FileNotFoundError(f"File not found: {file_path}")
        
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
        #if not os.path.exists(file_path):
        #    raise FileNotFoundError(f"File not found: {file_path}")
        
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
        #if not os.path.exists(file_path):
        #    raise FileNotFoundError(f"File not found: {file_path}")
        
        data = pd.read_excel(file_path)
        data.replace("?", np.nan, inplace=True)  # Sostituisce i "?" con NaN
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte in valori numerici
        return data
    

