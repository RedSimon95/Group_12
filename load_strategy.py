import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod

# INTERFACCIA ASTRATTA PER LE STRATEGIE DI CARICAMENTO

class LoadStrategy(ABC):
    """
    Classe astratta che definisce l'interfaccia per le strategie di caricamento dei dati.
    Utilizza il design pattern Strategy per permettere la flessibilità nel supporto di diversi formati di file.
    """
    
    @abstractmethod
    def load(self, file_path):
        """
        Metodo astratto che ogni strategia concreta deve implementare.
        
        Args:
            file_path (str): Il percorso del file da cui caricare i dati.
        
        Returns:
            pd.DataFrame: Dataset caricato sotto forma di DataFrame.
        """
        pass

# STRATEGIA CONCRETA: CARICAMENTO DA FILE CSV

class CSVLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file CSV, gestendo valori mancanti rappresentati con "?" e 
        cercando di convertire tutti i dati in formato numerico.
        """
        data = pd.read_csv(file_path)
        data.replace("?", np.nan, inplace=True)  # Sostituisce "?" con NaN per gestire i missing values
        data = data.apply(pd.to_numeric, errors="coerce")  # Converte colonne in numeriche, coerentemente con Pandas
        return data

# STRATEGIA CONCRETA: CARICAMENTO DA FILE JSON

class JSONLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file JSON e li tratta per la conversione in formato numerico.
        Simile alla strategia per CSV ma specifica per JSON.
        """
        data = pd.read_json(file_path)
        data.replace("?", np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors="coerce")
        return data

# STRATEGIA CONCRETA: CARICAMENTO DA FILE DI TESTO (delimitato da tabulazione)

class TextLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica dati da file di testo in formato tab-delimited (.txt).
        Questa strategia è utile per dati esportati da altri software come file separati da tabulazioni.
        """
        data = pd.read_csv(file_path, delimiter='\t')
        data.replace("?", np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors="coerce")
        return data

# STRATEGIA CONCRETA: CARICAMENTO DA FILE EXCEL

class ExcelLoadStrategy(LoadStrategy):
    def load(self, file_path):
        """
        Carica i dati da un file Excel (.xlsx), spesso usato per dati amministrativi o scientifici.
        Converte anche qui i dati in formato numerico dove possibile.
        """
        data = pd.read_excel(file_path)
        data.replace("?", np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors="coerce")
        return data

# CLASSE DI UTILITÀ PER LA SELEZIONE AUTOMATICA DELLA STRATEGIA

class SelectLoadStrategy:
    """
    Classe helper che sceglie la strategia di caricamento più adatta in base all'estensione del file fornito.
    Serve per astrarre la logica di selezione dal codice principale.
    """
    
    @staticmethod
    def select_load_strategy(file_path):
        """
        Seleziona e restituisce l'istanza della strategia di caricamento adeguata in base all'estensione del file.
        
        Args:
            file_path (str): Percorso del file con estensione.
        
        Returns:
            tuple: (istanza di LoadStrategy, percorso file eventualmente ripulito)
        """
        # In alcuni casi il path può essere tra virgolette, le rimuoviamo
        if file_path.find("\"") != -1:
            file_path = file_path.split("\"")[1]

        # Determina la strategia da usare in base all'estensione
        if file_path.endswith('.csv'):
            return CSVLoadStrategy(), file_path
        elif file_path.endswith('.json'):
            return JSONLoadStrategy(), file_path
        elif file_path.endswith('.txt'):
            return TextLoadStrategy(), file_path
        elif file_path.endswith('.xlsx'):
            return ExcelLoadStrategy(), file_path

        # Se nessun formato corrisponde, solleva errore
        raise FileNotFoundError("Nessun file di dataset trovato (CSV, JSON, TXT, XLSX).")
