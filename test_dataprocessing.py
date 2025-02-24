import unittest
import pandas as pd
from io import StringIO
import os

from dataprocessing import DataProcessor, CSVLoadStrategy  

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        """
        Prepara un dataset di test con 10 righe simulando un file CSV.
        Questo dataset viene utilizzato per verificare il comportamento del metodo `process` della classe `DataProcessor`.
        """
        # Creazione di un dataset di test come stringa CSV
        self.test_csv_data = """Sample code number,Feature1,Feature2,Feature3,Feature4,Feature5,Feature6,Feature7,Feature8,classtype_v1
        1000025,5,1,1,1,2,1,3,1,2
        1002945,5,4,4,5,7,10,3,2,2
        1015425,3,1,1,1,2,2,3,1,2
        1016277,6,8,8,1,3,4,3,7,2
        1017023,4,1,1,3,2,1,3,1,2
        1017122,8,10,10,8,7,10,9,7,4
        1018099,1,1,1,1,2,10,3,1,2
        1018561,2,1,2,1,2,1,3,1,2
        1033078,2,1,1,1,2,1,3,1,2
        1033078,4,2,1,1,2,1,3,1,2
        """
        # Simulazione di un file CSV usando StringIO
        self.mock_file = StringIO(self.test_csv_data)
        
    def test_process(self):
        """
        Testa il metodo `process` della classe `DataProcessor`.
        Questo test verifica che:
        - I dati vengano caricati correttamente e preprocessati.
        - Le colonne previste vengano rimosse.
        - I valori delle feature siano normalizzati tra 0 e 1.
        - I file di output siano generati correttamente.
        """
    
        # Scrive il dataset di test in un file temporaneo
        with open("test_dataset.csv", "w") as f:
            f.write(self.test_csv_data)

        # Creazione di un'istanza di DataProcessor con la strategia di caricamento CSV
        load_strategy = CSVLoadStrategy()
        processor = DataProcessor("test_dataset.csv", load_strategy, "test_features.csv", "test_target.csv")
        
        # Esegue il metodo process per trasformare i dati
        features, target = processor.process()

        # Controlla che `features` e `target` siano oggetti DataFrame di pandas
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.DataFrame)
        
        # Verifica che le colonne rimosse siano corrette
        expected_columns = ['Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7']
        self.assertListEqual(list(features.columns), expected_columns)

        # Controlla che tutti i valori delle feature siano normalizzati tra 0 e 1
        self.assertTrue(((features >= 0) & (features <= 1)).all().all())
        
        # Controlla che i file di output siano stati creati correttamente
        self.assertTrue(os.path.exists("test_features.csv"))
        self.assertTrue(os.path.exists("test_target.csv"))

        # Stampa un messaggio di successo
        print("Test riuscito: Il metodo `process` ha funzionato correttamente.")

        # Rimuove i file temporanei dopo il test per evitare accumulo di file inutili
        self.addCleanup(os.remove, "test_dataset.csv")
        self.addCleanup(os.remove, "test_features.csv")
        self.addCleanup(os.remove, "test_target.csv")

if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as e:
        print(f"Test fallito: {e}")