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

if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as e:
        print(f"Test fallito: {e}")