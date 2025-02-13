import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, input_file, output_log, output_features, output_classtype):
        # Inizializza i percorsi dei file di input e output
        self.input_file = input_file
        self.output_log = output_log
        self.output_features = output_features
        self.output_classtype = output_classtype
        # Legge il file CSV di input in un DataFrame
        self.df = pd.read_csv(input_file)
        self.classtype_col = None

    def process_data(self):
        # Apre il file di log per scrivere i dettagli del processo
        with open(self.output_log, "w") as log_file:
            # Scrive e stampa il numero di righe e colonne iniziali
            log_file.write(f"Righe iniziali: {self.df.shape[0]}, Colonne: {self.df.shape[1]}\n")
            print(f"Righe iniziali: {self.df.shape[0]}, Colonne: {self.df.shape[1]}")

            # Conta e logga il numero di valori NaN nella terz'ultima colonna
            nan_count = self.df.iloc[:, -3].isna().sum()
            log_file.write(f"Valori NaN nella terz'ultima colonna: {nan_count}\n")
            print(f"Valori NaN nella terz'ultima colonna: {nan_count}")
            # Rimuove le righe con valori NaN nella terz'ultima colonna
            self.df = self.df[~self.df.iloc[:, -3].isna()]

            # Scrive e stampa il numero di righe dopo la rimozione dei NaN
            log_file.write(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {self.df.shape[0]}\n")
            print(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {self.df.shape[0]}")

            # Rimuove le colonne specificate (0, 2, 7)
            self.df.drop(self.df.columns[[0, 2, 7]], axis=1, inplace=True)

            # Estrae la colonna classtype e rimuove l'ultima colonna
            self.classtype_col = self.df.iloc[:, -3]
            self.df = self.df.iloc[:, :-1]

            # Riempie i valori NaN con la media della colonna
            self.df = self.df.apply(self.fill_with_mean)

            # Seleziona solo le colonne numeriche e normalizza i valori nel range [0, 1]
            X_numeric = self.df.select_dtypes(include=[np.number])
            X_normalized = (X_numeric - X_numeric.min()) / (X_numeric.max() - X_numeric.min())
            X_normalized = X_normalized.clip(0, 1)

            # Conta e logga il numero totale di valori NaN rimasti
            total_nan_after = self.df.isna().sum().sum()
            log_file.write(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}\n")
            print(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}")

            # Stampa le dimensioni della matrice delle features e della colonna classtype
            print(f"\nDimensioni matrice delle features: {X_normalized.shape}")
            print(f"Dimensioni colonna classtype: {self.classtype_col.shape}")

            # Salva le features normalizzate e la colonna classtype nei file di output
            X_normalized.to_csv(self.output_features, index=False)
            self.classtype_col.to_csv(self.output_classtype, index=False, header=['classtype'])

        print("Preprocessing completato. I valori normalizzati sono nel range [0,1].")

    @staticmethod
    def fill_with_mean(col):
        # Riempie i valori NaN con la media della colonna
        mean_value = col.mean()
        return col.fillna(mean_value)

# Esempio di utilizzo della classe
if __name__ == "__main__":
    # Crea un'istanza della classe DataProcessor e avvia il processo di dati
    processor = DataProcessor('dataset.csv', 'output_log.txt', 'output_features.csv', 'output_classtype.csv')
    processor.process_data()