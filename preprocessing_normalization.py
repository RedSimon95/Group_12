import numpy as np
import pandas as pd

# Caricare il dataset
df = pd.read_csv('dataset.csv')

# Sostituire i '?' con NaN
df.replace('?', np.nan, inplace=True)

# Convertire tutte le colonne in numeriche, sostituendo errori con NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Aprire un file di log per scrivere i risultati
with open("output_log.txt", "w") as log_file:
    # Stampare il numero di righe e colonne iniziali
    log_file.write(f"Righe iniziali: {df.shape[0]}, Colonne: {df.shape[1]}\n")
    print(f"Righe iniziali: {df.shape[0]}, Colonne: {df.shape[1]}")

    # Controllare quanti NaN ci sono nella terz'ultima colonna (classtype)
    nan_count = df.iloc[:, -3].isna().sum()
    log_file.write(f"Valori NaN nella terz'ultima colonna: {nan_count}\n")
    print(f"Valori NaN nella terz'ultima colonna: {nan_count}")

    # Rimuovere le righe con NaN nella terz'ultima colonna (classtype)
    df = df[~df.iloc[:, -3].isna()]

    # Stampare il numero di righe dopo la pulizia
    log_file.write(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {df.shape[0]}\n")
    print(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {df.shape[0]}")

    # Rimuovere le colonne 1, 3, 8
    df.drop(df.columns[[1, 3, 8]], axis=1, inplace=True)

    # Separare la colonna classtype
    classtype_col = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    # Funzione per riempire i NaN con la media della colonna
    def fill_with_mean(col):
        mean_value = col.mean()
        return col.fillna(mean_value)

    # Applicare la funzione a tutte le colonne per riempire i NaN rimanenti
    df = df.apply(fill_with_mean)

    # Normalizzare i valori della matrice tra 1 e 10
    df = df.apply(lambda x:(x - x.min()) / (x.max() - x.min()))

    # Stampare il numero totale di NaN rimasti
    total_nan_after = df.isna().sum().sum()
    log_file.write(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}\n")
    print(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}")

    # Stampare le dimensioni delle matrici ottenute
    print(f"\nDimensioni matrice delle features: {df.shape}")
    print(f"Dimensioni colonna classtype: {classtype_col.shape}")

    # Salvare il dataset pulito
    df.to_csv('output_features.csv', index=False)
    classtype_col.to_csv('output_classtype.csv', index=False, header=['classtype'])

    # Stampare un messaggio di conferma
    print("Il dataset è stato pulito e salvato in due file separati: output_features.csv e output_classtype.csv.")
