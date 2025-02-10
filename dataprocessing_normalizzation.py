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

   
