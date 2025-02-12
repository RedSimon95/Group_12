import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')
#converto gli elementi ignoti con valori nan
# ?? df.replace('?', np.nan, inplace=True)

# Convertire tutte le colonne in numeriche, sostituendo errori con NaN
# ?? df = df.apply(pd.to_numeric, errors='coerce')

# Stampare il numero di righe e colonne iniziali
with open("output_log.txt", "w") as log_file:
    log_file.write(f"Righe iniziali: {df.shape[0]}, Colonne: {df.shape[1]}\n")
    print(f"Righe iniziali: {df.shape[0]}, Colonne: {df.shape[1]}")

    # Controllare quanti NaN ci sono nella terz'ultima colonna (classtype) e rimuovile
    nan_count = df.iloc[:, -3].isna().sum()
    log_file.write(f"Valori NaN nella terz'ultima colonna: {nan_count}\n")
    print(f"Valori NaN nella terz'ultima colonna: {nan_count}")
    df = df[~df.iloc[:, -3].isna()]

    # Stampare il numero di righe dopo la pulizia
    log_file.write(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {df.shape[0]}\n")
    print(f"Righe dopo la rimozione dei NaN nella terz'ultima colonna: {df.shape[0]}")

    # Rimuovere le colonne 0, 2, 7 e separa la colonna classtype
    df.drop(df.columns[[0, 2, 7]], axis=1, inplace=True)

    classtype_col = df.iloc[:, -3]
    df = df.iloc[:, :-1]

    # Riempimento dei NaN con la media dei valori della stessa colonna
    def fill_with_mean(col):
        mean_value = col.mean()
        return col.fillna(mean_value)
    df = df.apply(fill_with_mean)

    # Convertire solo le colonne numeriche per evitare errori di tipo
    X_numeric = df.select_dtypes(include=[np.number])

    # Normalizzazione manuale delle features tra 0 e 1
    X_normalized = (X_numeric - X_numeric.min()) / (X_numeric.max() - X_numeric.min())
    X_normalized = X_normalized.clip(0, 1)  # Assicura che tutti i valori siano nel range [0, 1]

    # Stampa il numero totale di NaN rimasti dopo le modifiche
    total_nan_after = df.isna().sum().sum()
    log_file.write(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}\n")
    print(f"Valori NaN rimasti dopo la pulizia: {total_nan_after}")

    # Stampare le dimensioni delle matrici ottenute
    print(f"\nDimensioni matrice delle features: {X_normalized.shape}")
    print(f"Dimensioni colonna classtype: {classtype_col.shape}")

    # Salva il dataset pulito dopo le modifiche
    X_normalized.to_csv('output_features.csv', index=False)
    classtype_col.to_csv('output_classtype.csv', index=False, header=['classtype'])

print("Preprocessing completato. I valori normalizzati sono nel range [0,1].")

