import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from load_strategy import LoadStrategy, CSVLoadStrategy, JSONLoadStrategy, TextLoadStrategy, ExcelLoadStrategy
from dataprocessing import DataProcessor
from model import Classifier, KNNClassifier
from validation import ModelEvaluator

def main():
    
    print("\n Inserire nella directory un file con una qualsiasi delle estensioni supportate: CSV, JSON, TXT, XLSX")

    # Funzione che seleziona la strategia di caricamento dei dati in base al formato del file
    # Questa funzione scansiona la cartella corrente per trovare un file con estensione supportata
    # e restituisce la strategia di caricamento appropriata.
    def select_load_strategy():
        directory_content=os.listdir('.')
        # Percorso directory corrente str(os.getcwd()).split('\\')[-1]+"/"+nome_file
        for file in directory_content:  # Scansiona la cartella corrente
            if file.endswith('.csv'):
                return CSVLoadStrategy(), file  # Restituisce la strategia per il caricamento di file CSV
            elif file.endswith('.json'):
                return JSONLoadStrategy(), file  # Restituisce la strategia per il caricamento di file JSON
            elif file.endswith('.txt'):
                return TextLoadStrategy(), file  # Restituisce la strategia per il caricamento di file TXT
            elif file.endswith('.xlsx'):
                return ExcelLoadStrategy(), file  # Restituisce la strategia per il caricamento di file XLSX
        raise FileNotFoundError("Nessun file di dataset trovato (CSV, JSON, TXT, XLSX).")
    

    # Selezione della strategia di caricamento e caricamento dei dati
    # Inizializza DataProcessor che gestisce il caricamento dei dati e la loro separazione
    # in features e target.
    load_strategy, file_path = select_load_strategy()  
    processor = DataProcessor(file_path, load_strategy)
    features, target = processor.process()

    X = features.to_numpy()
    y = target.to_numpy().flatten()
    
    k = int(input("Inserisci il numero di vicini (k): ").strip())
    if k <= 0:
        print("\n+++++++++++++++ ERRORE: Il numero di vicini deve essere positivo +++++++++++++++++++")
        return

    # Inizializza il modello KNN con il valore di k specificato
    knn = KNNClassifier(k)

    modelevaluator = ModelEvaluator(knn, X, y)
    
    method = input("Scegli il metodo di valutazione (holdout, kfold, stratified): ").strip().lower()
    
    # Funzione per disegnare la matrice di confusione
    def plot_confusion_matrix(conf_matrix):
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    # Valutazione del modello utilizzando il metodo Holdout (splitting semplice dei dati)
    if method == "holdout":
        # Chiede la percentuale di dati da utilizzare per il test (fra 20% e 50%)
        test_size = float(input("Inserisci la percentuale di test (es. 0.2 per il 20%): ").strip())
        if test_size < 0.2 or test_size > 0.5:
            print("\n ERRORE: Scegli una percentuale tra 20% e 50%")
        else:
            # Applica la validazione Holdout
            X_train, X_test, y_train, y_test = modelevaluator.holdout(test_size)
            knn.train(X_train, y_train)  # Allena il modello KNN sui dati di training
            y_pred = knn.predict(X_test)  # Predice sui dati di test
            metrics = modelevaluator.compute_metrics(y_test, y_pred)  # Calcola le metriche di valutazione
            print("\n Holdout Metrics:\n")
            plot_confusion_matrix(metrics['Confusion Matrix'])  # Visualizza la matrice di confusione

    # Valutazione del modello utilizzando K-Fold Cross Validation
    elif method == "kfold":
        # Chiede il numero di fold per la validazione K-fold
        k_folds = int(input("Inserisci il numero di folds per K-Fold: ").strip())
        if k_folds < 2:
            print("\n ERRORE: Il numero di folds deve essere almeno 2")
            return
        
        # Applica la validazione K-fold
        folds = modelevaluator.k_fold_cross_validation(k_folds) # Divide i dati in folds
        # Inizializza variabili per calcolare le metriche aggregate
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        # Ciclo su ciascun fold per allenare il modello e calcolare le metriche
        for i, fold in enumerate(folds):
            X_fold_train = X[fold]  # Dati di addestramento per il fold corrente
            y_fold_train = y[fold]  # Target di addestramento per il fold corrente
            knn.train(X_fold_train, y_fold_train)  # Allena il modello sui dati del fold
            y_fold_pred = knn.predict(X_fold_train)  # Predice sui dati del fold
            metrics = modelevaluator.compute_metrics(y_fold_train, y_fold_pred)  # Calcola le metriche di valutazione
            # Aggiunge le metriche della matrice di confusione per ciascun fold
            true_positive += metrics['Confusion Matrix'][1, 1]
            true_negative += metrics['Confusion Matrix'][0, 0]
            false_positive += metrics['Confusion Matrix'][0, 1]
            false_negative += metrics['Confusion Matrix'][1, 0]

        # Calcola le metriche medie per tutti i fold
        avg_metrics = {
            'true_positives': true_positive / k_folds,
            'true_negatives': true_negative / k_folds,
            'false_positives': false_positive / k_folds,
            'false_negatives': false_negative / k_folds
        }

        # Crea una matrice di confusione media per i fold
        avgmatrix = [[int(avg_metrics['true_positives']), int(avg_metrics['false_positives'])], 
                     [int(avg_metrics['false_negatives']), int(avg_metrics['true_negatives'])]]
        print("\n K-Fold Average Metrics:\n")
        plot_confusion_matrix(avgmatrix)  # Visualizza la matrice di confusione media

    # Valutazione del modello utilizzando Stratified Shuffle Split (divisione stratificata)
    elif method == "stratified":
        # Chiede il numero di split per Stratified Shuffle Split
        n_splits = int(input("Inserisci il numero di split per Stratified Shuffle: ").strip())
        if n_splits < 2:
            print("\n ERRORE: Il numero di split deve essere almeno 2")
            return
        
        # Chiede la percentuale di dati da utilizzare per il test (fra 20% e 50%)
        test_size = float(input("Inserisci la percentuale di test (es. 0.2 per il 20%): ").strip())
        if test_size < 0.2 or test_size > 0.5:
            print("\n ERRORE: Scegli una percentuale tra 20% e 50%")
            return
        
        # Applica Stratified Shuffle Split
        splits = modelevaluator.stratified_shuffle_split(test_size, n_splits)
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        # Ciclo su ciascun split per allenare il modello e calcolare le metriche
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train_split = X[train_idx]
            X_test_split = X[test_idx]
            y_train_split = y[train_idx]
            y_test_split = y[test_idx]
            knn.train(X_train_split, y_train_split)
            y_pred = knn.predict(X_test_split)
            metrics = modelevaluator.compute_metrics(y_test_split, y_pred)
            true_positive += metrics['Confusion Matrix'][1, 1]
            true_negative += metrics['Confusion Matrix'][0, 0]
            false_positive += metrics['Confusion Matrix'][0, 1]
            false_negative += metrics['Confusion Matrix'][1, 0]
        
        # Calcola le metriche medie per tutti gli split
        avg_metrics = {
            'true_positives': true_positive / n_splits,
            'true_negatives': true_negative / n_splits,
            'false_positives': false_positive / n_splits,
            'false_negatives': false_negative / n_splits
        }

        # Crea una matrice di confusione media per gli split
        avgmatrix = [[int(avg_metrics['true_positives']), int(avg_metrics['false_positives'])], 
                     [int(avg_metrics['false_negatives']), int(avg_metrics['true_negatives'])]]
        print("\n Stratified Shuffle Split Average Metrics:\n")
        plot_confusion_matrix(avgmatrix)  # Visualizza la matrice di confusione media

    # Salva le metriche di valutazione in un file CSV
    df_metrics = pd.DataFrame([metrics])  # Crea un DataFrame con le metriche
    df_metrics.to_csv("evaluation_results.csv", index=False)  # Salva le metriche in un file CSV
    print("\n >>> RISULTATI SALVATI IN 'evaluation_results.csv'")  # Conferma che i risultati sono stati salvati
    
if __name__ == "__main__":
    main()  # Esegue la funzione principale