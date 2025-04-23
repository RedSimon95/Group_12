import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from load_strategy import LoadStrategy, CSVLoadStrategy, JSONLoadStrategy, TextLoadStrategy, ExcelLoadStrategy, SelectLoadStrategy
from dataprocessing import DataProcessor
from model import Classifier, KNNClassifier
from validation import ModelEvaluator

def main():
    
    file_path=input("\nInserire il path del dataset: ").strip()
    
    # Selezione della strategia di caricamento e caricamento dei dati
    # Inizializza DataProcessor che gestisce il caricamento dei dati e la loro separazione
    # in features e target.
    load_strategy, file_path = SelectLoadStrategy.select_load_strategy(file_path)
    processor = DataProcessor(file_path, load_strategy)
    features, target = processor.process()

    X = features.to_numpy()
    y = target.to_numpy().flatten()
    
    k = int(input("\nInserisci il numero di vicini (k): ").strip())
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

    # Inizializzo un dict analogo a quello restituito dal calcolo delle metriche per salvare le metriche medie
    avg_metrics={
        "Confusion Matrix": np.array([[0,0],[0,0]]),
        "Accuracy": float(0),
        "Error Rate": float(0),
        "Sensitivity (Recall)": float(0),
        "Specificity": float(0),
        "Geometric Mean": float(0),
        "AUC": float(0)
    }

    # Inizializzo la stringa avg_metrics_str per la visualizzazione e schermo di un riepilogo dell'esecuzione corrente
    avg_metrics_str="\n--- RIEPILPGO ESECUZIONE ---\n\nDATABASE PATH:\n"+file_path+"\n\nCLASSIFICATORE KNN:\nNumero vicini = "+str(k)+"\n"

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
            y_pred, y_prob = knn.predict(X_test)  # Predice sui dati di test
            avg_metrics = modelevaluator.compute_metrics(y_test, y_pred, y_prob[:,1])  # Calcola le metriche di valutazione
        
        # Aggiorno la stringa per la visulizzazione dei risultati
        avg_metrics_str=avg_metrics_str+"\nMETODO: HOLDOUT\nPercentuale di test = "+str(test_size)+"\n\nMETRICHE:\n"

    # Valutazione del modello utilizzando K-Fold Cross Validation
    elif method == "kfold":
        # Chiede il numero di fold per la validazione K-fold
        k_folds = int(input("Inserisci il numero di folds per K-Fold: ").strip())
        if k_folds < 2:
            print("\n ERRORE: Il numero di folds deve essere almeno 2")
            return
        
        # Applica la validazione K-fold
        folds = modelevaluator.k_fold_cross_validation(k_folds) # Divide i dati in folds

        # Ciclo su ciascun fold per allenare il modello e calcolare le metriche
        for i, fold in enumerate(folds):
            X_fold_train = X[fold]  # Dati di addestramento per il fold corrente
            y_fold_train = y[fold]  # Target di addestramento per il fold corrente
            knn.train(X_fold_train, y_fold_train)  # Allena il modello sui dati del fold
            y_fold_pred, y_fold_prob = knn.predict(X_fold_train)  # Predice sui dati del fold
            metrics = modelevaluator.compute_metrics(y_fold_train, y_fold_pred, y_fold_prob[:,1])  # Calcola le metriche di valutazione
            # Somma le metriche restituite dal fold corrente al relativo campo di avg_metrics
            for key in avg_metrics.keys():
                if(key=="Confusion Matrix"):
                    avg_metrics['Confusion Matrix'][1, 1] += metrics['Confusion Matrix'][1, 1]
                    avg_metrics['Confusion Matrix'][0, 0] += metrics['Confusion Matrix'][0, 0]
                    avg_metrics['Confusion Matrix'][0, 1] += metrics['Confusion Matrix'][0, 1]
                    avg_metrics['Confusion Matrix'][1, 0] += metrics['Confusion Matrix'][1, 0]
                else:
                    avg_metrics[key] += float(metrics[key])

        # Calcola il valore medio per ogni campo di avg_metrics; i dati sono arrotondati al quarto decimale
        for key in avg_metrics.keys():
            if(key=="Confusion Matrix"):
                avg_metrics['Confusion Matrix'][1, 1] = int(avg_metrics['Confusion Matrix'][1, 1]/k_folds)
                avg_metrics['Confusion Matrix'][0, 0] = int(avg_metrics['Confusion Matrix'][0, 0]/k_folds)
                avg_metrics['Confusion Matrix'][0, 1] = int(avg_metrics['Confusion Matrix'][0, 1]/k_folds)
                avg_metrics['Confusion Matrix'][1, 0] = int(avg_metrics['Confusion Matrix'][1, 0]/k_folds)
            else:
                avg_metrics[key] = np.round(avg_metrics[key]/k_folds,4)

        # Aggiorno la stringa per la visulizzazione dei risultati
        avg_metrics_str=avg_metrics_str+"\nMETODO: K-FOLD\nNumero folds = "+str(k_folds)+"\n\nMETRICHE:\n"

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
            print("\n ERRORE: Scegli una percentuale tra 20 e 50%")
            return
        
        # Applica Stratified Shuffle Split
        splits = modelevaluator.stratified_shuffle_split(test_size, n_splits)
        
        
        # Ciclo su ciascun split per allenare il modello e calcolare le metriche
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train_split = X[train_idx]
            X_test_split = X[test_idx]
            y_train_split = y[train_idx]
            y_test_split = y[test_idx]
            knn.train(X_train_split, y_train_split) # Addestro il knn con i dati di train dello split corrente
            y_pred, y_prob = knn.predict(X_test_split) # Calcola le predizioni e le probabilità con i dati di test per lo split corrente
            metrics = modelevaluator.compute_metrics(y_test_split, y_pred, y_prob[:,1]) # Calcola le metriche dello split corrente
            # Somma le metriche restituite dello split corrente al relativo campo di avg_metrics
            for key in avg_metrics.keys():
                if(key=="Confusion Matrix"):
                    avg_metrics['Confusion Matrix'][1, 1] += metrics['Confusion Matrix'][1, 1]
                    avg_metrics['Confusion Matrix'][0, 0] += metrics['Confusion Matrix'][0, 0]
                    avg_metrics['Confusion Matrix'][0, 1] += metrics['Confusion Matrix'][0, 1]
                    avg_metrics['Confusion Matrix'][1, 0] += metrics['Confusion Matrix'][1, 0]
                else:
                    avg_metrics[key] += float(metrics[key])

        # Calcola il valore medio per ogni campo di avg_metrics; i dati sono arrotondati al quarto decimale
        for key in avg_metrics.keys():
            if(key=="Confusion Matrix"):
                avg_metrics['Confusion Matrix'][1, 1] = int(avg_metrics['Confusion Matrix'][1, 1]/n_splits)
                avg_metrics['Confusion Matrix'][0, 0] = int(avg_metrics['Confusion Matrix'][0, 0]/n_splits)
                avg_metrics['Confusion Matrix'][0, 1] = int(avg_metrics['Confusion Matrix'][0, 1]/n_splits)
                avg_metrics['Confusion Matrix'][1, 0] = int(avg_metrics['Confusion Matrix'][1, 0]/n_splits)
            else:
                avg_metrics[key] = np.round(avg_metrics[key]/n_splits,4)
        
        # Aggiorno la stringa per la visulizzazione dei risultati
        avg_metrics_str=avg_metrics_str+"\nMETODO: STRATIFIED SHUFFLE SPLIT\nNumero splits = "+str(n_splits)+", percentuale di test = "+str(test_size)+"\n\nMETRICHE:\n"

    plot_confusion_matrix(avg_metrics["Confusion Matrix"])
    # Salva le metriche di valutazione in un file CSV
    for key in avg_metrics.keys():
        if(key=="Confusion Matrix"):
            avg_metrics_str=avg_metrics_str+key+": "+str("\n\t"+str(avg_metrics[key][0])+"\n\t"+str(avg_metrics[key][1])+"\n")
        else:
            avg_metrics_str=avg_metrics_str+key+": "+str(avg_metrics[key])+"\n"
    
    print(avg_metrics_str) # Stampa a schermo il riepilogo dei risultati
    # Salva il riepilogo in un file txt
    with open("evaluation_results.txt", "w") as results_file:
        # Write the string to the file
        results_file.write(avg_metrics_str)
    print(" >>> RISULTATI SALVATI IN 'evaluation_results.txt'")  # Conferma che i risultati sono stati salvati
    
if __name__ == "__main__":
    main()  # Esegue la funzione principale
