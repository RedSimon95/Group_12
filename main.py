import pandas as pd
import numpy as np
import sys
from dataprocessing import CSVLoadStrategy, DataProcessor, JSONLoadStrategy, TextLoadStrategy, ExcelLoadStrategy
from model import KNNClassifier
from validation import ValidationFactory
import os


print("\n Inserire nella directory un file con una qualsiasi delle estenisone supportate: CSV, JSON, TXT, XLSX")
def errorMessage(point):

    # Stampa un messaggio di errore:
    #   Per l'inserimento di k quando point==1
    #   Per l'inserimento del metodo quando point==2
    #   Per l'inserimento della percentuale nell'holdout quando point==3,4
    # Input:
    #       point : [int] in base al suo valore viene stampato un messaggio di errore diverso, come specificato sopra

    print("\n+++++++++++++++ ERRORE +++++++++++++++++++")
    if point==1:
        print("++ Inserire un numero intero positivo.  ++")
    elif point==2:
        print("++ Scegliere uno dei seguenti metodi:   ++")
        print("++ holdout, kfold, stratified           ++")
    elif point==3 or point==4:
        if point==4:
            print("++ Non è possibile scegliere il 100%    ++")
        print("++ Scegliere una percentuale valida co- ++")
        print("++ me specificato (es. 0.3 per 30%)     ++")
    print("++ Altrimenti PER USCIRE DIGITA \"0\".    ++")
    print("++++++++++++++++++++++++++++++++++++++++++\n")



def welcomeMessage(contatoreEsecuzioni):

    # Stampa un messaggio di inizio esecuzione
    # Input:
    #       contatoreEsecuzioni : [int] numero dell'esecuzione corrente

    print("\n++++++++++++++++++++++++++++++++++++++++++")
    if contatoreEsecuzioni<10:
        print("++++++++++++++ ESECUZIONE "+str(contatoreEsecuzioni)+" ++++++++++++++")
    else:
        print("+++++++++++++ ESECUZIONE "+str(contatoreEsecuzioni)+" ++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++\n")



def exitMessage():

    # Stampa un messaggio di termine esecuzione

    print("\n++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++ TERMINE ESECUZIONE +++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++\n")



def verificaInteriPositivi(k):
    
    # Funzione per verificare che il valore in ingresso k sia positivo.
    # Input:
    #       k : [int]
    # Output:
    #       0 : se il valore è negativo
    #       1 : se il valore è positivo
    #       2 : se il valore è 0
    # I valori sono utili all'aggiornamneto della variabile inputError che gestisce l'inserimento dei valori nel main

    if k<0:
        errorMessage(1)
        return 0 #Input non valido, nuovo inserimento
    elif k==0 :
        exitMessage()
        return 2 #L'esecuzione termina
    else:
        return 1 #Input valido, passo successivo



def verificaPercentualiAnomale(percentuale):

    # Stampa un messaggio di conferma in caso di percentuali inusuali (inferiori al 20 e superiori al 30)
    # Input:
    #       percentuale : [float] percentuale fornita in input in valore tra 0 e 1 (es. 20% sarà 0.2)
    # Output:
    #       0 : nel caso in cui si scelga di effettuare un nuovo inserimento
    #       1 : nel caso in cui si confermi la scelta fatta
    # I valori sono utili all'aggiornamneto della variabile inputError che gestisce l'inserimento dei valori nel main

    if percentuale<0.2 or percentuale>0.3:
        verifica=input("Si consigliano valori di percentuale compresi tra il 20 e il 30. Continuare comunque? si/no > ").strip().lower()
        if verifica!="si":
            return 0 #Input non valido, nuovo inserimento
    return 1 #Input valido, passo successivo



def verificaPercentuali(k):

    # Verifica che le percentuali fornite in input siano ben scritte gestendo gli errori:
    #   Se una percentuale assume valori inusuali (inferiori al 20 e superiori al 30) esegue verificaPercentualiAnomale per confermare o meno la scelta
    #   Se una percentuale non viene espressa come valore tra 0 e 1 ma nel dormato 0%-100% chiede conferma
    # Input:
    #       percentuale : [int] percentuale fornita in input
    # Output:
    #       0 : percentuale non valida
    #       1 : percentuale valida
    #       2 : se il valore è 0
    # I valori sono utili all'aggiornamneto della variabile inputError che gestisce l'inserimento dei valori nel main

    if k<0 or k>=100:
        errorMessage(3)
        return 0 #Input non valido, nuovo inserimento
    elif k==0:
        exitMessage()
        return 2 #L'esecuzione termina
    elif k==1:
        errorMessage(4)
        return 0 #Input non valido, nuovo inserimento
    elif k>1:
        verifica=input("Forse intendevi "+str(int(k))+"% ? si/no > ").strip().lower()
        if verifica=="si":
            return verificaPercentualiAnomale(k/100)
        else:
            errorMessage(3)
            return 0 #Input non valido, nuovo inserimento
    else:
        return verificaPercentualiAnomale(k)



def main():

    # Esegue un flusso completo: preprocessing, addestramento e valutazione del modello.
    
    # INPUT DA CONSOLE

    # Richiede un input per k finchè non viene fornito un valore valido (numero intero positivo).
    # Il processo termina se viene inserito 0 (zero).

    # La variabile inputError assume i seguenti valori
    #       0 : (valore iniziale) finchè non viene fornito un input valido
    #       1 : non appena viene fornito un input valido, permette il passaggio passo successivo
    #       2 : condizione di uscita dall'esecuzione corrente

    inputError=0
    while inputError==0:
        try:
            k = int(input("Inserisci il numero di vicini (k) o digita 0 per uscire: ").strip())
            inputError=verificaInteriPositivi(k)
        except ValueError:
            errorMessage(1)

    if inputError!=2:
        inputError=0
        while inputError==0:
            method = input("Scegli il metodo di valutazione (holdout, kfold, stratified) o digita 0 per uscire: ").strip().lower()
            if method == "holdout":
                while inputError==0:
                    try:
                        test_size = float(input("Inserisci la percentuale di test (es. 0.2 per il 20%) o digita 0 per uscire: ").strip())
                        inputError=verificaPercentuali(test_size)
                        if (inputError==1) and (test_size>1):
                            test_size=test_size/100
                    except ValueError:
                        errorMessage(3)
                k_folds, n_splits = None, None
            elif method == "kfold":
                while inputError==0:
                    try:
                        k_folds = int(input("Inserisci il numero di folds per K-Fold o digita 0 per uscire: ").strip())
                        inputError=verificaInteriPositivi(k_folds)
                    except ValueError:
                        errorMessage(1)
                test_size, n_splits = None, None
            elif method == "stratified":
                while inputError==0:
                    try:
                        n_splits = int(input("Inserisci il numero di split per Stratified Shuffle o digita 0 per uscire: ").strip())
                        inputError=verificaInteriPositivi(n_splits)
                    except ValueError:
                        errorMessage(1)
                if inputError!=2:
                    inputError=0
                    while inputError==0:
                        try:
                            test_size = float(input("Inserisci la percentuale di test (es. 0.2 per il 20%) o digita 0 per uscire: ").strip())
                            inputError=verificaPercentuali(test_size)
                            if (inputError==1) and (test_size>1):
                                test_size=test_size/100
                        except ValueError:
                            errorMessage(3)
                    k_folds = None
            elif method == str(0):
                exitMessage()
                inputError=2
            else:
                errorMessage(2)
    
    if inputError!=2:

        def select_load_strategy():
            for file in os.listdir('.'):
                if file.endswith('.csv'):
                    return CSVLoadStrategy(), file
                elif file.endswith('.json'):
                    return JSONLoadStrategy(), file
                elif file.endswith('.txt'):
                    return TextLoadStrategy(), file
                elif file.endswith('.xlsx'):
                    return ExcelLoadStrategy(), file
            raise FileNotFoundError("Nessun file di dataset trovato (CSV, JSON, TXT, XLSX).")
        load_strategy = CSVLoadStrategy()
        output_features = "processed_features.csv"
        output_target = "processed_target.csv"

        # Preprocessing
        load_strategy, file_path = select_load_strategy()
        processor = DataProcessor( file_path, load_strategy, output_features, output_target)
    
        features, target = processor.process()

        # Convertire in NumPy
        X = features.to_numpy()
        y = target.to_numpy().flatten()

        # Inizializzazione del modello KNN
        knn = KNNClassifier(k)

        # Esegui il metodo di valutazione scelto dall'utente
        if method == "holdout":
            holdout = ValidationFactory.get_strategy("Holdout", 0.4)
            X_train, X_test, y_train, y_test = holdout.split_data(X, y)
            print(f" \n Holdout - Training set: \n {X_train.tolist()}, \n Test set: \n {X_test.tolist()}, \n  Training labels: \n {y_train.tolist()}, \n  Test labels: \n  {y_test.tolist()}")

            knn.train(X_train, y_train)
            metrics = holdout.compute_metrics(y_test, knn.predict(X_test))
            print("\n >>> METRICHE:")
            for metric, value in metrics.items():
                print(f"{metric}: \n {value}")

        elif method == "kfold":
            kfold = ValidationFactory.get_strategy("K-Fold", 5)
            folds = kfold.split_data(X, y)

            print("\n Suddivisione K-Fold dei dati:")
            for i, fold in enumerate(folds):
                X_fold_train = X[fold]
                y_fold_train = y[fold]
                print(f"\n Fold {i+1}: \n Training set: \n {X_fold_train.tolist()}, \n Training labels: \n {y_fold_train.tolist()}")
                knn.train(X_fold_train, y_fold_train)
                metrics = kfold.compute_metrics(y_fold_train, knn.predict(X_fold_train))
                print("\n >>> METRICHE:")
                for metric, value in metrics.items():
                    print(f"{metric}: \n {value}")
    
        elif method == "stratified":
            stratified = ValidationFactory.get_strategy("Stratified Shuffle", 0.4)
            splits = stratified.split_data(X, y)
            for i, (train_idx, test_idx) in enumerate(splits):
                X_train_split = X[train_idx]
                X_test_split = X[test_idx]
                y_train_split = y[train_idx]
                y_test_split = y[test_idx]
                print(f"\n Stratified Shuffle {i+1}: \n Training set: \n {X_train_split.tolist()}, \n Test set: \n {X_test_split.tolist()}, \n Training labels:\n  {y_train_split.tolist()}, \n Test labels:\n  {y_test_split.tolist()}")
                knn.train(X_train_split, y_train_split)
                metrics = stratified.compute_metrics(y_test_split, knn.predict(X_test_split))
                print("\n >>> METRICHE:")
                for metric, value in metrics.items():
                    print(f"{metric}: \n {value}")

        # Salva i risultati in un file Excel
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv("evaluation_results.csv", index=False)
        print("\n >>> RISULTATI SALVATI IN 'evaluation_results.csv'")
    
if __name__ == "__main__":
    print("\n++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++ ADDESTRAMENTO AI TUMORI +++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++\n")
    contatoreEsecuzioni=1
    continuare="si"
    while continuare=="si":
        
        welcomeMessage(contatoreEsecuzioni)

        main()

        print("\n++++++++++++++++++++++++++++++++++++++++++")
        continuare=input("\n ESECUZIONE COMPLETATA: vuoi iniziare una nuova analisi? si/no > ")

        while continuare!="si" and continuare!="no":
            continuare=input("\n Valore non valido. Scegliere tra si/no > ")

        if continuare=="no":
            exitMessage()

        contatoreEsecuzioni=contatoreEsecuzioni+1