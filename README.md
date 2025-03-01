L'obiettivo di questo progetto consiste nel sviluppare un modello di apprendimento automatico e verificarne le prestazioni per classificare i tumori in base allecaratteristiche fornite.

Il file dataprocessing serve per caricare e ripulire un dataset affinché possa essere utilizzato con tecniche di validazione come Holdout, k-Fold Cross Validation e Stratified Cross Validation. La classe principale, DataProcessor, si occupa di tutto il processo di preprocessing attraverso il metodo process(), che segue diversi passaggi fondamentali.

Innanzitutto, il dataset viene caricato utilizzando una strategia specifica in base al formato del file, supportando CSV, JSON, TXT ed Excel. Una volta caricato, vengono gestiti i valori mancanti eliminando le righe in cui la colonna del target (classtype_v1) è assente, mentre i valori mancanti nelle feature vengono riempiti con la media della rispettiva colonna.

Successivamente, alcune colonne non necessarie ai fini del progetto vengono rimosse. Una volta ripuliti, i dati vengono suddivisi in due parti: le features e la colonna delle classi, che rappresenta le classi da prevedere. Le feature vengono poi normalizzate, scalando i valori tra 0 e 1, per garantire che abbiano lo stesso peso durante l’addestramento del modello.

Infine, il dataset processato viene salvato in due file separati, uno per le feature e uno per le classi, rendendolo pronto per essere utilizzato nei modelli di machine learning.

Il file model.py implementa il metodo del k-NN, un algoritmo basato sulla distanza euclidea tra oggetti, i quali tendono a trovarsi vicini nello spazio delle feature se presentano caratteristiche simili.

L'algoritmo calcola la distanza tra il punto da classificare e tutti i punti noti, seleziona i k più vicini e assegna la classe più frequente tra essi. Questo codice implementa il k-NN utilizzando un approccio modulare.

La classe astratta Classifier definisce un'interfaccia comune, mentre la classe KNNClassifier fornisce una specifica implementazione dell'algoritmo. Il metodo train() memorizza il dataset di addestramento, mentre predict() calcola le distanze tra i punti di test e quelli di training, selezionando i più vicini per determinare la classe.

Per garantire il corretto funzionamento, è presente una funzione di test test_knn_classifier(), che confronta le predizioni del modello con i valori attesi. Se le previsioni sono corrette, il test viene superato.

validation.py implementa e testa diverse strategie di validazione dei modelli di machine learning, come la validazione Holdout, K-Fold e Stratified Shuffle Split.

Ogni strategia ha un comportamento diverso: la validazione Holdout separa i dati in due set, uno per l'addestramento e l'altro per il test, con un mescolamento casuale; la validazione K-Fold suddivide il dataset in k fold, eseguendo una validazione incrociata; mentre la validazione Stratified Shuffle Split preserva la distribuzione delle classi tra i due set.

Un'apposita factory consente di scegliere facilmente la strategia desiderata, e la funzione di test verifica che ogni strategia funzioni correttamente su un piccolo esempio di dati.
