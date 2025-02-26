L'obiettivo di questo progetto consiste nel sviluppare un modello di apprendimento automatico e verificarne le prestazioni per classificare i tumori in base allecaratteristiche fornite.

Il file dataprocessing serve per caricare e ripulire un dataset affinché possa essere utilizzato con tecniche di validazione come Holdout, k-Fold Cross Validation e Stratified Cross Validation. La classe principale, DataProcessor, si occupa di tutto il processo di preprocessing attraverso il metodo process(), che segue diversi passaggi fondamentali.

Innanzitutto, il dataset viene caricato utilizzando una strategia specifica in base al formato del file, supportando CSV, JSON, TXT ed Excel. Una volta caricato, vengono gestiti i valori mancanti eliminando le righe in cui la colonna del target (classtype_v1) è assente, mentre i valori mancanti nelle feature vengono riempiti con la media della rispettiva colonna.

Successivamente, alcune colonne non necessarie ai fini del progetto vengono rimosse. Una volta ripuliti, i dati vengono suddivisi in due parti: le features e la colonna delle classi, che rappresenta le classi da prevedere. Le feature vengono poi normalizzate, scalando i valori tra 0 e 1, per garantire che abbiano lo stesso peso durante l’addestramento del modello.

Infine, il dataset processato viene salvato in due file separati, uno per le feature e uno per le classi, rendendolo pronto per essere utilizzato nei modelli di machine learning.