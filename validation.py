import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from load_strategy import CSVLoadStrategy
from dataprocessing import DataProcessor  # Import per il preprocessing
from model import KNNClassifier # Import per il modello KNN
import matplotlib.pyplot as plt

class ModelEvaluator:
    """
    Classe per la valutazione del modello KNN utilizzando Holdout, K-Fold Cross Validation e Stratified Shuffle Split.
    """
    def __init__(self, model, X, y):
        """
        Inizializza l'evaluator con il modello e i dati.
        :param model: Modello KNN addestrabile.
        :param X: Features.
        :param y: Target.
        """
        # Inizializza le variabili per il modello, le features, il target e per il logging
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y).flatten() 
        self.logs = []  # Struttura per tenere traccia dei risultati
        self.global_conf_matrix = np.zeros((2, 2), dtype=int)  # Matrice di confusione globale

    def compute_confusion_matrix(self, y_true, y_pred):
        """Calcola la matrice di confusione."""
        # Calcolo dei veri positivi (TP): casi in cui il valore reale e la predizione sono entrambi 4
        tp = np.sum((y_true == 4) & (y_pred == 4))
        # Calcolo dei veri negativi (TN): casi in cui il valore reale e la predizione sono entrambi 2
        tn = np.sum((y_true == 2) & (y_pred == 2))
        # Calcolo dei falsi positivi (FP): casi in cui il valore reale è 2 ma la predizione è 4
        fp = np.sum((y_true == 2) & (y_pred == 4))
        # Calcolo dei falsi negativi (FN): casi in cui il valore reale è 4 ma la predizione è 2
        fn = np.sum((y_true == 4) & (y_pred == 2))
        
        # Restituisce la matrice di confusione
        return np.array([[tn, fp],[fn, tp]])

    def compute_metrics(self, y_true, y_pred, y_prob):

        # Per prevenire errori, trasforma gli ingressi in array numpy e arrotonda i valori di probabilità al sesto decimale 
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        y_prob_rounded=np.array([round(num, 6) for num in y_prob])

        """Calcola le metriche di valutazione."""
        # Calcola la matrice di confusione
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.ravel()
        
        # Accuratezza: proporzione di previsioni corrette rispetto al totale dei campioni
        accuracy = (tp + tn) / len(y_true)
        # Tasso di errore: complemento dell'accuratezza
        error_rate = 1 - accuracy
        # Sensibilità (Recall o True Positive Rate): capacità del modello di identificare correttamente i positivi
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificità (True Negative Rate): capacità del modello di identificare correttamente i negativi
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Media geometrica della sensibilità e specificità, utile per dataset sbilanciati
        geometric_mean = np.sqrt(sensitivity * specificity)

        """Calcola il valore AUC"""

        # Divide l'intervallo [0,1] in 100 parti, arrotonda i valori ottenuti al sesto decimale e ordina i valori in senso decrescente (ai fini dell'utilizzo di trapezoid)
        thresholds=np.linspace(0, 1, 100)
        thresholds_rounded=np.sort([round(num, 6) for num in thresholds])[::-1]
        # Inizializza le liste di True Positive Rate (tpr_values) e False Positive Rate (fpr_values)
        tpr_values = []
        fpr_values = []

        for threshold in thresholds_rounded:

            # Calcola la classe di apparteneza sulla base del threshold corrente (la classe è 0 se negativo, 1 se positivo)
            predicted_lables_zo = (y_prob_rounded >= threshold)
            # Converte le classi 0 e 1 in 2 e 4
            predicted_labels=np.array(2*pow(2,predicted_lables_zo.astype(int)))
            # Calcola la confusion matrix per il threshold corrente per ottenerne i valori al suo interno
            tn, fp, fn, tp = self.compute_confusion_matrix(y_true, predicted_labels).ravel()
            
            # Calculate TPR and FPR
            if tp==0:
                tpr=0
            else:
                tpr=tp/(tp+fn)
            if fp==0:
                fpr=0
            else:
                fpr=fp/(fp+tn)
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        """
        # Grfica la ROC curve
        plt.plot(fpr_values, tpr_values, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        """
        # Calcola l'area sottesa alla ROC Curve per ottenere l'AUC
        auc=np.trapezoid(tpr_values, fpr_values)

        # Restituisce un dizionario con tutte le metriche calcolate
        return {
            "Confusion Matrix": confusion_matrix,
            "Accuracy": f"{accuracy:.4f}",
            "Error Rate": f"{error_rate:.4f}",
            "Sensitivity (Recall)": f"{sensitivity:.4f}",
            "Specificity": f"{specificity:.4f}",
            "Geometric Mean": f"{geometric_mean:.4f}",
            "AUC": f"{auc:.4f}"
        }

    def holdout(self, test_size=0.2):
        """
        Suddivide il dataset in training e test set secondo il metodo Holdout.
        :param test_size: Percentuale di dati da destinare al test set (default = 0.2).
        """
        # Mescola gli indici dei dati per garantire la casualità
        indices = np.random.permutation(len(self.X))  
        
        # Calcola la posizione dove i dati saranno divisi in training e test set
        split = int(len(self.X) * (1-test_size)) 
        
        # Suddivide gli indici dei dati in due gruppi: uno per il training e uno per il test
        train_indices, test_indices = indices[:split], indices[split:]  
        return self.X[train_indices], self.X[test_indices], self.y[train_indices], self.y[test_indices]

    def k_fold_cross_validation(self, k=5):
        """
        Esegue la validazione incrociata K-Fold.
        :param k: Numero di fold per la cross-validation (default = 5).
        """
        # Mescola gli indici dei dati per garantire la casualità prima di suddividere in fold
        indices = np.random.permutation(len(self.X))  
        
        # Suddivide gli indici in k parti (fold)
        folds = np.array_split(indices, k)  
        return folds  # Restituisce gli ultimi risultati

    def stratified_shuffle_split(self, test_size=0.2, n_splits=5):
        """
        Implementazione manuale di Stratified Shuffle Split usando solo numpy e pandas.
        """
        # Calcola la distribuzione delle classi nel dataset
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        splits = []
        
        for _ in range(n_splits):
            # Per ogni split, prepariamo gli indici per il training e il test set
            train_indices = []
            test_indices = []
            
            # Per ogni classe, mescoliamo i campioni e li dividiamo tra training e test set
            for cls in unique_classes:
                cls_indices = np.where(self.y == cls)[0]  # Ottieni gli indici di una specifica classe
                np.random.shuffle(cls_indices)  # Mescola gli indici di quella classe
                
                # Calcoliamo quanti campioni di questa classe devono andare nel test set
                n_test = int(len(cls_indices) * test_size)  
                
                # Selezioniamo i primi indici per il test set e gli altri per il training set
                test_indices.extend(cls_indices[:n_test])  
                train_indices.extend(cls_indices[n_test:])  
            
            # Salviamo le suddivisioni (training e test) per ciascun split
            splits.append((np.array(train_indices), np.array(test_indices)))
        
        return splits  # Restituisce gli ultimi risultati

def validation_test():
    # Caricamento e preprocessing del dataset
    # Viene caricato il dataset tramite la classe DataProcessor che gestisce il caricamento e la preparazione dei dati

    output_features = "processed_features.csv"
    output_target = "processed_target.csv"

    data_processor = DataProcessor("dataset.csv", CSVLoadStrategy(), output_features, output_target)
    features, target = data_processor.process()  # Estrae le features e il target dal dataset

    # Creazione del modello KNN
    # Utilizza il factory pattern per creare un classificatore KNN con k=3
    knn_model = KNNClassifier(3)

    # Valutazione del modello
    # Viene inizializzata la classe ModelEvaluator per eseguire la valutazione del modello con i vari metodi
    model_evaluator = ModelEvaluator(knn_model, features, target)
    model_evaluator.holdout()  # Esegui la valutazione tramite il metodo Holdout
    model_evaluator.k_fold_cross_validation()  # Esegui la valutazione tramite K-Fold Cross Validation
    model_evaluator.stratified_shuffle_split()  # Esegui la valutazione tramite Stratified Shuffle Split

def compute_metrics_test():
    # Crea un modello knn per inizializzare im modo generico model_evaluator
    knn_model = KNNClassifier(3)
    model_evaluator = ModelEvaluator(knn_model, [0,0,0,0], [0,0,0,0])

    # Parametri d'esempio per compute_metrics (continuo dell'esempio di model.py)
    y_true=[2,4,4]
    y_pred=[2,4,4]
    y_prob=[0.3333333333333333, 0.6666666666666666, 1.0] # Probabilità che sia True (appartenenza alla classe 4)
    
    # Calcolo delle metriche
    metrics=model_evaluator.compute_metrics(y_true,y_pred,y_prob)

    # Risultati attesi
    metriche_attese={
        "Confusion Matrix": "[[1 0],[0 2]]",
        "Accuracy": "1.0000",
        "Error Rate": "0.0000",
        "Sensitivity (Recall)": "1.0000",
        "Specificity": "1.0000",
        "Geometric Mean": "1.0000",
        "AUC": "1.0000"
    }

    # Confronto con i valori attesi e stampa a video

    print("\n---RISULTATI DEI TEST SULLE METRICHE---\n")

    total_keys=0 # Conta le metriche totali
    correct_keys=0 # Conta le metriche corrette

    # Esegue il controllo solamente se le chiavi di metrics e metriche_attese corrispondono
    if(metrics.keys()==metriche_attese.keys()): 

        # Esegue il confronto per ogni chiave di metriche_attese
        for key in metriche_attese.keys(): 
            total_keys+=1

            #Converte la Confusion Matrix in uns tringa per agevolare il confronto
            if(key=="Confusion Matrix"):
                metrics[key]=str("["+str(metrics[key][0])+","+str(metrics[key][1])+"]")

            # Se le metriche generate per la chiave key sono pari a quelle attese, stampa un messaggio positivo, altrimenti di errore
            if metrics[key]==metriche_attese[key]:
                print(key,": CORRETTA")
                correct_keys+=1
            else:
                print(key,": ERRORE")

            #In ogni caso stampa le metriche generate e attese ai fini del debugging
            print(f" > Risultato generato: {metrics[key]}")
            print(f" > Risultato atteso:   {metriche_attese[key]}")

        # Al termine stampa una riga di resoconto con il numero dei test eseguiti con successo rispetto al totale
        print(f"\nRESOCONTO: {correct_keys} su {total_keys} test eseguiti con successo\n")  

    else:
        print("ERRORE: Le chiavi del dizionario generato e di quello di test non corrispondono!\n")
        
    
# ------ BLOCCO DI TEST ------
if __name__=='__main__':
    validation_test()
    compute_metrics_test()
