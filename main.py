import pandas as pd
import numpy as np
import sys
from dataprocessing import DataProcessor
from knn_classifier import KNNClassifier

def main():
    # Input da console
    try:
        k = int(input("Inserisci il numero di vicini (k): ").strip())
        method = input("Scegli il metodo di valutazione (holdout, kfold, stratified): ").strip().lower()
    except ValueError:
        print("Errore: Inserisci valori validi.")
        sys.exit(1)

    # Percorsi dei file
    file_path = "version_1.csv"
    output_features = "processed_features.csv"
    output_target = "processed_target.csv"

    # Preprocessing
    processor = DataProcessor(file_path, output_features, output_target)
    features, target = processor.process()

    # Convertire in NumPy
    X = features.to_numpy()
    y = target.to_numpy().flatten()

    # Inizializzazione del modello KNN
    knn = KNNClassifier(k)