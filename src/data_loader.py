from sklearn.datasets import load_iris
import pandas as pd
import os

# Definiamo il nome del file CSV di output
OUTPUT_FILE = 'iris_campionamento_casuale.csv'

# 1. Caricamento e preparazione del Dataset (necessario per definire la funzione)
iris = load_iris()
DF_IRIS = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# --- Funzione per l'inizializzazione del CSV (Devi chiamarla una volta prima dell'uso) ---
def initialize_csv_file(dataframe, filename):
    """
    Prepara il file CSV rimuovendo i vecchi dati e scrivendo l'intestazione.
    
    :param dataframe: Il DataFrame originale (usato solo per le intestazioni).
    :param filename: Il percorso del file CSV di output.
    """
    if os.path.exists(filename):
        os.remove(filename)

    # Creiamo un DataFrame vuoto con le stesse colonne per gestire l'intestazione
    empty_df = pd.DataFrame(columns=dataframe.columns)
    empty_df.to_csv(filename, index=False)
    print(f"Inizializzato file CSV: '{filename}' con l'intestazione. Il file è pronto per l'append.")
    print("Ricorda di chiamare questa funzione una volta sola.")


# --- Funzione Generatrice Principale ---
def random_data_sampler_generator(dataframe, filename):
    """
    Generatore che, ad ogni chiamata next(), campiona una riga casuale dal 
    DataFrame (con ripetizione), la salva sul file CSV specificato, e la restituisce 
    come DataFrame di una riga.
    
    ATTENZIONE: Questo è un generatore infinito (while True).
    
    :param dataframe: Il DataFrame da cui campionare i dati.
    :param filename: Il percorso del file CSV su cui salvare in append.
    :return: Un DataFrame di una riga (quella campionata e salvata).
    """
    # Usiamo un ciclo infinito per il campionamento casuale con ripetizione
    while True:
        # 1. Campionamento Casuale di una riga
        # dataframe.sample(n=1) restituisce un DataFrame di una riga.
        data_point_df = dataframe.sample(n=1)
        
        # 2. Salvataggio della riga sul CSV in modalità append ('a')
        # header=False previene la riscrittura dell'intestazione ad ogni riga.
        data_point_df.to_csv(filename, mode='a', header=False, index=False)
        
        print(f"Riga salvata su '{filename}'.")
        
        # 3. Restituisce il DataFrame di una riga campionata
        yield data_point_df

