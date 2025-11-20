from sklearn.datasets import load_iris
import pandas as pd
import os # Importiamo os per la gestione dei file

# 1. Caricamento e preparazione del Dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Definiamo il nome del file CSV di output
OUTPUT_FILE = 'iris_progressivo.csv'

# 2. La Funzione Generatrice Modificata (attende Invio o 'stop')
def csv_data_generator(dataframe):
    """
    Restituisce una riga alla volta, sospendendo l'esecuzione
    finché l'utente non preme Invio o digita 'stop'.
    """
    for index, row in dataframe.iterrows():
        # L'input senza prompt attende una riga di testo (Invio/Enter)
        command = input(f"\n--- Pronti per la riga {index} --- Premi INVIO per aggiungere al CSV o digita 'stop' per terminare: ")
        
        if command.lower() == 'stop':
            print("Generazione dati interrotta da comando utente.")
            break
        elif command == '': # Invio (stringa vuota)
            # Restituisce la riga come una Series Pandas
            yield row
        else:
            print(f"Comando non valido ('{command}'). Premi INVIO o digita 'stop'.")
            # In questo caso, il ciclo for avanza alla riga successiva senza
            # restituire il dato corrente.

# 3. Logica Principale di Salvataggio Continua
def save_data_to_csv_until_stop(dataframe):
    
    # 3a. Inizializzazione: Creiamo un file CSV vuoto
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE) # Rimuovi il vecchio file se esiste

    # Creiamo un DataFrame vuoto con le stesse colonne per gestire l'intestazione
    empty_df = pd.DataFrame(columns=dataframe.columns)
    empty_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Creato file CSV iniziale vuoto: '{OUTPUT_FILE}' con l'intestazione.")
    
    # Creiamo l'oggetto generatore
    iris_generator = csv_data_generator(dataframe)
    total_rows_saved = 0
    
    # Ciclo infinito per continuare a chiedere dati
    while True:
        try:
            # Tenta di ottenere il prossimo dato dal generatore
            data_point_series = next(iris_generator)
            
            # Trasformiamo la Series (il dato) in un DataFrame con una singola riga
            data_point_df = data_point_series.to_frame().T 

            # Aggiungiamo la riga al CSV in modalità append ('a')
            # L'header=False assicura che l'intestazione non venga riscritta
            data_point_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            total_rows_saved += 1
            print(f"Aggiunta riga al CSV. Totale righe salvate: {total_rows_saved}.")
            
        except StopIteration:
            # Eccezione sollevata quando il generatore finisce (fine del DataFrame)
            print(f"\nDataFrame esaurito. Nessun altro dato da salvare. File finale: '{OUTPUT_FILE}'.")
            break 
        except:
            # Cattura l'interruzione se l'utente ha digitato 'stop'
            print(f"\nOperation interrotta. File finale: '{OUTPUT_FILE}' con {total_rows_saved} righe.")
            break 

# Eseguiamo la funzione
save_data_to_csv_until_stop(df)