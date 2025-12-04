"""
Configurazione per Vector Search.

Questo modulo gestisce la configurazione dell'applicazione usando variabili d'ambiente.
Le variabili vengono caricate da un file .env esterno usando la libreria python-dotenv.

CONCETTI CHIAVE:
----------------
1. VARIABILI D'AMBIENTE: Sono valori configurabili che vengono letti dal sistema
   operativo o da file .env. Permettono di separare la configurazione dal codice,
   evitando di hardcodare valori sensibili come API key.

2. FILE .ENV: Un file di testo con formato KEY=value che contiene le configurazioni.
   Non va mai committato in git perché può contenere segreti (API key, password).

3. PATTERN SINGLETON: La variabile `config` alla fine del modulo è un'istanza
   globale della classe Config, accessibile da tutto il codice.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# =============================================================================
# CARICAMENTO CONFIGURAZIONE
# =============================================================================
# load_dotenv() cerca un file .env e carica le variabili in os.environ.
# Il file .env deve trovarsi nella root del progetto (test-embeddings/.env).
#
# Path(__file__) restituisce il path di questo file (config.py)
# .parent sale di una directory alla volta:
#   .parent -> vectorsearch/
#   .parent.parent -> src/
#   .parent.parent.parent -> test-embeddings/ (root del progetto)
# =============================================================================
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    """
    Classe di configurazione dell'applicazione.

    Tutti i valori sono attributi di classe (non di istanza) per semplicità.
    Vengono letti dalle variabili d'ambiente con os.getenv(), che restituisce
    il valore della variabile o un default se non esiste.

    Attributes:
        OPENAI_API_KEY: Chiave API per OpenAI. Necessaria per generare embeddings.
            Viene usata per autenticarsi con i servizi OpenAI.

        EMBEDDING_MODEL: Nome del modello di embedding da usare.
            - text-embedding-3-small: Modello compatto (1536 dimensioni), economico
            - text-embedding-3-large: Modello potente (3072 dimensioni), più costoso
            - text-embedding-ada-002: Modello legacy (1536 dimensioni)

        CHROMA_PERSIST_DIR: Directory dove Chroma salva i dati del vector store.
            I dati persistono tra le esecuzioni del programma.

        CHUNK_SIZE: Dimensione massima in caratteri di ogni chunk di testo.
            Documenti più lunghi vengono divisi in pezzi di questa dimensione.
            Valori tipici: 500-2000 caratteri.

        CHUNK_OVERLAP: Sovrapposizione tra chunk consecutivi in caratteri.
            Serve a non "tagliare" concetti a metà tra due chunk.
            Tipicamente 10-20% di CHUNK_SIZE.

        LLM_MODEL: Nome del modello LLM per la generazione di risposte RAG.
            - gpt-4o-mini: Modello veloce ed economico, buono per la maggior parte dei casi
            - gpt-4o: Modello più potente, per risposte di alta qualità
            - gpt-4-turbo: Alternativa a gpt-4o

        LLM_TEMPERATURE: Temperatura del modello LLM (0.0 - 2.0).
            - 0.0: Risposte deterministiche, sempre uguali
            - 0.7: Buon bilanciamento (default)
            - 1.0+: Risposte più creative/variabili

        RAG_NUM_DOCS: Numero di documenti da recuperare per il contesto RAG.
            Più documenti = più contesto, ma anche più token e costi.
    """

    # -------------------------------------------------------------------------
    # CONFIGURAZIONE OPENAI
    # -------------------------------------------------------------------------
    # os.getenv("NOME", "default") cerca la variabile NOME nell'ambiente.
    # Se non la trova, restituisce "default" (stringa vuota in questo caso).
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # -------------------------------------------------------------------------
    # CONFIGURAZIONE CHROMA (Vector Store)
    # -------------------------------------------------------------------------
    # Il path di default è relativo alla posizione di questo file.
    # Creiamo la directory chroma_db nella root del progetto.
    # -------------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(Path(__file__).parent.parent.parent / "chroma_db")
    )

    # -------------------------------------------------------------------------
    # CONFIGURAZIONE CHUNKING (Divisione documenti)
    # -------------------------------------------------------------------------
    # Il chunking è necessario perché:
    # 1. I modelli di embedding hanno un limite di token in input
    # 2. Chunk più piccoli permettono ricerche più precise
    # 3. Chunk troppo piccoli perdono contesto
    #
    # int() converte la stringa restituita da os.getenv in intero.
    # -------------------------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # -------------------------------------------------------------------------
    # CONFIGURAZIONE LLM (per RAG)
    # -------------------------------------------------------------------------
    # Queste variabili configurano il modello di linguaggio usato per generare
    # le risposte nella pipeline RAG (Retrieval-Augmented Generation).
    # -------------------------------------------------------------------------
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    RAG_NUM_DOCS: int = int(os.getenv("RAG_NUM_DOCS", "4"))

    @classmethod
    def validate(cls) -> None:
        """
        Valida che la configurazione sia completa e corretta.

        Questo metodo viene chiamato prima di usare le API OpenAI
        per dare un errore chiaro se manca la configurazione.

        Raises:
            ValueError: Se OPENAI_API_KEY non è configurata.
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY non configurata. "
                "Crea il file .env nella root del progetto (copia da .env.example)"
            )


# =============================================================================
# ISTANZA GLOBALE
# =============================================================================
# Creiamo un'istanza globale della configurazione.
# In questo modo, da qualsiasi modulo possiamo fare:
#   from .config import config
#   print(config.EMBEDDING_MODEL)
# =============================================================================
config = Config()
