"""
Gestione Chroma vector store.

Questo modulo fornisce un'interfaccia per salvare e cercare documenti
in un database vettoriale (vector store) usando Chroma.

CONCETTI CHIAVE:
----------------

1. COS'È UN VECTOR STORE?
   Un vector store è un database ottimizzato per memorizzare e cercare vettori.
   A differenza di un database tradizionale che cerca per corrispondenza esatta,
   un vector store trova elementi "simili" basandosi sulla distanza tra vettori.

   Database tradizionale: SELECT * FROM docs WHERE title = "gatto"
   Vector store: "Trova i documenti più simili a questo vettore"

2. COME FUNZIONA CHROMA?
   Chroma è un vector store open-source che può funzionare:
   - In memoria (dati persi al riavvio)
   - Con persistenza su disco (dati salvati in una directory)

   Struttura interna di Chroma:
   - Collection: come una "tabella" nel database
   - Documenti: testo originale + metadati
   - Embeddings: vettori associati ai documenti
   - IDs: identificatori univoci per ogni documento

3. SIMILARITY SEARCH (Ricerca per similarità)
   Quando cerchi un documento:
   1. La tua query viene convertita in un vettore
   2. Chroma calcola la distanza tra questo vettore e tutti quelli salvati
   3. Restituisce i K documenti con distanza minore (più simili)

   Il "score" restituito è la distanza: più basso = più simile.
   (Con distanza coseno: 0 = identico, 2 = opposto)

4. PERSISTENZA
   Con persist_directory, Chroma salva i dati su disco in formato SQLite.
   Questo permette di:
   - Chiudere il programma senza perdere i dati
   - Riaprire lo stesso vector store in seguito
   - Condividere il database tra esecuzioni diverse

5. COLLECTION
   Una collection è un gruppo logico di documenti.
   Puoi avere più collection nello stesso vector store,
   ad esempio una per ogni tipo di documento o progetto.
"""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import config
from .embeddings import get_embeddings


class VectorStore:
    """
    Wrapper per Chroma vector store con persistenza locale.

    Questa classe semplifica l'uso di Chroma fornendo un'interfaccia
    pulita per le operazioni comuni: aggiungere documenti, cercare,
    ottenere statistiche.

    Attributes:
        persist_dir: Directory dove Chroma salva i dati
        model: Nome del modello di embedding usato
        collection_name: Nome della collection in Chroma

    Example:
        >>> store = VectorStore()
        >>> store.add_documents([Document(page_content="Testo...")])
        >>> results = store.search("query di ricerca")
        >>> for doc, score in results:
        ...     print(f"Score: {score}, Testo: {doc.page_content[:50]}")
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        model: str | None = None,
        collection_name: str = "documents",
    ):
        """
        Inizializza o carica un vector store esistente.

        Se la directory di persistenza esiste già con dati, Chroma
        carica automaticamente i documenti esistenti. Altrimenti,
        crea un nuovo vector store vuoto.

        Args:
            persist_dir: Directory per la persistenza dei dati.
                Se None, usa config.CHROMA_PERSIST_DIR.
                I dati vengono salvati in formato SQLite in questa directory.

            model: Modello di embedding da usare per vettorizzare i documenti.
                Se None, usa config.EMBEDDING_MODEL.
                IMPORTANTE: Usa sempre lo stesso modello per indicizzare e cercare!
                Modelli diversi producono vettori incompatibili.

            collection_name: Nome della collection Chroma.
                Permette di avere più "tabelle" separate nello stesso database.
                Default: "documents"
        """
        # Salva i parametri come attributi dell'istanza
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.model = model
        self.collection_name = collection_name

        # Crea l'oggetto per generare embeddings
        # Questo verrà usato sia per indicizzare che per cercare
        self._embeddings = get_embeddings(model)

        # Crea la directory di persistenza se non esiste
        # parents=True: crea anche le directory intermedie
        # exist_ok=True: non dà errore se già esiste
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Inizializza Chroma
        # Se persist_directory contiene già dati, li carica automaticamente
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,  # Funzione per generare vettori
            persist_directory=self.persist_dir,   # Dove salvare i dati
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Aggiunge documenti al vector store.

        Per ogni documento:
        1. Estrae il testo (page_content)
        2. Genera l'embedding chiamando OpenAI
        3. Salva embedding + testo + metadati in Chroma

        Args:
            documents: Lista di oggetti Document di LangChain.
                Ogni Document ha:
                - page_content: il testo del documento
                - metadata: dizionario con informazioni aggiuntive
                  (es: {"source": "file.md", "page": 1})

        Returns:
            Lista di ID univoci assegnati ai documenti.
            Questi ID possono essere usati per recuperare o eliminare
            documenti specifici.

        Note:
            - Documenti duplicati vengono aggiunti (Chroma non fa dedup)
            - I metadati sono opzionali ma utili per filtrare le ricerche
            - L'operazione può essere lenta per molti documenti (chiamate API)
        """
        return self._store.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Ricerca documenti per similarità con la query.

        Processo di ricerca:
        1. La query viene convertita in un vettore (embedding)
        2. Chroma calcola la distanza coseno tra query e tutti i documenti
        3. Restituisce i K documenti con distanza minore

        Args:
            query: Testo da cercare. Può essere una domanda, una frase,
                o qualsiasi testo. Non serve che contenga le stesse parole
                dei documenti - la ricerca è semantica!

            k: Numero massimo di risultati da restituire.
                Default: 5. Aumenta se vuoi più risultati.

            filter: Filtro opzionale sui metadati dei documenti.
                Esempio: {"source": "manuale.md"} restituisce solo
                documenti con quel source nei metadati.

        Returns:
            Lista di tuple (Document, score).
            - Document: l'oggetto documento con page_content e metadata
            - score: distanza coseno dalla query (più basso = più simile)
              - 0.0: documenti identici
              - 1.0: documenti non correlati
              - 2.0: documenti opposti (raro)

        Example:
            >>> results = store.search("come installare python", k=3)
            >>> for doc, score in results:
            ...     print(f"[{score:.2f}] {doc.page_content[:100]}")
            [0.25] Per installare Python, scarica il file...
            [0.31] Python può essere installato su Windows...
            [0.45] Il linguaggio Python richiede...
        """
        return self._store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Restituisce statistiche sul vector store.

        Utile per debug e per verificare lo stato del database.

        Returns:
            Dizionario con:
            - collection_name: nome della collection
            - persist_dir: directory di persistenza
            - embedding_model: modello usato per gli embeddings
            - document_count: numero totale di documenti/chunk indicizzati
        """
        # Usa il metodo pubblico get() di Chroma per ottenere il conteggio
        # get() restituisce un dizionario con "ids", "documents", "metadatas", ecc.
        # Contiamo gli ID per avere il numero di documenti
        data = self._store.get()
        count = len(data["ids"]) if data and "ids" in data else 0

        return {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_model": self.model or config.EMBEDDING_MODEL,
            "document_count": count,
        }

    def delete_collection(self) -> None:
        """
        Elimina la collection e tutti i suoi documenti.

        ATTENZIONE: Operazione irreversibile!
        Tutti i documenti e i loro embeddings vengono eliminati.
        La directory di persistenza rimane ma sarà vuota.

        Utile per:
        - Ricominciare da zero con nuovi documenti
        - Cambiare modello di embedding (richiede re-indicizzazione)
        - Pulire dati di test
        """
        self._store.delete_collection()
