"""
Indicizzazione documenti Markdown.

Questo modulo gestisce il processo di preparazione dei documenti
per il vector store: caricamento, divisione in chunk, e indicizzazione.

CONCETTI CHIAVE:
----------------

1. PERCHÉ DIVIDERE I DOCUMENTI IN CHUNK?
   I modelli di embedding hanno limiti sulla lunghezza del testo in input
   (es: ~8000 token per text-embedding-3-small). Ma non è solo questo:

   - Chunk troppo grandi: Embedding "annacquato" che rappresenta troppi concetti
   - Chunk troppo piccoli: Perdita di contesto, risultati frammentati

   Il chunking bilancia questi aspetti per ottenere ricerche efficaci.

2. CHUNK SIZE E OVERLAP
   - chunk_size: Dimensione massima di ogni pezzo (in caratteri)
   - chunk_overlap: Quanti caratteri si sovrappongono tra chunk consecutivi

   Esempio con overlap:
   Documento: "ABCDEFGHIJKLMNOP" (16 caratteri)
   chunk_size=10, chunk_overlap=3

   Chunk 1: "ABCDEFGHIJ" (caratteri 0-9)
   Chunk 2: "HIJKLMNOP"  (caratteri 7-16, overlap di 3: HIJ)

   L'overlap evita di "tagliare" concetti importanti tra due chunk.

3. RECURSIVE CHARACTER TEXT SPLITTER
   LangChain offre diversi text splitter. RecursiveCharacterTextSplitter:
   - Prova a dividere prima su "\n\n" (paragrafi)
   - Se i pezzi sono ancora troppo grandi, divide su "\n" (righe)
   - Poi su " " (parole)
   - Infine su "" (caratteri singoli)

   Questo preserva la struttura logica del testo il più possibile.

4. DOCUMENT LOADER
   I document loader di LangChain leggono file e li convertono in oggetti
   Document con:
   - page_content: il testo
   - metadata: informazioni sul file (path, nome, ecc.)

   Usiamo UnstructuredMarkdownLoader, il loader specifico di LangChain per file
   Markdown. Questo loader utilizza la libreria 'unstructured' per processare
   correttamente la struttura del documento Markdown (titoli, paragrafi, liste, ecc.).

5. PIPELINE DI INDICIZZAZIONE
   La pipeline completa è:

   File .md → Document Loader → Documenti → Text Splitter → Chunk →
   → Embedding (OpenAI) → Vettori → Vector Store (Chroma)
"""

from pathlib import Path

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config
from .vectorstore import VectorStore


def load_markdown_files(directory: str | Path) -> list[Document]:
    """
    Carica tutti i file Markdown (.md) da una directory.

    Questa funzione:
    1. Cerca ricorsivamente tutti i file .md nella directory
    2. Legge ogni file usando UnstructuredMarkdownLoader di LangChain
    3. Aggiunge metadati utili (source, filename)
    4. Restituisce una lista di oggetti Document

    UnstructuredMarkdownLoader vs TextLoader:
    - TextLoader: legge il file come testo grezzo
    - UnstructuredMarkdownLoader: processa la struttura Markdown, riconoscendo
      titoli, paragrafi, liste, codice, ecc. Questo può migliorare la qualità
      del chunking e delle ricerche semantiche.

    Args:
        directory: Path della directory da scansionare.
            Può essere un oggetto Path o una stringa.
            La ricerca è ricorsiva (include sottodirectory).

    Returns:
        Lista di oggetti Document. Ogni Document ha:
        - page_content: contenuto testuale del file (processato)
        - metadata: dizionario con "source" (path completo) e "filename" (nome file)

    Raises:
        FileNotFoundError: Se la directory non esiste.
        ValueError: Se non vengono trovati file .md.

    Example:
        >>> docs = load_markdown_files("./docs")
        >>> print(len(docs))
        3
        >>> print(docs[0].metadata["filename"])
        "guida.md"
    """
    # Converti in Path per usare i metodi di pathlib
    directory = Path(directory)
    documents: list[Document] = []

    # Verifica che la directory esista
    if not directory.exists():
        raise FileNotFoundError(f"Directory non trovata: {directory}")

    # glob("**/*.md") cerca ricorsivamente tutti i file .md
    # ** significa "qualsiasi sottodirectory, anche annidata"
    md_files = list(directory.glob("**/*.md"))

    if not md_files:
        raise ValueError(f"Nessun file .md trovato in: {directory}")

    # Carica ogni file
    for md_file in md_files:
        # UnstructuredMarkdownLoader è il loader specifico per file Markdown
        # - mode="single": restituisce un singolo Document per file
        # - strategy="fast": parsing veloce senza OCR o modelli ML
        loader = UnstructuredMarkdownLoader(
            str(md_file),
            mode="single",
            strategy="fast",
        )

        # load() restituisce una lista di Document
        docs = loader.load()

        # Aggiungi metadati personalizzati
        # Questi saranno utili per filtrare le ricerche o mostrare la fonte
        for doc in docs:
            doc.metadata["source"] = str(md_file)      # Path completo
            doc.metadata["filename"] = md_file.name    # Solo nome file

        documents.extend(docs)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Divide i documenti in chunk di dimensione gestibile.

    Il text splitter divide il testo cercando di mantenere
    la coerenza semantica: preferisce dividere su paragrafi,
    poi su righe, poi su spazi.

    Args:
        documents: Lista di Document da dividere.

        chunk_size: Dimensione massima di ogni chunk in caratteri.
            Se None, usa config.CHUNK_SIZE (default: 1000).
            Valori consigliati: 500-2000 caratteri.

        chunk_overlap: Sovrapposizione tra chunk consecutivi in caratteri.
            Se None, usa config.CHUNK_OVERLAP (default: 200).
            Valori consigliati: 10-20% di chunk_size.

    Returns:
        Lista di Document, dove ogni documento è un chunk.
        Il numero di chunk sarà >= numero di documenti originali.
        I metadati vengono preservati in ogni chunk.

    Example:
        >>> docs = [Document(page_content="Testo molto lungo...")]
        >>> chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        >>> print(len(chunks))
        4
        >>> print(len(chunks[0].page_content))
        500
    """
    # Crea lo splitter con i parametri specificati
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        # Ordine dei separatori: dal più preferito al meno preferito
        # Prima prova a dividere su doppio a-capo (paragrafi)
        # Poi su singolo a-capo (righe)
        # Poi su spazi (parole)
        # Infine su caratteri singoli (ultima risorsa)
        separators=["\n\n", "\n", " ", ""],
    )

    # split_documents gestisce automaticamente i metadati:
    # ogni chunk eredita i metadati del documento originale
    return splitter.split_documents(documents)


def index_directory(
    directory: str | Path,
    store: VectorStore,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """
    Pipeline completa: carica file .md, divide in chunk, indicizza.

    Questa è la funzione principale per indicizzare una directory di documenti.
    Combina le funzioni load_markdown_files e split_documents,
    poi salva tutto nel vector store.

    Flusso di esecuzione:
    1. Carica tutti i file .md dalla directory
    2. Divide ogni documento in chunk
    3. Per ogni chunk, genera l'embedding (chiamata API OpenAI)
    4. Salva chunk + embedding in Chroma

    Args:
        directory: Directory contenente i file .md da indicizzare.

        store: Istanza di VectorStore dove salvare i documenti.
            Deve essere già inizializzato.

        chunk_size: Dimensione chunk (opzionale, default da config).

        chunk_overlap: Overlap tra chunk (opzionale, default da config).

    Returns:
        Numero totale di chunk indicizzati.
        Questo numero sarà >= al numero di file, perché file lunghi
        vengono divisi in più chunk.

    Note:
        - L'indicizzazione è ADDITIVA: i nuovi documenti si aggiungono
          a quelli esistenti nel vector store
        - Per re-indicizzare, elimina prima la collection con delete_collection()
        - L'operazione può richiedere tempo per directory grandi (API calls)

    Example:
        >>> store = VectorStore()
        >>> count = index_directory("./docs", store)
        Caricati 5 documenti da ./docs
        Creati 23 chunk
        Indicizzati 23 chunk nel vector store
        >>> print(count)
        23
    """
    # Step 1: Carica i documenti
    documents = load_markdown_files(directory)
    print(f"Caricati {len(documents)} documenti da {directory}")

    # Step 2: Dividi in chunk
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"Creati {len(chunks)} chunk")

    # Step 3: Indicizza (genera embeddings e salva)
    # Questa è l'operazione più lenta perché fa chiamate API per ogni chunk
    store.add_documents(chunks)
    print(f"Indicizzati {len(chunks)} chunk nel vector store")

    return len(chunks)
