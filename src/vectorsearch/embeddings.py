"""
Gestione embeddings OpenAI.

Questo modulo si occupa di creare embeddings (rappresentazioni vettoriali) del testo
usando i modelli di OpenAI.

CONCETTI CHIAVE:
----------------

1. COSA SONO GLI EMBEDDINGS?
   Gli embeddings sono rappresentazioni numeriche (vettori) di testo.
   Ogni pezzo di testo viene convertito in un array di numeri (es: 1536 numeri
   per text-embedding-3-small).

   Esempio concettuale:
   - "gatto" -> [0.1, -0.3, 0.8, 0.2, ...]  (1536 numeri)
   - "felino" -> [0.15, -0.28, 0.75, 0.22, ...]  (simile a "gatto"!)
   - "automobile" -> [0.9, 0.1, -0.5, 0.7, ...]  (molto diverso)

2. PERCHÉ SONO UTILI?
   I vettori permettono di calcolare la "distanza" semantica tra testi.
   Testi con significato simile avranno vettori vicini nello spazio.
   Questo permette di trovare documenti "simili" a una query,
   anche se non contengono le stesse parole esatte.

3. COME FUNZIONA LA SIMILARITY SEARCH?
   - L'utente fa una query: "come addestrare un gatto"
   - La query viene convertita in un vettore
   - Si cercano i vettori più "vicini" nel database
   - I documenti corrispondenti sono i risultati

4. DISTANZA TRA VETTORI
   La "vicinanza" si misura con metriche come:
   - Distanza coseno: misura l'angolo tra due vettori (0 = identici, 2 = opposti)
   - Distanza euclidea: misura la distanza geometrica
   Chroma usa la distanza coseno di default.

5. MODELLI DI EMBEDDING OPENAI
   - text-embedding-3-small: 1536 dimensioni, veloce, economico (~$0.02/1M token)
   - text-embedding-3-large: 3072 dimensioni, più preciso (~$0.13/1M token)
   - text-embedding-ada-002: 1536 dimensioni, modello legacy

   Più dimensioni = rappresentazione più ricca ma più costosa e lenta.
"""

from langchain_openai import OpenAIEmbeddings

from .config import config


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """
    Crea un'istanza di OpenAIEmbeddings per generare vettori dal testo.

    Questa funzione è un "factory" che crea oggetti OpenAIEmbeddings configurati.
    OpenAIEmbeddings è una classe di LangChain che wrappa le API di OpenAI
    per la generazione di embeddings.

    Come funziona internamente:
    1. Quando chiami embeddings.embed_query("testo"), LangChain:
    2. Fa una chiamata HTTP alle API OpenAI
    3. OpenAI processa il testo col modello scelto
    4. Restituisce un array di float (il vettore)

    Args:
        model: Nome del modello di embedding da usare.
            Se None, usa il valore da config.EMBEDDING_MODEL.

            Modelli disponibili:
            - "text-embedding-3-small": Veloce ed economico, buono per la maggior parte dei casi
            - "text-embedding-3-large": Più preciso, per applicazioni che richiedono alta qualità
            - "text-embedding-ada-002": Modello precedente, ancora supportato

    Returns:
        OpenAIEmbeddings: Oggetto che può generare embeddings.

        Metodi principali dell'oggetto restituito:
        - embed_query(text: str) -> list[float]: Genera embedding per una query
        - embed_documents(texts: list[str]) -> list[list[float]]: Genera embeddings per più documenti

    Raises:
        ValueError: Se OPENAI_API_KEY non è configurata.

    Example:
        >>> embeddings = get_embeddings()
        >>> vector = embeddings.embed_query("Ciao mondo")
        >>> print(len(vector))  # 1536 per text-embedding-3-small
        1536
        >>> print(vector[:5])  # Primi 5 valori del vettore
        [0.123, -0.456, 0.789, ...]
    """
    # Valida che la API key sia presente prima di procedere
    config.validate()

    # Usa il modello specificato o quello di default dalla configurazione
    model_name = model or config.EMBEDDING_MODEL

    # Crea e restituisce l'oggetto OpenAIEmbeddings
    # Questo oggetto NON fa ancora chiamate API - le farà quando
    # chiamerai embed_query() o embed_documents()
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=config.OPENAI_API_KEY,
    )
