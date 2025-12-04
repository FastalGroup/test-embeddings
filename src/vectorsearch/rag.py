"""
RAG (Retrieval-Augmented Generation) - CLI Didattica.

Questo modulo implementa una pipeline RAG completa per rispondere a domande
usando documenti dal vector store come contesto.

================================================================================
                            COS'È RAG?
================================================================================

RAG (Retrieval-Augmented Generation) è una tecnica che combina:
1. **Retrieval**: Recupero di documenti rilevanti da un database
2. **Augmentation**: Arricchimento del prompt con il contesto recuperato
3. **Generation**: Generazione della risposta usando un LLM

Perché usare RAG invece di chiedere direttamente all'LLM?
- L'LLM ha una "data di cutoff" e non conosce informazioni recenti
- L'LLM può "allucinare" (inventare) informazioni
- RAG permette di usare documenti proprietari/specifici
- Le risposte sono basate su fonti verificabili

================================================================================
                         FLUSSO RAG
================================================================================

    Domanda utente: "Come mi associo alla CNA?"
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  1. RETRIEVE (Recupero)             │
    │                                     │
    │  La domanda viene convertita in     │
    │  embedding e usata per cercare      │
    │  documenti simili nel vector store. │
    │                                     │
    │  Input: "Come mi associo alla CNA?" │
    │  Output: [doc1, doc2, doc3, doc4]   │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  2. AUGMENT (Arricchimento)         │
    │                                     │
    │  I documenti recuperati vengono     │
    │  inseriti in un prompt template     │
    │  insieme alla domanda originale.    │
    │                                     │
    │  Template:                          │
    │  "Contesto: {docs}                  │
    │   Domanda: {question}               │
    │   Risposta:"                        │
    └─────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  3. GENERATE (Generazione)          │
    │                                     │
    │  Il prompt completo viene inviato   │
    │  all'LLM che genera una risposta    │
    │  basata sul contesto fornito.       │
    │                                     │
    │  LLM: gpt-4o-mini                   │
    │  Output: "Per associarti..."        │
    └─────────────────────────────────────┘
                           │
                           ▼
    Risposta: "Per associarti alla CNA puoi..."

================================================================================
                      COMPONENTI LANGCHAIN USATI
================================================================================

1. ChatOpenAI: Client per i modelli chat di OpenAI (gpt-4o-mini, gpt-4o, ecc.)
   - Gestisce la comunicazione con le API OpenAI
   - Supporta streaming, temperature, ecc.

2. PromptTemplate: Template per costruire prompt
   - Definisce la struttura del prompt
   - Gestisce le variabili da sostituire ({context}, {question})

3. Chroma (via VectorStore): Database vettoriale per il retrieval
   - Già implementato nel modulo vectorstore.py
"""

import argparse
import sys
from dataclasses import dataclass

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .config import config
from .vectorstore import VectorStore


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# Il prompt template definisce come strutturare la richiesta all'LLM.
# È fondamentale per ottenere risposte di qualità.
#
# Elementi chiave di un buon prompt RAG:
# 1. Ruolo/Contesto del sistema ("Sei un assistente...")
# 2. Istruzioni chiare ("Rispondi basandoti SOLO sul contesto")
# 3. Gestione casi limite ("Se non sai, dillo")
# 4. Struttura chiara (CONTESTO, DOMANDA, RISPOSTA)
# =============================================================================

RAG_PROMPT_TEMPLATE = """Sei un assistente della CNA (Confederazione Nazionale dell'Artigianato e della Piccola e Media Impresa).
Il tuo compito è rispondere alle domande degli utenti basandoti ESCLUSIVAMENTE sul contesto fornito.

ISTRUZIONI:
- Rispondi in modo chiaro, cordiale 
- Rispondi sempre nella stessa lingua usata dall'utente nella domanda
- Usa un tono amichevole e sii completo ed esaustivo nelle risposte 
- Usa le informazioni presenti nel contesto se la domanda si riferisce a CNA
- se la domanda non si riferisce a CNA, segnala che sei l'assistente di CNA e non puoi dare altre informazioni
- Se il contesto non contiene informazioni sufficienti per rispondere, dillo chiaramente
- Non inventare informazioni non presenti nel contesto
- Se appropriato, suggerisci di contattare la CNA per maggiori dettagli

CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:"""


# =============================================================================
# DATA CLASS PER I RISULTATI
# =============================================================================
# Usiamo una dataclass per strutturare i risultati della pipeline RAG.
# Questo rende il codice più leggibile e type-safe.
# =============================================================================

@dataclass
class RAGResult:
    """
    Risultato di una query RAG.

    Attributes:
        question: La domanda originale dell'utente
        context_docs: Lista di documenti usati come contesto
        prompt: Il prompt completo inviato all'LLM (per debug)
        response: La risposta generata dall'LLM
        model: Il modello LLM usato
    """
    question: str
    context_docs: list[tuple[Document, float]]  # (documento, score)
    prompt: str
    response: str
    model: str


# =============================================================================
# FASE 1: RETRIEVE (Recupero documenti)
# =============================================================================

def retrieve_context(
    query: str,
    k: int | None = None,
    embedding_model: str | None = None,
) -> list[tuple[Document, float]]:
    """
    Recupera documenti rilevanti dal vector store.

    Questa è la fase di RETRIEVAL della pipeline RAG.
    La query viene convertita in un embedding e usata per trovare
    i documenti più simili nel vector store.

    Come funziona internamente:
    1. La query viene inviata a OpenAI per generare l'embedding
    2. L'embedding viene confrontato con tutti i vettori nel DB
    3. I K documenti con distanza minore vengono restituiti

    Args:
        query: La domanda dell'utente
        k: Numero di documenti da recuperare (default: config.RAG_NUM_DOCS)
        embedding_model: Modello per gli embeddings (default: config.EMBEDDING_MODEL)

    Returns:
        Lista di tuple (Document, score) ordinate per rilevanza.
        Score più basso = più rilevante.

    Example:
        >>> docs = retrieve_context("Come mi associo?", k=3)
        >>> for doc, score in docs:
        ...     print(f"[{score:.2f}] {doc.page_content[:50]}...")
    """
    # Usa i valori di default dalla config se non specificati
    num_docs = k or config.RAG_NUM_DOCS

    # Crea/apre il vector store
    store = VectorStore(model=embedding_model)

    # Esegue la similarity search
    results = store.search(query, k=num_docs)

    return results


# =============================================================================
# FASE 2: AUGMENT (Costruzione del prompt)
# =============================================================================

def build_prompt(
    question: str,
    context_docs: list[tuple[Document, float]],
) -> str:
    """
    Costruisce il prompt completo per l'LLM.

    Questa è la fase di AUGMENTATION della pipeline RAG.
    Combina la domanda dell'utente con il contesto recuperato
    usando il template definito sopra.

    Il prompt risultante ha questa struttura:
    1. System message (ruolo e istruzioni)
    2. Contesto (documenti recuperati)
    3. Domanda dell'utente
    4. Placeholder per la risposta

    Args:
        question: La domanda dell'utente
        context_docs: Documenti recuperati dalla fase di retrieval

    Returns:
        Il prompt completo come stringa, pronto per essere inviato all'LLM.

    Note:
        I documenti vengono concatenati con separatori per chiarezza.
        Lo score viene incluso per debugging ma non mostrato all'LLM.
    """
    # Costruisce il contesto concatenando i documenti
    # Ogni documento è separato da una linea per chiarezza
    context_parts = []
    for i, (doc, score) in enumerate(context_docs, 1):
        # Estrae il nome del file dai metadata
        source = doc.metadata.get("source", "Sconosciuto")
        filename = Path(source).name if source != "Sconosciuto" else source

        # Formatta il documento
        context_parts.append(f"[Documento {i} - {filename}]\n{doc.page_content}")

    # Unisce tutti i documenti con doppio a-capo
    context_text = "\n\n".join(context_parts)

    # Crea il PromptTemplate di LangChain
    # Questo gestisce automaticamente la sostituzione delle variabili
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # Genera il prompt finale sostituendo le variabili
    prompt = prompt_template.format(
        context=context_text,
        question=question,
    )

    return prompt


# =============================================================================
# FASE 3: GENERATE (Generazione risposta)
# =============================================================================


def _get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    Crea un'istanza configurata di ChatOpenAI.

    Funzione helper per evitare duplicazione del codice di creazione LLM.

    Args:
        model: Nome del modello (default: config.LLM_MODEL)
        temperature: Temperatura per la generazione (default: config.LLM_TEMPERATURE)

    Returns:
        Istanza di ChatOpenAI configurata.
    """
    config.validate()

    model_name = model or config.LLM_MODEL
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    return ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=config.OPENAI_API_KEY,
    )


def generate_response(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Genera una risposta usando l'LLM.

    Questa è la fase di GENERATION della pipeline RAG.
    Il prompt completo (con contesto) viene inviato all'LLM
    che genera una risposta.

    Come funziona ChatOpenAI:
    1. Crea una connessione con le API OpenAI
    2. Invia il prompt come messaggio "human"
    3. Riceve la risposta come messaggio "assistant"
    4. Restituisce il contenuto testuale della risposta

    Args:
        prompt: Il prompt completo da inviare all'LLM
        model: Nome del modello (default: config.LLM_MODEL)
        temperature: Temperatura per la generazione (default: config.LLM_TEMPERATURE)
            - 0.0: Risposte deterministiche
            - 0.7: Bilanciato (default)
            - 1.0+: Più creativo/variabile

    Returns:
        La risposta generata dall'LLM come stringa.

    Note:
        - La API key viene presa automaticamente dalla config
        - Modelli disponibili: gpt-4o-mini, gpt-4o, gpt-4-turbo, ecc.
    """
    # Crea il client LLM usando l'helper
    llm = _get_llm(model, temperature)

    # Invoca l'LLM con il prompt
    # invoke() gestisce automaticamente la formattazione del messaggio
    response = llm.invoke(prompt)

    # response è un oggetto AIMessage
    # Usiamo .text (property di convenienza) per estrarre il testo della risposta
    # .text gestisce automaticamente il caso in cui content sia una lista di blocchi
    return response.text


# =============================================================================
# GENERAZIONE SENZA CONTESTO (per confronto didattico)
# =============================================================================
# Questa funzione genera una risposta SENZA usare il contesto RAG.
# È utile per mostrare la differenza tra:
# - Risposta "nuda" dell'LLM (può allucinare o essere generica)
# - Risposta RAG (basata su documenti specifici)
#
# Questo confronto è fondamentale per capire il VALORE del RAG.
# =============================================================================

def generate_without_context(
    question: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Genera una risposta SENZA contesto RAG.

    Questa funzione è usata per confrontare cosa risponde l'LLM
    quando NON ha accesso ai documenti specifici.

    IMPORTANTE: Questa è una chiamata LLM "pura", senza alcun documento
    di contesto. Il modello risponde solo con la sua conoscenza pre-training.

    Perché è utile questo confronto?
    1. Mostra cosa "sa" l'LLM di base (conoscenza pre-training)
    2. Evidenzia quando l'LLM "allucina" o inventa informazioni
    3. Dimostra il valore del RAG nel fornire risposte accurate

    Args:
        question: La domanda dell'utente
        model: Nome del modello LLM (default: config.LLM_MODEL)
        temperature: Temperatura per la generazione

    Returns:
        La risposta generata dall'LLM SENZA contesto documentale.
    """
    # Crea il client LLM usando l'helper
    llm = _get_llm(model, temperature)

    # =========================================================================
    # CHIAMATA LLM SENZA CONTESTO
    # =========================================================================
    # Usiamo il formato messaggi esplicito per chiarezza:
    # - SystemMessage: istruzioni generiche per l'assistente
    # - HumanMessage: solo la domanda dell'utente, NESSUN documento
    #
    # Questo è diverso dal RAG dove il contesto viene iniettato nel prompt.
    # =========================================================================
    messages = [
        SystemMessage(content="Sei un assistente utile. Rispondi alle domande in modo chiaro e professionale."),
        HumanMessage(content=question),  # Solo la domanda, nessun contesto!
    ]

    # Genera la risposta
    response = llm.invoke(messages)
    return response.text


# =============================================================================
# PIPELINE COMPLETA
# =============================================================================

def rag_query(
    question: str,
    k: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
    embedding_model: str | None = None,
) -> RAGResult:
    """
    Esegue la pipeline RAG completa.

    Questa funzione orchestra le tre fasi:
    1. RETRIEVE: Recupera documenti rilevanti
    2. AUGMENT: Costruisce il prompt con contesto
    3. GENERATE: Genera la risposta con l'LLM

    Args:
        question: La domanda dell'utente
        k: Numero di documenti per il contesto
        model: Modello LLM da usare
        temperature: Temperatura per la generazione
        embedding_model: Modello per gli embeddings

    Returns:
        RAGResult con tutti i dettagli della query:
        - question: domanda originale
        - context_docs: documenti usati
        - prompt: prompt completo (per debug)
        - response: risposta generata
        - model: modello usato

    Example:
        >>> result = rag_query("Come mi associo alla CNA?")
        >>> print(result.response)
        "Per associarti alla CNA puoi..."
    """
    # Fase 1: RETRIEVE
    context_docs = retrieve_context(
        query=question,
        k=k,
        embedding_model=embedding_model,
    )

    # Fase 2: AUGMENT
    prompt = build_prompt(question, context_docs)

    # Fase 3: GENERATE
    model_name = model or config.LLM_MODEL
    response = generate_response(
        prompt=prompt,
        model=model_name,
        temperature=temperature,
    )

    # Costruisce e restituisce il risultato
    return RAGResult(
        question=question,
        context_docs=context_docs,
        prompt=prompt,
        response=response,
        model=model_name,
    )


# =============================================================================
# OUTPUT FORMATTATO
# =============================================================================

def print_result(
    result: RAGResult,
    show_context: bool = False,
    show_prompt: bool = False,
) -> None:
    """
    Stampa il risultato RAG in formato leggibile.

    Args:
        result: Il risultato della query RAG
        show_context: Se True, mostra i documenti usati come contesto
        show_prompt: Se True, mostra il prompt completo (debug)
    """
    width = 70

    print()
    print("=" * width)
    print("                         RAG Query")
    print("=" * width)
    print()
    print(f"Domanda: {result.question}")
    print(f"Modello: {result.model}")
    print(f"Documenti recuperati: {len(result.context_docs)}")

    # Mostra il contesto se richiesto
    if show_context:
        print()
        print("-" * width)
        print("                      CONTESTO USATO")
        print("-" * width)
        print()

        for i, (doc, score) in enumerate(result.context_docs, 1):
            source = doc.metadata.get("source", "N/A")
            filename = Path(source).name if source != "N/A" else source
            print(f"[{i}] {filename} (score: {score:.3f})")

            # Mostra un'anteprima del contenuto (prime 200 caratteri)
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"    {preview}...")
            print()

    # Mostra il prompt se richiesto (debug)
    if show_prompt:
        print()
        print("-" * width)
        print("                    PROMPT COMPLETO")
        print("-" * width)
        print()
        print(result.prompt)

    # Mostra sempre la risposta
    print()
    print("-" * width)
    print("                        RISPOSTA")
    print("-" * width)
    print()
    print(result.response)
    print()
    print("=" * width)


# =============================================================================
# OUTPUT CONFRONTO (RAG vs NO-RAG)
# =============================================================================
# Questa funzione mostra un confronto side-by-side tra:
# - Risposta senza contesto (LLM "nudo")
# - Risposta con contesto RAG
#
# È uno strumento didattico potentissimo per capire perché RAG è utile.
# =============================================================================

def print_comparison(
    question: str,
    response_without_rag: str,
    result_with_rag: RAGResult,
    show_context: bool = False,
) -> None:
    """
    Stampa il confronto tra risposta senza RAG e con RAG.

    Questo output è progettato per essere didattico:
    - Mostra chiaramente le due risposte side-by-side
    - Include note esplicative su cosa significano
    - Aiuta a capire il valore del RAG

    Args:
        question: La domanda originale
        response_without_rag: Risposta dell'LLM senza contesto
        result_with_rag: Risultato completo della query RAG
        show_context: Se True, mostra anche i documenti usati come contesto
    """
    width = 70

    print()
    print("=" * width)
    print("                      CONFRONTO RAG")
    print("=" * width)
    print()
    print(f"Domanda: {question}")
    print(f"Modello: {result_with_rag.model}")
    print()

    # --- Sezione 1: Risposta SENZA RAG ---
    print("-" * width)
    print("                    RISPOSTA SENZA RAG")
    print("              (LLM senza contesto documenti)")
    print("-" * width)
    print()
    print(response_without_rag)
    print()
    print("[Nota: L'LLM risponde basandosi solo sulla sua conoscenza")
    print(" pre-training. Può essere generico o impreciso sui dettagli.]")
    print()

    # --- Sezione 2 (opzionale): Contesto usato ---
    if show_context:
        print("-" * width)
        print("                    CONTESTO RECUPERATO")
        print(f"                  ({len(result_with_rag.context_docs)} documenti)")
        print("-" * width)
        print()

        for i, (doc, score) in enumerate(result_with_rag.context_docs, 1):
            source = doc.metadata.get("source", "N/A")
            filename = Path(source).name if source != "N/A" else source
            print(f"[{i}] {filename} (score: {score:.3f})")

            # Mostra un'anteprima del contenuto
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"    {preview}...")
            print()

    # --- Sezione 3: Risposta CON RAG ---
    print("-" * width)
    print("                     RISPOSTA CON RAG")
    print(f"              (LLM con contesto da {len(result_with_rag.context_docs)} documenti)")
    print("-" * width)
    print()
    print(result_with_rag.response)
    print()
    print("[Nota: L'LLM risponde basandosi sui documenti specifici")
    print(" recuperati dal vector store. Risposte più accurate e pertinenti.]")
    print()
    print("=" * width)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """
    Entry point della CLI RAG.

    Utilizzo:
        uv run rag "domanda"
        uv run rag "domanda" -k 6 --show-context
        uv run rag "domanda" --model gpt-4o --debug
    """
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Query - Risposte basate su documenti con LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  uv run rag "Come mi associo alla CNA?"
  uv run rag "Quali sono i servizi?" -k 6
  uv run rag "domanda" --show-context
  uv run rag "domanda" --model gpt-4o --debug
  uv run rag "domanda" --compare          # Confronta RAG vs no-RAG
        """,
    )

    parser.add_argument(
        "question",
        help="La domanda da porre",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help=f"Numero documenti per il contesto (default: {config.RAG_NUM_DOCS})",
    )

    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Modello LLM (default: {config.LLM_MODEL})",
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help=f"Temperatura LLM 0.0-2.0 (default: {config.LLM_TEMPERATURE})",
    )

    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Modello embeddings (default: {config.EMBEDDING_MODEL})",
    )

    parser.add_argument(
        "--show-context", "-c",
        action="store_true",
        help="Mostra i documenti usati come contesto",
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Mostra il prompt completo inviato all'LLM",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Confronta risposta RAG con risposta senza contesto (2 chiamate LLM)",
    )

    args = parser.parse_args()

    try:
        # Modalità confronto: mostra sia risposta senza RAG che con RAG
        if args.compare:
            # --- FASE 1: Risposta SENZA contesto ---
            # Genera prima la risposta "nuda" dell'LLM
            response_without_rag = generate_without_context(
                question=args.question,
                model=args.model,
                temperature=args.temperature,
            )

            # --- FASE 2: Risposta CON RAG ---
            # Esegue la pipeline RAG completa
            result_with_rag = rag_query(
                question=args.question,
                k=args.k,
                model=args.model,
                temperature=args.temperature,
                embedding_model=args.embedding_model,
            )

            # --- Output confronto ---
            print_comparison(
                question=args.question,
                response_without_rag=response_without_rag,
                result_with_rag=result_with_rag,
                show_context=args.show_context,
            )

        else:
            # Modalità normale: solo RAG
            result = rag_query(
                question=args.question,
                k=args.k,
                model=args.model,
                temperature=args.temperature,
                embedding_model=args.embedding_model,
            )

            # Stampa il risultato
            print_result(
                result,
                show_context=args.show_context,
                show_prompt=args.debug,
            )

    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
