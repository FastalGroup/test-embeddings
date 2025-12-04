# Vector Search - Progetto Didattico RAG con LangChain

Questo progetto è una guida pratica per comprendere i concetti fondamentali di:

- **Embeddings**: rappresentazioni vettoriali del testo
- **Vector Store**: database per ricerche di similarità semantica
- **RAG (Retrieval-Augmented Generation)**: arricchire le risposte LLM con documenti

## Obiettivo Didattico

Il codice è scritto con **documentazione estesa** per chi vuole capire come funzionano questi concetti. Ogni modulo contiene commenti dettagliati che spiegano:

- Cosa fa ogni funzione
- Perché si usa un certo approccio
- Come i componenti LangChain interagiscono tra loro

## Architettura

```
                         FLUSSO RAG

   Documenti .md                    Domanda utente
        |                                 |
        v                                 v
    +----------+                    +-----------+
    | Chunking |                    | Embedding |
    | (split)  |                    | (query)   |
    +----------+                    +-----------+
        |                                 |
        v                                 v
    +-----------+                   +------------+
    | Embedding |                   | Similarity |
    | (docs)    |<----------------->| Search     |
    +-----------+                   +------------+
        |                                 |
        v                                 v
    +----------+                    +-----------+
    | Chroma   |                    | Top-K Docs|
    | (persist)|                    | (context) |
    +----------+                    +-----------+
                                          |
                                          v
                                    +-----------+
                                    |  Prompt   |
                                    | + Context |
                                    +-----------+
                                          |
                                          v
                                    +-----------+
                                    |    LLM    |
                                    | (gpt-4o)  |
                                    +-----------+
                                          |
                                          v
                                      Risposta
```

## Struttura del Progetto

```
test-embeddings/
├── .env                    # Configurazione (API keys, modelli)
├── .env.example            # Template configurazione
├── pyproject.toml          # Dipendenze e entry points
├── chroma_db/              # Database vettoriale persistente
├── docs/                   # Documenti da indicizzare
└── src/vectorsearch/
    ├── config.py           # Gestione configurazione
    ├── embeddings.py       # Factory per modelli embedding
    ├── vectorstore.py      # Wrapper Chroma DB
    ├── indexer.py          # Pipeline indicizzazione documenti
    ├── cli.py              # CLI per vector search (vs)
    └── rag.py              # CLI per RAG (rag)
```

## Setup

### 1. Prerequisiti

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)

### 2. Installazione

```bash
# Clona il progetto
git clone https://github.com/FastalGroup/test-embeddings.git
cd test-embeddings

# Installa dipendenze
uv sync
```

### 3. Configurazione

```bash
# Copia il template
cp .env.example .env

# Modifica .env e inserisci la tua OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 4. Indicizzazione Documenti

```bash
# Indicizza i file .md dalla directory docs/
uv run vs index ./docs

# Verifica lo stato del vector store
uv run vs stats
```

## Utilizzo

### CLI Vector Search (`vs`)

```bash
# Ricerca per similarità
uv run vs search "come configurare"

# Ricerca con più risultati
uv run vs search "installazione" -k 10

# Mostra chunk completi con paginazione
uv run vs search "query" --full

# Modalità interattiva (REPL)
uv run vs repl

# Statistiche vector store
uv run vs stats
```

### CLI RAG (`rag`)

```bash
# Query RAG base
uv run rag "Come mi associo alla CNA?"

# Con più documenti di contesto
uv run rag "Quali sono i servizi?" -k 6

# Mostra i documenti usati come contesto
uv run rag "domanda" --show-context

# Mostra il prompt completo (debug)
uv run rag "domanda" --debug

# CONFRONTO: RAG vs LLM senza contesto
uv run rag "domanda" --compare
```

## Concetti Chiave

### Embeddings

Gli **embeddings** sono rappresentazioni numeriche (vettori) del testo. Testi con significato simile hanno vettori "vicini" nello spazio multidimensionale.

```
"Come mi associo?"    -->  [0.12, -0.34, 0.56, ...]  (1536 dimensioni)
"Voglio iscrivermi"   -->  [0.11, -0.32, 0.58, ...]  (vettore simile!)
"Che tempo fa?"       -->  [0.89, 0.23, -0.45, ...]  (vettore diverso)
```

### Vector Store (Chroma)

Un **vector store** è un database ottimizzato per:
1. Memorizzare vettori (embeddings)
2. Cercare i vettori più simili a una query (similarity search)

### RAG (Retrieval-Augmented Generation)

**RAG** combina retrieval e generazione:

1. **Retrieve**: Trova documenti rilevanti dal vector store
2. **Augment**: Inserisci i documenti nel prompt come contesto
3. **Generate**: L'LLM genera una risposta basata sul contesto

#### Perché RAG?

- L'LLM ha una "data di cutoff" e non conosce informazioni recenti
- L'LLM può "allucinare" (inventare) informazioni
- RAG permette di usare documenti proprietari/specifici
- Le risposte sono basate su fonti verificabili

### Confronto RAG vs No-RAG

Il flag `--compare` mostra la differenza:

```bash
uv run rag "Quali sono i vantaggi di associarsi?" --compare
```

Output:
```
============================================================
                      CONFRONTO RAG
============================================================

------------------------------------------------------------
                    RISPOSTA SENZA RAG
              (LLM senza contesto documenti)
------------------------------------------------------------

I vantaggi generali di associarsi a un'organizzazione includono...
[Risposta generica basata sulla conoscenza pre-training]

------------------------------------------------------------
                     RISPOSTA CON RAG
              (LLM con contesto da 4 documenti)
------------------------------------------------------------

I vantaggi di associarsi alla CNA includono:
1. Assistenza fiscale e tributaria...
2. Consulenza del lavoro...
[Risposta specifica basata sui documenti indicizzati]

============================================================
```

## Configurazione Avanzata

Variabili disponibili in `.env`:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `OPENAI_API_KEY` | - | API key OpenAI (obbligatoria) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modello embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Modello LLM per RAG |
| `LLM_TEMPERATURE` | `0.7` | Temperatura LLM (0.0-2.0) |
| `CHUNK_SIZE` | `1000` | Dimensione chunk in caratteri |
| `CHUNK_OVERLAP` | `200` | Sovrapposizione tra chunk |
| `RAG_NUM_DOCS` | `4` | Documenti per contesto RAG |

## Dipendenze

- **LangChain**: Framework per applicazioni LLM
- **Chroma**: Vector store locale
- **OpenAI**: Embeddings e LLM
- **python-dotenv**: Gestione configurazione

## Licenza

Progetto didattico - uso libero per apprendimento.
