# CLAUDE.md - Contesto per Claude Code

Questo file fornisce contesto a Claude quando lavora su questo progetto.

## Panoramica Progetto

Progetto didattico Python per insegnare i concetti di:
- **Embeddings** e **Vector Store** (Chroma)
- **RAG** (Retrieval-Augmented Generation) con LangChain
- Integrazione con OpenAI (embeddings + LLM)

Il codice è scritto con documentazione estesa per scopi educativi.

## Stack Tecnologico

- **Python**: >= 3.12
- **Package Manager**: uv (non pip)
- **Framework**: LangChain
- **Vector Store**: Chroma (persistente su filesystem)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI gpt-4o-mini (configurabile)

## Struttura Progetto

```
test-embeddings/
├── .env                    # Configurazione (API keys) - NON committare
├── .env.example            # Template configurazione
├── pyproject.toml          # Dipendenze e entry points uv
├── chroma_db/              # Database vettoriale (generato)
├── docs/                   # Documenti .md da indicizzare
└── src/vectorsearch/
    ├── config.py           # Singleton configurazione da .env
    ├── embeddings.py       # Factory OpenAIEmbeddings
    ├── vectorstore.py      # Wrapper Chroma con persistenza
    ├── indexer.py          # Chunking e indicizzazione .md
    ├── cli.py              # CLI "vs" (index, search, stats, repl)
    └── rag.py              # CLI "rag" (query RAG con confronto)
```

## Comandi Principali

```bash
# Installazione dipendenze
uv sync

# Vector Search CLI
uv run vs index ./docs      # Indicizza documenti
uv run vs search "query"    # Ricerca similarità
uv run vs stats             # Statistiche DB
uv run vs repl              # Modalità interattiva

# RAG CLI
uv run rag "domanda"                    # Query RAG
uv run rag "domanda" --show-context     # Mostra documenti usati
uv run rag "domanda" --compare          # Confronta RAG vs no-RAG
```

## Convenzioni Codice

### Stile Didattico

Ogni modulo contiene:
1. **Docstring iniziale** con spiegazione concetti
2. **Commenti di sezione** con `# ===` per separare blocchi logici
3. **Docstring funzioni** con Args, Returns, Example
4. **Commenti inline** che spiegano il "perché", non solo il "cosa"

### Pattern Usati

- **Config Singleton**: `config.py` espone istanza globale `config`
- **Factory Pattern**: `get_embeddings()` in `embeddings.py`
- **Wrapper Pattern**: `VectorStore` wrappa Chroma
- **Dataclass**: `RAGResult` per risultati strutturati

### Gestione Errori

- `config.validate()` solleva `ValueError` se manca API key
- CLI cattura eccezioni e stampa su stderr con `sys.exit(1)`

## Entry Points (pyproject.toml)

```toml
[project.scripts]
vs = "vectorsearch.cli:main"
rag = "vectorsearch.rag:main"
```

## Configurazione (.env)

```bash
OPENAI_API_KEY=sk-...           # Obbligatoria
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_NUM_DOCS=4
```

## Flusso RAG

```
1. RETRIEVE: query → embedding → similarity search → top-K docs
2. AUGMENT:  docs + question → prompt template
3. GENERATE: prompt → LLM → risposta
```

Il flag `--compare` esegue anche una chiamata LLM senza contesto per mostrare la differenza.

## Note per Modifiche

- Mantenere lo stile documentativo esteso
- Testare con `uv run vs --help` e `uv run rag --help`
- Il database Chroma è in `./chroma_db/` (può essere eliminato per reindicizzare)
- I documenti sorgente sono in `./docs/`
