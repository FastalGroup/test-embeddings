"""
CLI (Command Line Interface) per Vector Search.

Questo modulo implementa l'interfaccia a riga di comando dell'applicazione
usando argparse, la libreria standard di Python per parsing degli argomenti.

CONCETTI CHIAVE:
----------------

1. COS'È UNA CLI?
   Una CLI (Command Line Interface) permette di interagire con un programma
   tramite comandi testuali nel terminale, invece che con un'interfaccia grafica.

   Esempio: `uv run vs search "come installare"`
   - "vs" è il comando principale
   - "search" è un sottocomando
   - "come installare" è un argomento

2. ARGPARSE
   argparse è la libreria standard Python per creare CLI.
   Gestisce automaticamente:
   - Parsing degli argomenti dalla riga di comando
   - Validazione dei tipi
   - Generazione dell'help (--help)
   - Messaggi di errore per argomenti mancanti

3. SUBCOMMAND PATTERN
   Molte CLI usano sottocomandi per organizzare funzionalità diverse:
   - `git commit`, `git push`, `git pull` (sottocomandi di git)
   - `docker run`, `docker build`, `docker ps`
   - `vs index`, `vs search`, `vs repl` (questa applicazione)

   In argparse si implementano con add_subparsers().

4. REPL (Read-Eval-Print Loop)
   Un REPL è un ambiente interattivo che:
   - Read: Legge l'input dell'utente
   - Eval: Valuta/esegue il comando
   - Print: Stampa il risultato
   - Loop: Ripete il ciclo

   È utile per esplorare dati o eseguire più query senza rieseguire il comando.

5. ENTRY POINT
   La funzione main() è definita come entry point in pyproject.toml:
   [project.scripts]
   vs = "vectorsearch.cli:main"

   Questo permette di eseguire il programma con `uv run vs` invece di
   `uv run python -m vectorsearch.cli`.
"""

import argparse
import shutil
import sys
from pathlib import Path

from .config import config
from .indexer import index_directory
from .vectorstore import VectorStore


def print_paginated(text: str, header: str = "") -> bool:
    """
    Stampa testo con paginazione interattiva.

    Divide il testo in linee e lo mostra pagina per pagina,
    adattandosi all'altezza del terminale.

    Args:
        text: Testo da stampare
        header: Header da mostrare sopra il contenuto (opzionale)

    Returns:
        True se l'utente ha visto tutto, False se ha interrotto (q)

    Note:
        - Premi INVIO per continuare alla prossima pagina
        - Premi 'q' per interrompere e tornare ai risultati
        - L'altezza della pagina si adatta al terminale
    """
    # Ottiene le dimensioni del terminale
    # shutil.get_terminal_size() restituisce (columns, lines)
    terminal_size = shutil.get_terminal_size()
    page_height = terminal_size.lines - 4  # Lascia spazio per prompt e header

    lines = text.split('\n')
    total_lines = len(lines)
    current_line = 0

    while current_line < total_lines:
        # Calcola quante linee mostrare
        end_line = min(current_line + page_height, total_lines)

        # Stampa header solo nella prima pagina
        if current_line == 0 and header:
            print(header)
            print("=" * min(len(header), terminal_size.columns))

        # Stampa le linee della pagina corrente
        for line in lines[current_line:end_line]:
            print(line)

        current_line = end_line

        # Se ci sono altre pagine, chiedi di continuare
        if current_line < total_lines:
            try:
                response = input("\n--- Premi INVIO per continuare, 'q' per uscire ---").strip().lower()
                if response == 'q':
                    return False
            except (EOFError, KeyboardInterrupt):
                print()
                return False

    return True


# =============================================================================
# FUNZIONI COMANDO
# =============================================================================
# Ogni sottocomando ha una funzione dedicata che riceve gli argomenti
# parsati da argparse come oggetto Namespace.
# =============================================================================


def cmd_index(args: argparse.Namespace) -> None:
    """
    Comando 'index': Indicizza documenti Markdown da una directory.

    Questo comando:
    1. Crea/apre il vector store
    2. Carica tutti i file .md dalla directory specificata
    3. Divide i documenti in chunk
    4. Genera embeddings e li salva nel vector store

    Args:
        args: Namespace con gli argomenti:
            - args.directory: path della directory con i file .md
            - args.model: modello di embedding da usare (opzionale)

    Esempio d'uso:
        uv run vs index ./docs
        uv run vs index ./docs --model text-embedding-3-large
    """
    # Crea il vector store (o apre quello esistente)
    # Se args.model è None, usa il default dalla configurazione
    store = VectorStore(model=args.model)

    # Esegue la pipeline di indicizzazione
    count = index_directory(args.directory, store)

    print(f"\nIndicizzazione completata: {count} chunk")


def cmd_search(args: argparse.Namespace) -> None:
    """
    Comando 'search': Esegue una ricerca per similarità.

    Questo comando:
    1. Apre il vector store esistente
    2. Converte la query in un embedding
    3. Cerca i documenti più simili
    4. Stampa i risultati con score e preview (o contenuto intero con --full)

    Args:
        args: Namespace con gli argomenti:
            - args.query: testo da cercare
            - args.k: numero di risultati (default: 5)
            - args.model: modello di embedding (opzionale)
            - args.full: se True, mostra chunk interi con paginazione

    Note sul risultato:
        Lo "score" è la distanza coseno tra query e documento.
        - Score basso (es: 0.2) = molto simile
        - Score alto (es: 1.5) = poco simile

    Esempio d'uso:
        uv run vs search "come configurare"
        uv run vs search "installazione" -k 10
        uv run vs search "query" --full
    """
    store = VectorStore(model=args.model)
    results = store.search(args.query, k=args.k)

    if not results:
        print("Nessun risultato trovato.")
        return

    print(f"\nRisultati per: '{args.query}' ({len(results)} trovati)\n")

    # Modalità full: mostra chunk interi con paginazione
    if args.full:
        for i, (doc, score) in enumerate(results, 1):
            # Prepara l'header del risultato
            source = doc.metadata.get('source', 'N/A')
            # Estrae solo il nome del file dal path
            filename = Path(source).name if source != 'N/A' else source
            header = f"[{i}/{len(results)}] Score: {score:.4f} | File: {filename}"

            # Mostra il contenuto completo con paginazione
            if not print_paginated(doc.page_content, header):
                # L'utente ha premuto 'q', interrompi
                break

            # Tra un risultato e l'altro, chiedi se continuare
            if i < len(results):
                try:
                    response = input(f"\n>>> Premi INVIO per il prossimo risultato, 'q' per uscire: ").strip().lower()
                    if response == 'q':
                        break
                    print()  # Linea vuota prima del prossimo risultato
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
    else:
        # Modalità normale: mostra preview
        for i, (doc, score) in enumerate(results, 1):
            print(f"[{i}] Score: {score:.4f}")
            print(f"    Source: {doc.metadata.get('source', 'N/A')}")
            # Mostra solo i primi 200 caratteri del contenuto
            print(f"    {doc.page_content[:200]}...")
            print()


def cmd_stats(args: argparse.Namespace) -> None:
    """
    Comando 'stats': Mostra statistiche del vector store.

    Utile per verificare:
    - Quanti documenti sono indicizzati
    - Quale modello di embedding è in uso
    - Dove sono salvati i dati

    Args:
        args: Namespace con gli argomenti:
            - args.model: modello di embedding (opzionale)

    Esempio d'uso:
        uv run vs stats
    """
    store = VectorStore(model=args.model)
    stats = store.get_stats()

    print("\n=== Vector Store Stats ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def cmd_repl(args: argparse.Namespace) -> None:
    """
    Comando 'repl': Avvia modalità interattiva.

    Il REPL permette di eseguire più ricerche senza rieseguire il comando.
    È utile per esplorare i documenti indicizzati.

    Comandi disponibili nel REPL:
    - Qualsiasi testo: esegue una ricerca
    - /model <nome>: cambia il modello di embedding
    - /stats: mostra statistiche
    - /help: mostra aiuto
    - exit o quit: esce dal REPL

    Args:
        args: Namespace con gli argomenti:
            - args.model: modello di embedding iniziale (opzionale)

    Note tecniche:
        - Il loop usa input() per leggere dalla stdin
        - Gestisce Ctrl+C e Ctrl+D per uscire gracefully
        - Cambiare modello richiede di ricaricare il vector store

    Esempio d'uso:
        uv run vs repl
        uv run vs repl --model text-embedding-3-large
    """
    # Determina il modello iniziale
    current_model = args.model or config.EMBEDDING_MODEL
    store = VectorStore(model=current_model)

    # Stampa il banner di benvenuto
    print(f"\nVector Search REPL")
    print(f"Modello: {current_model}")
    print("Comandi: /model <nome>, /stats, /help, exit\n")

    # Loop principale del REPL
    while True:
        try:
            # Legge l'input con prompt "vs> "
            query = input("vs> ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError: Ctrl+D (fine input)
            # KeyboardInterrupt: Ctrl+C (interruzione)
            print("\nBye!")
            break

        # Ignora righe vuote
        if not query:
            continue

        # --- Gestione comandi speciali ---

        # Uscita
        if query.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Bye!")
            break

        # Help
        if query == "/help":
            print("  /model <nome>  - Cambia modello embeddings")
            print("  /stats         - Mostra statistiche")
            print("  exit           - Esci")
            print("  <query>        - Esegui ricerca")
            continue

        # Statistiche
        if query == "/stats":
            stats = store.get_stats()
            print("\n=== Stats ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            continue

        # Cambio modello
        if query.startswith("/model "):
            new_model = query[7:].strip()  # Estrae il nome dopo "/model "
            if new_model:
                current_model = new_model
                # Ricrea il vector store con il nuovo modello
                # ATTENZIONE: se i documenti sono stati indicizzati con un altro
                # modello, i risultati non saranno corretti!
                store = VectorStore(model=current_model)
                print(f"Modello cambiato a: {current_model}")
            else:
                print(f"Modello attuale: {current_model}")
            continue

        # Comando sconosciuto
        if query.startswith("/"):
            print(f"Comando sconosciuto: {query}")
            continue

        # --- Esecuzione ricerca ---

        results = store.search(query, k=5)

        if not results:
            print("Nessun risultato.\n")
            continue

        # Stampa i risultati
        print()
        for i, (doc, score) in enumerate(results, 1):
            print(f"[{i}] Score: {score:.4f}")
            # Mostra un'anteprima più corta rispetto al comando search
            print(f"    {doc.page_content[:150]}...")
            print()


# =============================================================================
# ENTRY POINT E CONFIGURAZIONE ARGPARSE
# =============================================================================


def main() -> None:
    """
    Entry point principale della CLI.

    Questa funzione:
    1. Configura il parser principale e i sottocomandi
    2. Parsa gli argomenti dalla riga di comando
    3. Esegue la funzione associata al sottocomando scelto

    Struttura dei comandi:
        vs [--model MODEL] {index,search,stats,repl} ...

    Esempi:
        vs index ./docs
        vs search "query" -k 10
        vs --model text-embedding-3-large search "query"
        vs repl
        vs stats
    """
    # --- Parser principale ---
    parser = argparse.ArgumentParser(
        prog="vs",  # Nome del programma (mostrato nell'help)
        description="Vector Search - Ricerche di similarity con LangChain",
    )

    # Argomento globale --model, disponibile per tutti i sottocomandi
    parser.add_argument(
        "--model", "-m",
        help=f"Modello embeddings (default: {config.EMBEDDING_MODEL})",
        default=None,  # Se non specificato, usa il default dalla config
    )

    # --- Sottocomandi ---
    # add_subparsers crea un "contenitore" per i sottocomandi
    # dest="command" salva il nome del sottocomando scelto in args.command
    # required=True rende obbligatorio specificare un sottocomando
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sottocomando: index
    p_index = subparsers.add_parser(
        "index",
        help="Indicizza documenti .md da una directory"
    )
    p_index.add_argument(
        "directory",
        help="Directory contenente i file .md da indicizzare"
    )
    # set_defaults associa la funzione da chiamare per questo sottocomando
    p_index.set_defaults(func=cmd_index)

    # Sottocomando: search
    p_search = subparsers.add_parser(
        "search",
        help="Ricerca per similarity nel vector store"
    )
    p_search.add_argument(
        "query",
        help="Testo da cercare (può essere una frase o una domanda)"
    )
    p_search.add_argument(
        "-k",
        type=int,
        default=5,
        help="Numero massimo di risultati (default: 5)"
    )
    p_search.add_argument(
        "--full", "-f",
        action="store_true",
        help="Mostra i chunk interi con paginazione interattiva"
    )
    p_search.set_defaults(func=cmd_search)

    # Sottocomando: stats
    p_stats = subparsers.add_parser(
        "stats",
        help="Mostra statistiche del vector store"
    )
    p_stats.set_defaults(func=cmd_stats)

    # Sottocomando: repl
    p_repl = subparsers.add_parser(
        "repl",
        help="Avvia modalità interattiva (REPL)"
    )
    p_repl.set_defaults(func=cmd_repl)

    # --- Parsing e esecuzione ---

    # parse_args() legge sys.argv e restituisce un Namespace con gli argomenti
    args = parser.parse_args()

    try:
        # Chiama la funzione associata al sottocomando
        # (impostata con set_defaults(func=...))
        args.func(args)
    except Exception as e:
        # Gestione errori: stampa il messaggio e esce con codice 1
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


# Questo blocco viene eseguito solo se il file è eseguito direttamente
# (non quando è importato come modulo)
if __name__ == "__main__":
    main()
