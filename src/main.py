"""
Document Intelligence Refinery — CLI Entrypoint.

Orchestrates: Triage → Extraction Router → outputs profiles + ledger.

Usage:
    python -m src.main --input corpus/
    python -m src.main --input corpus/somefile.pdf
    python -m src.main --input corpus/ --config rubric/extraction_rules.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.data.vector_store import VectorStore
from src.agents.indexer import PageIndexBuilder

console = Console()
logger = logging.getLogger("refinery")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def process_document(
    pdf_path: Path,
    triage_agent: TriageAgent,
    router: ExtractionRouter,
    chunker: ChunkingEngine,
    vector_store: VectorStore,
    indexer: PageIndexBuilder,
    skip_extraction: bool = False,
) -> None:
    """Run the full pipeline on a single document."""
    console.print(f"\n[bold cyan]{'='*60}[/]")
    console.print(f"[bold]Processing:[/] {pdf_path.name}")
    console.print(f"[bold cyan]{'='*60}[/]")

    # ── Stage 1: Triage ──
    console.print("\n[bold yellow]Stage 1: Triage Agent[/]")
    profile = triage_agent.profile(pdf_path)

    # Display profile summary
    profile_table = Table(title=f"DocumentProfile: {pdf_path.name}", show_lines=True)
    profile_table.add_column("Field", style="cyan")
    profile_table.add_column("Value", style="green")

    profile_table.add_row("Doc ID", profile.doc_id)
    profile_table.add_row("Pages", str(profile.page_count))
    profile_table.add_row("File Size", f"{profile.file_size_bytes / 1024:.1f} KB")
    profile_table.add_row("Origin Type", profile.origin_type.value)
    profile_table.add_row("Layout Complexity", profile.layout_complexity.value)
    profile_table.add_row("Domain Hint", profile.domain_hint.value)
    profile_table.add_row("Language", f"{profile.language} ({profile.language_confidence:.1%})")
    profile_table.add_row("Estimated Cost (USD)", f"${profile.estimated_cost_usd:.4f}")
    profile_table.add_row("Avg Confidence", f"{profile.avg_confidence:.3f}")
    profile_table.add_row("Pages Below Threshold", str(profile.pages_below_threshold))

    console.print(profile_table)

    if skip_extraction:
        console.print("[dim]Skipping extraction (--triage-only mode)[/]")
        return

    # ── Stage 2: Extraction ──
    console.print("\n[bold yellow]Stage 2: Extraction Router[/]")

    try:
        result = router.extract(pdf_path, profile)

        # Display extraction summary
        result_table = Table(title="Extraction Result", show_lines=True)
        result_table.add_column("Field", style="cyan")
        result_table.add_column("Value", style="green")

        result_table.add_row("Strategy Used", result.strategy_used)
        result_table.add_row("Confidence", f"{result.confidence_score:.3f}")
        result_table.add_row("Pages Extracted", str(len(result.pages)))
        result_table.add_row("Total Tables", str(result.total_tables))
        result_table.add_row("Total Figures", str(result.total_figures))
        result_table.add_row("Total Text Blocks", str(result.total_text_blocks))
        result_table.add_row("Processing Time", f"{result.processing_time_s:.1f}s")

        console.print(result_table)
        
        # ── Stage 3: Semantic Chunking & Vector Store ──
        console.print("\n[bold yellow]Stage 3: Semantic Chunking Engine[/]")
        ldus = chunker.process_document(result)
        vector_store.ingest_ldus(ldus)
        console.print(f"[green]✔ Ingested {len(ldus)} chunks into ChromaDB[/]")
        
        # ── Stage 4: PageIndex Builder & FactTable ──
        console.print("\n[bold yellow]Stage 4: PageIndex Builder[/]")
        page_index = indexer.build_index(profile, ldus)
        console.print(f"[green]✔ Built PageIndex tree with {len(page_index.child_sections)} sections[/]")

    except Exception as e:
        console.print(f"[bold red]Extraction failed:[/] {e}")
        logger.exception(f"Extraction failed for {pdf_path.name}")


def main() -> None:
    """CLI entrypoint."""
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Document Intelligence Refinery — Process PDFs through the refinery pipeline",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a PDF file or directory of PDFs",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to extraction_rules.yaml (default: rubric/extraction_rules.yaml)",
    )
    parser.add_argument(
        "--triage-only",
        action="store_true",
        help="Only run triage (Stage 1), skip extraction",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set config path if specified
    if args.config:
        os.environ["REFINERY_CONFIG_PATH"] = args.config

    setup_logging(args.log_level)

    input_path = Path(args.input)

    if not input_path.exists():
        console.print(f"[bold red]Error:[/] {input_path} does not exist")
        sys.exit(1)

    # Collect PDF files
    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            console.print(f"[bold red]Error:[/] No PDF files found in {input_path}")
            sys.exit(1)
        console.print(f"[bold]Found {len(pdf_files)} PDF files in {input_path}[/]")
    else:
        pdf_files = [input_path]

    # Initialize agents
    triage_agent = TriageAgent()
    router = ExtractionRouter()
    chunker = ChunkingEngine()
    vector_store = VectorStore()
    indexer = PageIndexBuilder()

    console.print(Panel(
        "[bold]Document Intelligence Refinery[/]\n"
        f"Input: {input_path}\n"
        f"Documents: {len(pdf_files)}\n"
        f"Mode: {'Triage Only' if args.triage_only else 'Full Pipeline'}",
        title="🏭 Refinery",
        border_style="blue",
    ))

    # Process each document
    for pdf_path in pdf_files:
        try:
            process_document(
                pdf_path,
                triage_agent,
                router,
                chunker,
                vector_store,
                indexer,
                skip_extraction=args.triage_only,
            )
        except Exception as e:
            console.print(f"[bold red]Error processing {pdf_path.name}:[/] {e}")
            logger.exception(f"Pipeline error for {pdf_path.name}")

    console.print(f"\n[bold green]✅ Pipeline complete. Processed {len(pdf_files)} documents.[/]")
    console.print(f"[dim]Profiles saved to: .refinery/profiles/[/]")
    if not args.triage_only:
        console.print(f"[dim]Ledger saved to: .refinery/extraction_ledger.jsonl[/]")

    # Force immediate process exit to kill lingering ONNX/Chroma daemon threads
    # that cause hanging file locks on Windows
    os._exit(0)

if __name__ == "__main__":
    main()
