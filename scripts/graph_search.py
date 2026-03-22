"""
graph_search.py — Query the knowledge graph built by graph_ingest.py.

Supports three query modes:
  - global: Map-reduce over community reports for thematic/broad questions
  - local:  Entity-centric search with graph traversal for specific questions
  - naive:  Standard vector similarity over chunks (baseline comparison)

Usage:
    python scripts/graph_search.py "What are the main themes?"                    # Global (default)
    python scripts/graph_search.py "How does OAuth2 relate to rate limiting?" --mode local
    python scripts/graph_search.py "What is token refresh?" --mode naive
    python scripts/graph_search.py "Tell me about authentication" --context-only  # Show context, skip LLM
    python scripts/graph_search.py --entities                                     # List all extracted entities
    python scripts/graph_search.py --stats                                        # Show graph statistics
"""

from __future__ import annotations

import os
import sys
import asyncio
import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

GRAPH_WORKING_DIR = os.getenv("GRAPH_WORKING_DIR", "./graph_cache")


def _get_rag():
    """Load the GraphRAG instance from the existing working directory."""
    if not os.path.exists(GRAPH_WORKING_DIR):
        console.print(
            f"[red]Graph cache not found at '{GRAPH_WORKING_DIR}'.[/red]\n"
            f"Run `python scripts/graph_ingest.py` first."
        )
        sys.exit(1)

    # Import here to avoid slow startup when just showing help
    sys.path.insert(0, os.path.dirname(__file__))
    from graph_ingest import build_graph_rag

    return build_graph_rag()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
async def run_query(query: str, mode: str, context_only: bool, top_k: int):
    from nano_graphrag import QueryParam

    rag = _get_rag()

    param_kwargs = {"mode": mode, "only_need_context": context_only, "top_k": top_k}

    # Override OpenAI-specific JSON mode for Claude in global mode
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
    if llm_provider == "anthropic" and mode == "global":
        param_kwargs["global_special_community_map_llm_kwargs"] = {}

    param = QueryParam(**param_kwargs)

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Mode: {mode} | Top-K: {top_k}[/dim]\n")

    if not context_only:
        console.print("[dim]Generating answer...[/dim]")

    result = await rag.aquery(query, param=param)

    if context_only:
        console.print(
            Panel(
                result,
                title=f"[bold yellow]Context ({mode} mode)[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                Markdown(result),
                title=f"[bold green]Graph RAG Answer ({mode} mode)[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


# ---------------------------------------------------------------------------
# Graph inspection
# ---------------------------------------------------------------------------
def show_entities(limit: int):
    rag = _get_rag()
    graph = rag.chunk_entity_relation_graph._graph

    if graph.number_of_nodes() == 0:
        console.print("[yellow]No entities found. Run graph_ingest.py first.[/yellow]")
        return

    table = Table(title="Extracted Entities", show_lines=True)
    table.add_column("Entity", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Connections", justify="right")
    table.add_column("Description", max_width=60)

    sorted_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)

    for name, degree in sorted_nodes[:limit]:
        attrs = graph.nodes[name]
        entity_type = attrs.get("entity_type", "unknown")
        description = attrs.get("description", "")
        if len(description) > 60:
            description = description[:57] + "..."
        table.add_row(name, entity_type, str(degree), description)

    console.print(table)
    console.print(
        f"\n[dim]Showing top {min(limit, len(sorted_nodes))} of "
        f"{graph.number_of_nodes()} entities[/dim]"
    )


def show_stats():
    rag = _get_rag()
    graph = rag.chunk_entity_relation_graph._graph

    console.print("\n[bold]Knowledge Graph Statistics[/bold]\n")
    console.print(f"  Entities:      {graph.number_of_nodes()}")
    console.print(f"  Relationships: {graph.number_of_edges()}")

    if graph.number_of_nodes() > 0:
        degrees = [d for _, d in graph.degree()]
        console.print(f"  Avg degree:    {sum(degrees) / len(degrees):.1f}")
        console.print(f"  Max degree:    {max(degrees)}")

        # Entity type distribution
        type_counts = {}
        for _, attrs in graph.nodes(data=True):
            t = attrs.get("entity_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        console.print("\n[bold]Entity types:[/bold]")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {t}: {count}")

    # Show sample relationships
    if graph.number_of_edges() > 0:
        console.print("\n[bold]Sample relationships:[/bold]")
        for i, (src, tgt, attrs) in enumerate(graph.edges(data=True)):
            desc = attrs.get("description", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            console.print(f"  {src} → {tgt}: {desc}")
            if i >= 9:
                break

    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the knowledge graph"
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--mode",
        choices=["global", "local", "naive"],
        default="global",
        help="Query mode (default: global)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of results for vector search (default: 20)",
    )
    parser.add_argument(
        "--context-only", action="store_true",
        help="Return assembled context without LLM synthesis",
    )
    parser.add_argument(
        "--entities", action="store_true",
        help="List extracted entities instead of querying",
    )
    parser.add_argument(
        "--entities-limit", type=int, default=30,
        help="Max entities to show (default: 30)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show graph statistics",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.entities:
        show_entities(args.entities_limit)
    elif args.query:
        asyncio.run(
            run_query(args.query, args.mode, args.context_only, args.top_k)
        )
    else:
        parser.print_help()
