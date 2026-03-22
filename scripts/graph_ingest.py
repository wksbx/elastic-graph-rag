"""
graph_ingest.py — Build a knowledge graph from markdown files using nano-graphrag.

Pipeline:
  1. Read .md files from DOCS_PATH
  2. Feed raw text into nano-graphrag
  3. nano-graphrag handles: chunking → entity extraction (Claude) → relationship
     extraction → Leiden community detection → community report generation
  4. Graph + embeddings + community reports persisted to GRAPH_WORKING_DIR

Usage:
    python scripts/graph_ingest.py                    # Ingest all files in DOCS_PATH
    python scripts/graph_ingest.py path/to/file.md    # Ingest specific file(s)
    python scripts/graph_ingest.py --reindex          # Clear graph cache and re-ingest
"""

from __future__ import annotations

import os
import sys
import glob
import json
import hashlib
import asyncio
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCS_PATH = os.getenv("DOCS_PATH", "./data/docs")
GRAPH_WORKING_DIR = os.getenv("GRAPH_WORKING_DIR", "./graph_cache")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Custom LLM function for Anthropic Claude
# ---------------------------------------------------------------------------
def _make_anthropic_llm():
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = LLM_MODEL or "claude-sonnet-4-20250514"

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list = [],
        **kwargs,
    ) -> str:
        hashing_kv = kwargs.pop("hashing_kv", None)

        messages = list(history_messages) + [{"role": "user", "content": prompt}]

        # Check cache
        if hashing_kv is not None:
            args_hash = hashlib.md5(
                json.dumps(
                    {"model": model, "messages": messages}, sort_keys=True
                ).encode()
            ).hexdigest()
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.messages.create(
            model=model,
            max_tokens=16384,
            system=system_prompt or "",
            messages=messages,
        )
        result = response.content[0].text

        # Store in cache
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": model}}
            )
            await hashing_kv.index_done_callback()

        return result

    return llm_func


def _make_openai_llm():
    import openai

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = LLM_MODEL or "gpt-4o"

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list = [],
        **kwargs,
    ) -> str:
        hashing_kv = kwargs.pop("hashing_kv", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Check cache
        if hashing_kv is not None:
            args_hash = hashlib.md5(
                json.dumps(
                    {"model": model, "messages": messages}, sort_keys=True
                ).encode()
            ).hexdigest()
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.chat.completions.create(
            model=model,
            max_tokens=16384,
            messages=messages,
        )
        result = response.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": model}}
            )
            await hashing_kv.index_done_callback()

        return result

    return llm_func


# ---------------------------------------------------------------------------
# Custom embedding function
# ---------------------------------------------------------------------------
def _make_embedding_func():
    from nano_graphrag._utils import wrap_embedding_func_with_attrs

    if EMBEDDING_MODEL == "voyage":
        import voyageai

        client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIMENSIONS, max_token_size=16000
        )
        async def embed_func(texts: list[str]) -> np.ndarray:
            resp = await client.embed(
                texts, model=EMBEDDING_MODEL_NAME, input_type="document"
            )
            return np.array(resp.embeddings, dtype=np.float32)

        return embed_func

    elif EMBEDDING_MODEL == "openai":
        import openai

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIMENSIONS, max_token_size=8191
        )
        async def embed_func(texts: list[str]) -> np.ndarray:
            resp = await client.embeddings.create(
                model=EMBEDDING_MODEL_NAME, input=texts
            )
            return np.array(
                [item.embedding for item in resp.data], dtype=np.float32
            )

        return embed_func

    else:
        raise ValueError(
            f"Unsupported embedding model for graph: {EMBEDDING_MODEL}. "
            f"Choose from: voyage, openai"
        )


# ---------------------------------------------------------------------------
# Build the GraphRAG instance
# ---------------------------------------------------------------------------
def build_graph_rag():
    from nano_graphrag import GraphRAG

    if LLM_PROVIDER == "anthropic":
        llm_func = _make_anthropic_llm()
        extra_kwargs = {
            # Claude doesn't support OpenAI-style response_format
            "special_community_report_llm_kwargs": {},
        }
    elif LLM_PROVIDER == "openai":
        llm_func = _make_openai_llm()
        extra_kwargs = {}
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    return GraphRAG(
        working_dir=GRAPH_WORKING_DIR,
        enable_local=True,
        enable_naive_rag=True,
        # LLM
        best_model_func=llm_func,
        cheap_model_func=llm_func,
        best_model_max_token_size=200000 if LLM_PROVIDER == "anthropic" else 128000,
        cheap_model_max_token_size=200000 if LLM_PROVIDER == "anthropic" else 128000,
        best_model_max_async=4,
        cheap_model_max_async=4,
        # Embedding
        embedding_func=_make_embedding_func(),
        embedding_batch_num=16,
        embedding_func_max_async=8,
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
async def ingest_files(files: list[str], reindex: bool = False):
    if reindex and os.path.exists(GRAPH_WORKING_DIR):
        console.print(f"[red]Clearing graph cache at '{GRAPH_WORKING_DIR}'...[/red]")
        shutil.rmtree(GRAPH_WORKING_DIR)

    rag = build_graph_rag()

    console.print(
        f"\n[bold]Ingesting {len(files)} file(s) into graph "
        f"(working_dir: {GRAPH_WORKING_DIR})[/bold]\n"
    )

    for filepath in files:
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")
        console.print(f"  Processing [cyan]{path.name}[/cyan] ({len(content)} chars)...")

        try:
            await rag.ainsert(content)
            console.print(f"  [green]Done[/green]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    # Print graph stats
    graph = rag.chunk_entity_relation_graph._graph
    console.print(
        f"\n[green]Graph built: "
        f"{graph.number_of_nodes()} entities, "
        f"{graph.number_of_edges()} relationships[/green]"
    )

    # Show top entities by degree
    if graph.number_of_nodes() > 0:
        degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        console.print("\n[bold]Top entities by connections:[/bold]")
        for name, degree in degrees[:10]:
            console.print(f"  {name} ({degree} connections)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from markdown files"
    )
    parser.add_argument(
        "files", nargs="*",
        help="Specific files to ingest (default: all .md in DOCS_PATH)",
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Clear graph cache and re-ingest everything",
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(
            glob.glob(os.path.join(DOCS_PATH, "**/*.md"), recursive=True)
        )

    if not files:
        console.print(f"[yellow]No .md files found in {DOCS_PATH}[/yellow]")
        sys.exit(1)

    asyncio.run(ingest_files(files, reindex=args.reindex))
