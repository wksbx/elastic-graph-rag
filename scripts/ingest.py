"""
ingest.py — Ingest markdown files into the hybrid search index.

Pipeline:
  1. Read .md files from DOCS_PATH
  2. Parse headings to extract section hierarchy
  3. Chunk using recursive splitting with overlap
  4. Generate embeddings via OpenAI or self-hosted model
  5. Bulk index into Elasticsearch with metadata

Usage:
    python scripts/ingest.py                    # Ingest all files in DOCS_PATH
    python scripts/ingest.py path/to/file.md    # Ingest a single file
    python scripts/ingest.py --reindex          # Delete index and re-ingest everything
"""

import os
import re
import sys
import glob
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, cast

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")
INDEX_NAME = os.getenv("INDEX_NAME", "knowledge-base")
DOCS_PATH = os.getenv("DOCS_PATH", "./data/docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_embedder():
    """Return an embedding function based on config."""
    if EMBEDDING_MODEL == "openai":
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def embed_batch(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=texts,
            )
            return [item.embedding for item in response.data]

        return embed_batch
    elif EMBEDDING_MODEL == "voyage":
        from voyageai.client import Client as VoyageClient

        client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))

        def embed_batch(texts: list[str]) -> list[list[float]]:
            result = client.embed(texts, model=EMBEDDING_MODEL_NAME)
            return cast(list[list[float]], result.embeddings)

        return embed_batch
    elif EMBEDDING_MODEL == "elasticsearch":
        es = Elasticsearch(ES_HOST, basic_auth=("elastic", ES_PASSWORD), request_timeout=120)

        def embed_batch(texts: list[str]) -> list[list[float]]:
            embeddings = []
            for text in texts:
                result = es.ml.infer_trained_model(
                    model_id=EMBEDDING_MODEL_NAME,
                    docs=[{"text_field": text}],
                )
                embeddings.append(result["inference_results"][0]["predicted_value"])
            return embeddings

        return embed_batch
    else:
        raise ValueError(
            f"Unsupported embedding model: {EMBEDDING_MODEL}. "
            f"Choose from: openai, voyage, elasticsearch"
        )


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------
def extract_title(content: str) -> str:
    """Extract the first H1 heading as the document title."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def extract_tags(content: str) -> list[str]:
    """Extract tags from YAML frontmatter if present."""
    tags = []
    fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).split("\n"):
            if line.strip().startswith("tags:"):
                # Handle both inline [tag1, tag2] and list format
                tag_str = line.split(":", 1)[1].strip()
                tag_str = tag_str.strip("[]")
                tags = [t.strip().strip("'\"") for t in tag_str.split(",") if t.strip()]
    return tags


def build_heading_hierarchy(content: str, position: int) -> str:
    """
    Build the heading hierarchy at a given character position.
    Returns something like: "API Guide > Authentication > OAuth2"
    """
    headings = []
    current_levels = {}

    for match in re.finditer(r"^(#{1,6})\s+(.+)$", content[:position], re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        current_levels[level] = title
        # Clear deeper levels when a shallower heading appears
        for deeper in range(level + 1, 7):
            current_levels.pop(deeper, None)

    return " > ".join(current_levels[k] for k in sorted(current_levels))


# ---------------------------------------------------------------------------
# Chunking — recursive splitting with overlap
# ---------------------------------------------------------------------------
def chunk_markdown(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Recursively split markdown into chunks, respecting structure.

    Strategy:
      1. Split on H2/H3 headings first (section boundaries)
      2. If a section is still too long, split on paragraphs
      3. If a paragraph is still too long, split on sentences
      4. Apply overlap between chunks

    Returns list of dicts with 'text' and 'char_offset' keys.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # Strip frontmatter
    content_clean = re.sub(r"^---\n.*?\n---\n?", "", content, flags=re.DOTALL)

    # Split on heading boundaries (H2 and H3)
    sections = re.split(r"(?=^#{2,3}\s)", content_clean, flags=re.MULTILINE)
    sections = [s for s in sections if s.strip()]

    chunks = []

    for section in sections:
        section_tokens = enc.encode(section)

        if len(section_tokens) <= chunk_size:
            # Section fits in one chunk
            chunks.append({
                "text": section.strip(),
                "char_offset": content.find(section[:50]),
            })
        else:
            # Split section into paragraphs
            paragraphs = re.split(r"\n\n+", section)
            current_chunk = ""
            current_tokens = 0

            for para in paragraphs:
                para_tokens = len(enc.encode(para))

                if current_tokens + para_tokens <= chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "char_offset": content.find(current_chunk[:50]),
                        })

                    # Handle overlap: keep last N tokens of previous chunk
                    if overlap > 0 and current_chunk:
                        overlap_text = enc.decode(enc.encode(current_chunk)[-overlap:])
                        current_chunk = overlap_text + "\n\n" + para
                        current_tokens = overlap + para_tokens
                    else:
                        current_chunk = para
                        current_tokens = para_tokens

            # Don't forget the last chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "char_offset": content.find(current_chunk[:50]),
                })

    return chunks


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def process_file(filepath: str, embed_fn) -> Generator[dict, None, None]:
    """Process a single markdown file into indexable documents."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")

    title = extract_title(content)
    tags = extract_tags(content)
    last_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    chunks = chunk_markdown(content)
    total_chunks = len(chunks)

    if not chunks:
        console.print(f"  [yellow]⚠ No chunks produced from {path.name}[/yellow]")
        return

    # Batch embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = embed_fn(texts)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        hierarchy = build_heading_hierarchy(content, chunk.get("char_offset", 0))

        yield {
            "_index": INDEX_NAME,
            "_id": f"{path.stem}__chunk_{i}",
            "_source": {
                "content": chunk["text"],
                "content_embedding": embedding,
                "title": f"{title} (chunk {i+1}/{total_chunks})",
                "parent_title": title,
                "heading_hierarchy": hierarchy or title,
                "source_file": str(path.relative_to(DOCS_PATH)) if DOCS_PATH in str(path) else path.name,
                "section_path": f"{path.name} > {hierarchy}" if hierarchy else path.name,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "tags": tags,
                "last_updated": last_modified.isoformat(),
                "access_level": "internal",  # Default; override per-doc as needed
            },
        }


def ingest(files: list[str], reindex: bool = False):
    """Main ingestion entrypoint."""
    es = Elasticsearch(ES_HOST, basic_auth=("elastic", ES_PASSWORD), request_timeout=120)
    embed_fn = get_embedder()

    if reindex and es.indices.exists(index=INDEX_NAME):
        console.print(f"[red]Deleting index '{INDEX_NAME}' for reindex...[/red]")
        es.indices.delete(index=INDEX_NAME)
        # Re-run setup
        import subprocess
        subprocess.run([sys.executable, "scripts/setup_index.py"], check=True)

    console.print(f"\n[bold]Ingesting {len(files)} file(s) into '{INDEX_NAME}'[/bold]\n")

    total_indexed = 0
    errors = []

    for filepath in track(files, description="Processing files..."):
        try:
            actions = list(process_file(filepath, embed_fn))
            if actions:
                success, err = bulk(es, actions, raise_on_error=False)
                total_indexed += success
                if isinstance(err, list):
                    errors.extend(err)
        except Exception as e:
            from rich.markup import escape
            console.print(f"  [red]✗ Error processing {filepath}: {escape(str(e))}[/red]")
            errors.append(str(e))

    # Force refresh so documents are immediately searchable
    es.indices.refresh(index=INDEX_NAME)

    console.print(f"\n[green]✓ Indexed {total_indexed} chunks[/green]")
    if errors:
        console.print(f"[red]  {len(errors)} error(s) occurred[/red]")
        for err in errors[:5]:
            from rich.markup import escape
            console.print(f"    {escape(str(err))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown files into Elasticsearch")
    parser.add_argument("files", nargs="*", help="Specific files to ingest (default: all .md in DOCS_PATH)")
    parser.add_argument("--reindex", action="store_true", help="Delete and recreate the index first")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(DOCS_PATH, "**/*.md"), recursive=True))

    if not files:
        console.print(f"[yellow]No .md files found in {DOCS_PATH}[/yellow]")
        console.print("Place your markdown files there, or pass file paths as arguments.")
        sys.exit(1)

    ingest(files, reindex=args.reindex)
