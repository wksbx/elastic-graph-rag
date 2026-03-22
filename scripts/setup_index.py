"""
setup_index.py — Create the Elasticsearch index for hybrid search.

This creates an index with:
  - Full-text fields (BM25 keyword search)
  - Dense vector field (semantic similarity via kNN)
  - Metadata fields (source, section, date, tags)
  - ELSER pipeline (optional, for learned sparse retrieval)

Run once before ingestion:
    python scripts/setup_index.py
"""

import os
import sys
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from rich.console import Console

load_dotenv()
console = Console()

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")
INDEX_NAME = os.getenv("INDEX_NAME", "knowledge-base")
_dims = os.getenv("EMBEDDING_DIMENSIONS")
if not _dims:
    console.print("[red]EMBEDDING_DIMENSIONS must be set in .env (e.g. 1024 for Voyage, 1536 for OpenAI, 384 for E5)[/red]")
    sys.exit(1)
EMBEDDING_DIMENSIONS = int(_dims)


def get_client() -> Elasticsearch:
    return Elasticsearch(
        ES_HOST,
        basic_auth=("elastic", ES_PASSWORD),
        request_timeout=60,
    )


def create_index(es: Elasticsearch):
    """Create the hybrid search index."""

    if es.indices.exists(index=INDEX_NAME):
        console.print(f"[yellow]Index '{INDEX_NAME}' already exists. Delete it first to recreate.[/yellow]")
        console.print(f"  curl -X DELETE '{ES_HOST}/{INDEX_NAME}' -u elastic:$ELASTIC_PASSWORD")
        sys.exit(1)

    index_body = {
        "settings": {
            "number_of_shards": 1,       # Single shard for dev; increase for production
            "number_of_replicas": 0,      # No replicas for dev; set to 1+ for production
            "index": {
                "default_pipeline": "embedding-pipeline",  # Auto-embed on ingest
            },
            "analysis": {
                "analyzer": {
                    # Custom analyzer for better handling of technical content
                    "technical_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "stop",
                            "snowball",  # Stemming
                        ],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                # ---------------------------------------------------------------
                # Full-text fields (BM25 keyword search)
                # ---------------------------------------------------------------
                "content": {
                    "type": "text",
                    "analyzer": "technical_analyzer",
                    "fields": {
                        "exact": {  # Sub-field for exact phrase matching
                            "type": "text",
                            "analyzer": "standard",
                        }
                    },
                },
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword"},  # For aggregations/filtering
                    },
                },

                # ---------------------------------------------------------------
                # Dense vector field (semantic similarity via kNN)
                # ---------------------------------------------------------------
                "content_embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",          # Hierarchical navigable small world
                        "m": 16,                  # Connections per node (default: 16)
                        "ef_construction": 200,   # Build-time accuracy (higher = better, slower)
                    },
                },

                # ---------------------------------------------------------------
                # Metadata fields (filtering + citations)
                # ---------------------------------------------------------------
                "source_file": {"type": "keyword"},     # Original filename
                "section_path": {"type": "keyword"},    # e.g., "docs/api/auth.md > Authentication"
                "chunk_index": {"type": "integer"},     # Position within document
                "total_chunks": {"type": "integer"},    # Total chunks from this doc
                "tags": {"type": "keyword"},            # User-defined tags
                "last_updated": {"type": "date"},       # Document freshness
                "access_level": {"type": "keyword"},    # RBAC: "public", "internal", "confidential"

                # ---------------------------------------------------------------
                # Chunk context (for better citation display)
                # ---------------------------------------------------------------
                "parent_title": {"type": "keyword"},    # Document-level title
                "heading_hierarchy": {"type": "text"},  # e.g., "API Guide > Auth > OAuth2"
            }
        },
    }

    es.indices.create(index=INDEX_NAME, body=index_body)
    console.print(f"[green]✓ Index '{INDEX_NAME}' created successfully[/green]")
    console.print(f"  Dimensions: {EMBEDDING_DIMENSIONS}")
    console.print(f"  Similarity: cosine")
    console.print(f"  Analyzer: technical_analyzer (stemming + stopwords)")


def create_embedding_pipeline(es: Elasticsearch):
    """
    Create an ingest pipeline that auto-generates embeddings on ingest.

    This uses a placeholder — in production, replace with:
      - Elasticsearch inference API (if using ES-hosted models)
      - Or remove the pipeline and embed externally before indexing
    """

    # Check if pipeline already exists
    try:
        es.ingest.get_pipeline(id="embedding-pipeline")
        console.print("[yellow]Pipeline 'embedding-pipeline' already exists, skipping.[/yellow]")
        return
    except Exception:
        pass

    # Placeholder pipeline — embedding happens in the ingest script instead
    # This pipeline just ensures the field exists and copies content for search
    pipeline_body = {
        "description": "Placeholder pipeline for knowledge base ingestion. "
                       "Actual embedding is done in the ingest script.",
        "processors": [
            {
                "set": {
                    "field": "_source.ingested_at",
                    "value": "{{_ingest.timestamp}}",
                }
            }
        ],
    }

    es.ingest.put_pipeline(id="embedding-pipeline", body=pipeline_body)
    console.print("[green]✓ Ingest pipeline 'embedding-pipeline' created[/green]")


def print_next_steps():
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Copy .env.example → .env and fill in your API keys")
    console.print("  2. Place your .md files in data/docs/")
    console.print("  3. Run: python scripts/ingest.py")
    console.print("  4. Test: python scripts/search.py 'your query here'")


if __name__ == "__main__":
    console.print("[bold]Setting up Elasticsearch index for hybrid search...[/bold]\n")

    es = get_client()

    # Verify connection
    info = es.info()
    console.print(f"Connected to Elasticsearch {info['version']['number']}")

    create_embedding_pipeline(es)
    create_index(es)
    print_next_steps()
