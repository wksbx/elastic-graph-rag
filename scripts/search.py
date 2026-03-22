"""
search.py — Hybrid search + RAG query pipeline.

Retrieval flow:
  1. User query → embed with same model used for ingestion
  2. Run hybrid search: BM25 + kNN vector, fused with RRF
  3. (Optional) Rerank top-N results with a cross-encoder
  4. (Optional) Pass top-K chunks to LLM for RAG answer generation

Usage:
    python scripts/search.py "how do I authenticate?"           # Hybrid search only
    python scripts/search.py "how do I authenticate?" --rag     # Search + RAG answer
    python scripts/search.py "error code 403" --top-k 5         # Adjust result count
"""

import os
import sys
import json
import argparse
from typing import Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()
console = Console()

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")
INDEX_NAME = os.getenv("INDEX_NAME", "knowledge-base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Embedding (same model as ingestion — critical for consistency)
# ---------------------------------------------------------------------------
def embed_query(text: str) -> list:
    """Embed a single query string."""
    if EMBEDDING_MODEL == "voyage":
        from voyageai.client import Client as VoyageClient
        client = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))
        result = client.embed([text], model=EMBEDDING_MODEL_NAME)
        return result.embeddings[0]
    elif EMBEDDING_MODEL == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=[text])
        return response.data[0].embedding
    elif EMBEDDING_MODEL == "elasticsearch":
        es = Elasticsearch(ES_HOST, basic_auth=("elastic", ES_PASSWORD), request_timeout=30)
        result = es.ml.infer_trained_model(
            model_id=EMBEDDING_MODEL_NAME,
            docs=[{"text_field": text}],
        )
        return result["inference_results"][0]["predicted_value"]
    else:
        raise ValueError(
            f"Unsupported embedding model: {EMBEDDING_MODEL}. "
            f"Choose from: openai, voyage, elasticsearch"
        )


# ---------------------------------------------------------------------------
# Hybrid search with RRF
# ---------------------------------------------------------------------------
def hybrid_search(
    es: Elasticsearch,
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Execute hybrid search combining BM25 + vector kNN, fused with RRF.

    RRF (Reciprocal Rank Fusion) merges ranked lists without needing
    score normalization — it works by summing 1/(rank + k) across
    each retriever's results. This is the recommended fusion method
    for Elasticsearch hybrid search.
    """
    count = es.count(index=INDEX_NAME)["count"]
    if count == 0:
        console.print(
            f"[yellow]Index '{INDEX_NAME}' is empty. Run `python scripts/ingest.py` first.[/yellow]"
        )
        return []

    query_vector = embed_query(query)

    # Build filter clause if provided
    filter_clauses = []
    if filters:
        if "access_level" in filters:
            filter_clauses.append({"term": {"access_level": filters["access_level"]}})
        if "tags" in filters:
            filter_clauses.append({"terms": {"tags": filters["tags"]}})
        if "source_file" in filters:
            filter_clauses.append({"term": {"source_file": filters["source_file"]}})

    # -----------------------------------------------------------------------
    # Elasticsearch retriever syntax (8.14+)
    # Uses RRF to fuse BM25 and kNN results
    # -----------------------------------------------------------------------
    search_body = {
        "retriever": {
            "rrf": {
                "retrievers": [
                    # Retriever 1: BM25 full-text search
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "fields": [
                                                    "content^2",    # Boost content matches
                                                    "content.exact",
                                                    "title^1.5",
                                                    "heading_hierarchy",
                                                ],
                                                "type": "best_fields",
                                                "fuzziness": "AUTO",
                                            }
                                        },
                                        # Exact phrase boost for precision
                                        {
                                            "match_phrase": {
                                                "content": {
                                                    "query": query,
                                                    "boost": 3,
                                                }
                                            }
                                        },
                                    ],
                                    "filter": filter_clauses or [],
                                }
                            }
                        }
                    },
                    # Retriever 2: kNN vector similarity
                    {
                        "knn": {
                            "field": "content_embedding",
                            "query_vector": query_vector,
                            "k": top_k,
                            "num_candidates": top_k * 10,  # Wider candidate pool for accuracy
                            **({"filter": {"bool": {"filter": filter_clauses}}} if filter_clauses else {}),
                        }
                    },
                ],
                "rank_window_size": top_k * 5,  # RRF considers top N from each retriever
                "rank_constant": 60,             # Default RRF constant (higher = less weight to rank)
            }
        },
        "size": top_k,
        "_source": {
            "excludes": ["content_embedding"],  # Don't return the large vector
        },
        "highlight": {
            "fields": {
                "content": {
                    "fragment_size": 200,
                    "number_of_fragments": 2,
                    "pre_tags": ["**"],
                    "post_tags": ["**"],
                }
            }
        },
    }

    response = es.search(index=INDEX_NAME, body=search_body)
    return response["hits"]["hits"]


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL")


def generate_rag_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer using retrieved chunks as context.

    The prompt instructs the LLM to:
      - Answer ONLY from the provided context
      - Cite sources using [Source: filename] format
      - Say "I don't have enough information" if context is insufficient
    """

    # Build context from retrieved chunks
    context_parts = []
    for i, hit in enumerate(chunks):
        source = hit["_source"]
        context_parts.append(
            f"[Document {i+1}] Source: {source.get('source_file', 'unknown')}\n"
            f"Section: {source.get('heading_hierarchy', 'N/A')}\n"
            f"Content:\n{source['content']}\n"
        )
    context = "\n---\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based ONLY on the
provided context documents. Follow these rules strictly:

1. Answer the question using ONLY information from the provided documents.
2. After each claim, cite the source using [Source: filename] format.
3. If the context doesn't contain enough information to answer, say:
   "I don't have enough information in the knowledge base to answer this fully."
4. Be concise and direct. Use the same terminology as the source documents.
5. If multiple documents discuss the topic, synthesize them into a coherent answer.
"""

    user_prompt = f"""Context documents:

{context}

---

Question: {query}

Answer based only on the context above, with source citations:"""

    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = LLM_MODEL or "claude-sonnet-4-20250514"
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_block = next(b for b in response.content if b.type == "text")
        return text_block.text
    elif LLM_PROVIDER == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = LLM_MODEL or "gpt-4o"
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}. "
            f"Choose from: anthropic, openai"
        )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_results(query: str, hits: list[dict], rag_answer: Optional[str] = None):
    """Pretty-print search results and optional RAG answer."""

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Found {len(hits)} result(s)[/dim]\n")

    if rag_answer:
        console.print(Panel(
            Markdown(rag_answer),
            title="[bold green]RAG Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    for i, hit in enumerate(hits):
        source = hit["_source"]
        score = hit.get("_score", 0)

        # Use highlight if available, otherwise truncate content
        if "highlight" in hit and "content" in hit["highlight"]:
            preview = " ... ".join(hit["highlight"]["content"])
        else:
            preview = source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"]

        console.print(Panel(
            f"[dim]Score: {score:.4f} | Source: {source.get('source_file', '?')} | "
            f"Section: {source.get('heading_hierarchy', 'N/A')}[/dim]\n\n{preview}",
            title=f"[bold]#{i+1}[/bold] {source.get('parent_title', 'Untitled')}",
            border_style="blue",
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid search + RAG over your knowledge base")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--rag", action="store_true", help="Generate a RAG answer from results")
    parser.add_argument("--tag", action="append", help="Filter by tag (repeatable)")
    parser.add_argument("--access", default=None, help="Filter by access level")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST, basic_auth=("elastic", ES_PASSWORD), request_timeout=30)

    # Build filters
    filters = {}
    if args.tag:
        filters["tags"] = args.tag
    if args.access:
        filters["access_level"] = args.access

    # Search
    hits = hybrid_search(es, args.query, top_k=args.top_k, filters=filters or None)

    if args.json:
        # Raw JSON output for piping to other tools
        results = []
        for hit in hits:
            h = hit["_source"].copy()
            h.pop("content_embedding", None)
            h["_score"] = hit.get("_score")
            results.append(h)
        print(json.dumps(results, indent=2))
    else:
        # RAG answer if requested
        rag_answer = None
        if args.rag and hits:
            console.print("[dim]Generating RAG answer...[/dim]")
            rag_answer = generate_rag_answer(args.query, hits)

        display_results(args.query, hits, rag_answer)
