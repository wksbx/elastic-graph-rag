# AI-Powered Search Infrastructure

Hybrid search (BM25 + vector) over markdown documentation,
powered by Elasticsearch and RAG answer generation.

## Architecture

```
Markdown files → Chunker → Embedder → Elasticsearch (hybrid index)
                                            ↓
User query → Embed → Hybrid Search (BM25 + kNN + RRF fusion)
                          ↓
                     Top-K results → LLM → Cited answer
```

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — uncomment and fill in your chosen embedding + LLM provider
```

### 2. Start Elasticsearch

```bash
docker compose up -d
# Wait for Elasticsearch to be healthy (~30s)
docker compose logs -f elasticsearch | grep -m1 "started"
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Create the search index

```bash
python scripts/setup_index.py
```

### 5. Add your documents

Place `.md` files in `data/docs/`. Nested directories are supported.

```bash
cp -r ~/my-wiki/*.md data/docs/
```

### 6. Ingest documents

```bash
python scripts/ingest.py                    # All .md files in data/docs/
python scripts/ingest.py path/to/file.md    # Single file
python scripts/ingest.py --reindex          # Full re-ingestion
```

### 7. Search

```bash
# Hybrid search (BM25 + vector)
python scripts/search.py "how do I authenticate with the API?"

# Hybrid search + RAG answer generation
python scripts/search.py "how do I authenticate?" --rag

# Filter by tag or access level
python scripts/search.py "deployment" --tag devops --access internal

# JSON output for piping to other tools
python scripts/search.py "error handling" --json
```

## Project Structure

```
search-infra/
├── docker-compose.yml          # Elasticsearch + Kibana
├── config/
│   └── elasticsearch.yml       # ES tuning (memory, ML, circuit breakers)
├── scripts/
│   ├── setup_index.py          # Create index with hybrid mappings
│   ├── ingest.py               # Parse, chunk, embed, and index .md files
│   └── search.py               # Hybrid search + RAG query pipeline
├── data/
│   └── docs/                   # Your markdown files go here
├── .env.example                # Configuration template
└── requirements.txt            # Python dependencies
```

## Configuration

All settings live in `.env`. Copy `.env.example` to get started.

### Embedding providers

| Provider | `EMBEDDING_MODEL` | Model example | Dimensions | API key env var |
|----------|-------------------|---------------|------------|-----------------|
| Voyage AI | `voyage` | `voyage-3.5` | 1024 | `VOYAGE_API_KEY` |
| OpenAI | `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` |
| Elasticsearch | `elasticsearch` | `.multilingual-e5-small` | 384 | (none — self-hosted) |

### LLM providers (for RAG)

| Provider | `LLM_PROVIDER` | Model example | API key env var |
|----------|----------------|---------------|-----------------|
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |

### Ingestion settings

| Setting | Default | Notes |
|---------|---------|-------|
| `CHUNK_SIZE` | 512 tokens | Smaller = more precise retrieval, larger = more context per chunk |
| `CHUNK_OVERLAP` | 64 tokens | ~12% overlap. Prevents information loss at chunk boundaries |
| `EMBEDDING_DIMENSIONS` | — | Must match your chosen model (see table above) |

## Tuning Guide

### Chunk size
- Start with 512 tokens. If answers miss context, increase to 768.
- If answers include too much irrelevant text, decrease to 256.

### Retrieval quality
- Use `--top-k 10` and examine which results are relevant vs noise.
- If keyword matches dominate, lower the `content^2` boost in `search.py`.
- If semantic matches dominate, increase the phrase match boost.

### Adding a reranker (recommended for production)

Add a cross-encoder reranker between retrieval and RAG generation:

```python
# Example with Cohere Rerank
import cohere
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
reranked = co.rerank(query=query, documents=[h["_source"]["content"] for h in hits], top_n=5)
```

### Scaling for production
- **Nodes**: Move from single-node to 3+ node cluster
- **Shards**: Increase `number_of_shards` based on data volume (~20-40GB per shard)
- **Replicas**: Set `number_of_replicas: 1` for high availability
- **ML nodes**: Dedicate nodes with GPU for embedding inference
- **Security**: Enable TLS, configure RBAC roles

## Monitoring

Access Kibana at `http://localhost:5601` (user: `elastic`, password from `.env`).

## License

MIT
