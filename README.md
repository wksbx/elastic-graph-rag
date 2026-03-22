# elastic-graph-rag

Hybrid search and RAG over markdown documentation, powered by Elasticsearch and knowledge graphs.

Combines **BM25 keyword search**, **vector semantic search**, and **graph-based retrieval** to answer questions over your documents with cited sources.

## How It Works

This system has two retrieval pipelines that can work independently or together:

```
                          Your Markdown Documents
                                   |
                    +--------------+--------------+
                    |                             |
              Vector Pipeline               Graph Pipeline
              (Elasticsearch)              (nano-graphrag)
                    |                             |
        +---------+----------+         +----------+---------+
        |                    |         |                    |
    BM25 Search      kNN Vector    Entity              Community
    (keywords)       (semantic)    Extraction           Detection
        |                    |         |                    |
        +--------+-----------+    Knowledge Graph     Community
                 |                 (NetworkX)          Reports
            RRF Fusion                 |                    |
                 |              Local Search         Global Search
                 |             (entity-centric)      (thematic)
                 |                     |                    |
                 +----------+----------+--------------------+
                            |
                     LLM (Claude / GPT)
                            |
                      Cited Answer
```

### Vector Pipeline (Elasticsearch)

The core retrieval path. Documents are chunked, embedded, and indexed into Elasticsearch for hybrid search.

| Stage | What happens | Why |
|-------|-------------|-----|
| **Chunk** | Split markdown on heading boundaries, then paragraphs, then sentences. 512 tokens with 64-token overlap. | Keeps semantic units together. Overlap prevents information loss at boundaries. |
| **Embed** | Generate vector embeddings via Voyage AI, OpenAI, or ES inference. | Enables semantic similarity search beyond keyword matching. |
| **Index** | Store chunks with embeddings + metadata (source file, heading hierarchy, tags, access level). | Rich metadata enables filtering and citation in answers. |
| **Search** | BM25 full-text + kNN vector search, fused with RRF (Reciprocal Rank Fusion). | RRF combines ranked lists without score normalization -- best of both worlds. |
| **Generate** | Top-K chunks passed to Claude/GPT with instructions to cite sources. | Grounded answers with traceability back to source documents. |

### Graph Pipeline (nano-graphrag)

An optional layer that builds a knowledge graph from your documents, enabling queries that connect information across multiple documents or summarize themes across the entire corpus.

| Stage | What happens | Why |
|-------|-------------|-----|
| **Extract** | LLM identifies entities (people, concepts, systems) and relationships from each chunk. | Captures structured knowledge that flat text search misses. |
| **Build** | Entities become nodes, relationships become edges in a NetworkX graph. | Enables traversal -- "what connects X to Y?" |
| **Cluster** | Leiden algorithm groups related entities into communities at multiple hierarchy levels. | Enables thematic understanding of the corpus. |
| **Summarize** | LLM generates a report for each community describing its key entities and themes. | Powers global search -- answering questions about the entire dataset. |
| **Query** | Three modes: **local** (entity-centric traversal), **global** (community report map-reduce), **naive** (vector baseline). | Different question types need different retrieval strategies. |

### When to Use Which Pipeline

| Question type | Pipeline | Example |
|--------------|----------|---------|
| Specific fact lookup | Vector | "What is the rate limit for Pro tier?" |
| How-to questions | Vector | "How do I authenticate with the API?" |
| Multi-hop reasoning | Graph (local) | "How does OAuth2 authentication relate to rate limiting?" |
| Thematic / summary | Graph (global) | "What are the main topics covered in the documentation?" |
| Entity exploration | Graph (local) | "Tell me everything about the token refresh mechanism" |

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env -- set your API keys and choose embedding/LLM providers
```

### 2. Start Elasticsearch + Kibana

```bash
docker compose up -d
# Wait for healthy status (~30s)
docker compose logs -f elasticsearch | grep -m1 "started"
```

### 3. Install dependencies

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

### 6. Ingest and search

```bash
# Index documents into Elasticsearch
python scripts/ingest.py

# Hybrid search
python scripts/search.py "how do I authenticate?"

# Hybrid search + RAG answer with citations
python scripts/search.py "how do I authenticate?" --rag

# Filter by tag or access level
python scripts/search.py "deployment" --tag devops --access internal
```

### 7. Graph pipeline (optional)

```bash
# Build knowledge graph from your documents
python scripts/graph_ingest.py

# Query the graph
python scripts/graph_search.py "What are the main themes?" --mode global
python scripts/graph_search.py "How does X relate to Y?" --mode local

# Inspect extracted entities and relationships
python scripts/graph_search.py --entities
python scripts/graph_search.py --stats
```

## Project Structure

```
elastic-rag/
├── docker-compose.yml          # Elasticsearch 8.17 + Kibana
├── config/
│   └── elasticsearch.yml       # ES cluster tuning (memory, ML, circuit breakers)
├── scripts/
│   ├── setup_index.py          # Create ES index with hybrid mappings
│   ├── ingest.py               # Parse, chunk, embed, and index markdown files
│   ├── search.py               # Hybrid search (BM25 + kNN + RRF) + RAG generation
│   ├── graph_ingest.py         # Build knowledge graph via nano-graphrag
│   └── graph_search.py         # Query the knowledge graph (local/global/naive)
├── data/
│   └── docs/                   # Your markdown files go here
│       └── example-doc.md      # Sample document for testing
├── .env.example                # Configuration template
└── requirements.txt            # Python dependencies
```

## Configuration

All settings live in `.env`. See `.env.example` for the full template.

### Embedding providers

| Provider | `EMBEDDING_MODEL` | Model example | Dimensions | API key |
|----------|-------------------|---------------|------------|---------|
| Voyage AI | `voyage` | `voyage-3.5` | 1024 | `VOYAGE_API_KEY` |
| OpenAI | `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` |
| Elasticsearch | `elasticsearch` | `.multilingual-e5-small` | 384 | (self-hosted) |

### LLM providers

| Provider | `LLM_PROVIDER` | Model example | API key |
|----------|----------------|---------------|---------|
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |

### Key settings

| Setting | Default | Notes |
|---------|---------|-------|
| `CHUNK_SIZE` | 512 | Tokens per chunk. Smaller = more precise, larger = more context. |
| `CHUNK_OVERLAP` | 64 | ~12% overlap. Prevents information loss at boundaries. |
| `GRAPH_WORKING_DIR` | `./graph_cache` | Where nano-graphrag persists graph data. |
| `INDEX_NAME` | `knowledge-base` | Elasticsearch index name. |

## How Components Relate

```
.env                    Shared config for all scripts (API keys, model choices, paths)
  │
  ├── docker-compose.yml      Runs Elasticsearch (port 9200) + Kibana (port 5601)
  │     │
  │     └── config/elasticsearch.yml    Tuning for the ES cluster
  │
  ├── setup_index.py          Creates the ES index schema (run once)
  │     │
  │     └── Defines: field mappings, vector dimensions, analyzers, HNSW settings
  │
  ├── ingest.py               Reads markdown → chunks → embeds → indexes into ES
  │     │
  │     ├── Uses: EMBEDDING_MODEL to generate vectors
  │     ├── Writes to: INDEX_NAME in Elasticsearch
  │     └── Metadata: source_file, heading_hierarchy, tags, access_level
  │
  ├── search.py               Queries ES with hybrid search + optional RAG
  │     │
  │     ├── Uses: same EMBEDDING_MODEL (must match ingest)
  │     ├── Reads from: INDEX_NAME in Elasticsearch
  │     ├── Fusion: RRF combines BM25 + kNN results
  │     └── RAG: sends top-K chunks to LLM_PROVIDER for cited answer
  │
  ├── graph_ingest.py         Reads markdown → feeds to nano-graphrag
  │     │
  │     ├── Uses: LLM_PROVIDER for entity/relationship extraction
  │     ├── Uses: EMBEDDING_MODEL for entity embeddings
  │     ├── Writes to: GRAPH_WORKING_DIR (local files, not ES)
  │     └── Produces: knowledge graph + community reports
  │
  └── graph_search.py         Queries the knowledge graph
        │
        ├── Reads from: GRAPH_WORKING_DIR
        ├── Modes: global (themes), local (entities), naive (vector baseline)
        └── Uses: LLM_PROVIDER for answer synthesis
```

### Elasticsearch Index Schema

The `knowledge-base` index stores document chunks with these fields:

| Field | Type | Purpose |
|-------|------|---------|
| `content` | text | Full-text search (BM25) with custom technical analyzer |
| `content_embedding` | dense_vector | Semantic similarity search (kNN with HNSW) |
| `title` | text | Document title with keyword sub-field |
| `source_file` | keyword | Original filename for citations |
| `heading_hierarchy` | text | Nested heading path (e.g., "API Guide > Auth > OAuth2") |
| `section_path` | keyword | Breadcrumb for navigation |
| `chunk_index` / `total_chunks` | integer | Position tracking within a document |
| `tags` | keyword | User-defined tags from YAML frontmatter |
| `access_level` | keyword | RBAC filtering: `public`, `internal`, `confidential` |
| `last_updated` | date | Document freshness tracking |

### How Hybrid Search Works

```
User query: "How do I refresh an expired token?"
                    │
    ┌───────────────┼───────────────┐
    │                               │
    ▼                               ▼
BM25 Full-Text                 kNN Vector Search
  Matches: "token",              Matches: semantically
  "refresh", "expired"           similar chunks even
  with fuzzy matching            without exact keywords
    │                               │
    ▼                               ▼
  Rank list A                   Rank list B
  [chunk7, chunk3, chunk1]      [chunk3, chunk7, chunk9]
    │                               │
    └───────────┬───────────────────┘
                │
         RRF Fusion
    score = sum(1 / (rank + 60))
                │
                ▼
    Final: [chunk3, chunk7, chunk1, chunk9, ...]
                │
                ▼ (if --rag)
         Claude / GPT
    "Based on the documentation..."
    [Source: example-doc.md]
```

## Monitoring

Kibana is available at `http://localhost:5601` for index management, query testing (Dev Tools), and data exploration.

To set up Kibana authentication, generate a service account token:

```bash
# Generate token (run once, after ES is healthy)
curl -s -X POST -u "elastic:$ELASTIC_PASSWORD" \
  "http://localhost:9200/_security/service/elastic/kibana/credential/token/kibana-token"

# Add the token value to your .env as KIBANA_SERVICE_TOKEN
```

Login to Kibana with user `elastic` and your `ELASTIC_PASSWORD`.

## Tuning Guide

### Chunk size
- Start with 512 tokens. If answers miss context, increase to 768.
- If answers include too much irrelevant text, decrease to 256.

### Retrieval quality
- Use `--top-k 10` and examine which results are relevant vs noise.
- If keyword matches dominate, lower the `content^2` boost in `search.py`.
- If semantic matches dominate, increase the phrase match boost.

### Graph pipeline
- Entity extraction quality depends on your LLM. Claude and GPT-4o produce good results.
- For large document sets, graph ingestion will make many LLM API calls. Monitor costs.
- Use `--mode local` for entity-specific questions, `--mode global` for thematic questions.

### Scaling for production
- **Nodes**: Move from single-node to 3+ node cluster.
- **Shards**: Increase `number_of_shards` based on data volume (~20-40GB per shard).
- **Replicas**: Set `number_of_replicas: 1` for high availability.
- **Reranking**: Add a cross-encoder (e.g., Cohere Rerank) between retrieval and generation.
- **Security**: Enable TLS, configure RBAC roles.

## License

MIT
