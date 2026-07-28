# finngraph-ai

Extraction pipeline and query API for a knowledge graph of business relationships (M&A, investment, partnerships, supply contracts, etc.) mined from Korean financial and economic news.

## Project Overview

finngraph-ai turns raw Korean news articles into structured `(subject, predicate, object)` triples describing business relationships between companies, and between companies and governments/countries — for example:

- `(SK Hynix, supplies, NVIDIA)` — supply contract
- `(Samsung Electronics, acquires, Harman)` — M&A
- `(Kakao, invests in, SM Entertainment)` — equity investment
- `(Chile, produces, lithium)` — country-level resource production

Pure macro events such as geopolitical events (sanctions, negotiations, war) and macroeconomic indicators (interest rates, FX) are out of scope; only COMPANY-centered business relationships are extracted.

The repository has two parts:

1. **Extraction pipeline** (`app/graph`) — a [LangGraph](https://github.com/langchain-ai/langgraph) workflow that extracts triplet from raw financial news articles as mentioned.

2. **API** (`app/api`) — a FastAPI service that reads the resulting graph from Neo4j and exposes endpoints such as a company's N-hop relationship network and theme-associated corporate relations.

## Directory Structure

```
finngraph-ai/
├── app/
│   ├── main.py
│   ├── models.py                    # domain models (placeholder)
│   ├── schemas.py                   # FastAPI request/response DTOs
│   ├── crud.py                      # Neo4j read queries backing the API routes
│   ├── api/
│   │   ├── main.py                    # FastAPI app + router registration
│   │   └── routes/
│   │       ├── company.py
│   │       ├── themes.py
│   │       └── news.py                # WIP, not yet registered on the router
│   ├── core/
│   │   ├── config.py                  # pydantic-settings Settings, loaded from .env
│   │   ├── db.py                      # Neo4j async driver wrapper
│   │   ├── llm.py                     # get_llm(): Gemini(ChatGoogleGenerativeAI) 전용 팩토리
│   │   └── exceptions.py              # global exception handling
│   └── graph/
│       ├── workflow.py                # LangGraph Runner
│       ├── state.py                   # GraphState
│       ├── models.py                  # pydantic models (Entity, SRLFrame, Triple, ...)
│       ├── nodes/
│       │   ├── ner.py                   # named entity recognition (KPF-BERT-NER)
│       │   ├── srl.py                   # semantic role labeling (LLM)
│       │   └── fpdf.py                  # predicate-dictionary filtering
│       ├── ontology/
│       │   ├── predicate_dict.py        # binary predicate whitelist
│       │   └── predicate_dict_nary.py   # n-ary predicate whitelist used by fpdf
│       └── utils/
│           └── kpf_labels.py          # KPF label constants and tag mapping
├── docs/                             # design notes and investigation write-ups
├── tests/                            # pytest suite
│   └── data/                          # sample article fixtures used by tests
├── KPF-bert-ner/                     # cloned HuggingFace model (gitignored, see below)
├── pyproject.toml                   # dependencies (uv-managed)
└── .env.example                     # required environment variables
```

## How to Run

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
# install dependencies
uv sync
```

Copy `.env.example` to `.env` and fill in every value (see [Environment Configuration](#environment-configuration) below) before running anything — both Docker Compose and the app read from `.env`.

**Start Neo4j** with Docker Compose:

```bash
# start Neo4j in the background
docker compose up -d

# check status / wait for the healthcheck to pass
docker compose ps

# stop it later
docker compose down
```

This launches a `finngraph-neo4j` container exposing the Neo4j Browser at http://localhost:7474 and the Bolt driver at `bolt://localhost:7687`. Data and logs are bind-mounted to `./neo4j_data` and `./neo4j_logs`. Credentials come from `NEO4J_USERNAME` / `NEO4J_PASSWORD` in your `.env`.

**Run the extraction pipeline** (`app/main.py`):

```bash
uv run python -m app.main
```

On first run this seeds Neo4j if it's empty, then builds the LangGraph workflow, invokes it once on the sample article baked into `app/main.py`, and prints `Graph completed successfully. N triplets saved.` on completion. Requires the Neo4j container (above) to be up.

**Run the FastAPI server**:

```bash
uv run fastapi dev app/api/main.py
```

This reads the graph from Neo4j and serves the query API. Populate the graph via the extraction pipeline first.

**Run tests**:

```bash
uv run pytest
```

## Environment Configuration

Copy `.env.example` to `.env` and fill in every value — `Settings` in `app/core/config.py` has no defaults, so a missing variable fails fast with a `ValidationError` at startup, even for variables you don't think you need (e.g. LangSmith tracing, or the API key of the LLM provider you aren't using).

## Install HuggingFace Model Locally

finngraph-ai uses [KPF-BERT-NER](https://huggingface.co/KPF/KPF-bert-ner) for entity recognition in the extraction pipeline.

The pipeline loads `KPF/KPF-bert-ner` via `from_pretrained`, so `transformers` will download and cache it automatically on first run.

If you'd rather fetch it ahead of time (e.g. for a faster first run or an offline environment), clone it manually:

```bash
# install git-xet to pull the large model files
brew install git-xet
git xet install

# clone the model from the project root
git clone https://huggingface.co/KPF/KPF-bert-ner
```

KPF-BERT-NER bundles both the NER model and its KPF-BERT tokenizer, so cloning this one repo is enough — no separate tokenizer download needed.
