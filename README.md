# finngraph-core

Triplet extraction pipeline for a knowledge graph of business relationships (M&A, investment, partnerships, supply contracts, etc.) mined from Korean financial and economic news.

## Project Overview

finngraph-core turns raw Korean news articles into structured `(subject, predicate, object)` triples describing business relationships between companies, and between companies and governments/countries — for example:

- `(SK Hynix, supplies, NVIDIA)` — supply contract
- `(Samsung Electronics, acquires, Harman)` — M&A
- `(Kakao, invests in, SM Entertainment)` — equity investment
- `(Chile, produces, lithium)` — country-level resource production

Pure macro events such as geopolitical events (sanctions, negotiations, war) and macroeconomic indicators (interest rates, FX) are out of scope; only COMPANY-centered business relationships are extracted.

This is not an API server — it is a [LangGraph](https://github.com/langchain-ai/langgraph) extraction pipeline (`app/graph`) that runs as a workflow of nodes:

```
normalizer → entity_extractor → relation_extractor → triplet_builder
```

- **normalizer** — standardizes entity names in the article using gazetteer dictionaries
- **entity_extractor** — extracts entities via gazetteer matching (a KPF-BERT-NER extractor also exists in `app/graph/nodes/entity_extractor.py` but is not wired into the current workflow)
- **relation_extractor** — LLM-based relation extraction between the found entities
- **triplet_builder** — filters relations against the predicate whitelist and assembles final `(s, p, o)` triplets

The resulting triplets are persisted to Neo4j.

## Directory Structure

```
finngraph-core/
├── app/
│   ├── main.py                        # entry point: seeds Neo4j, runs the workflow once
│   ├── crud.py                        # Neo4j write queries (triplet upsert)
│   ├── core/
│   │   ├── config.py                    # pydantic-settings Settings, loaded from .env
│   │   ├── db.py                        # Neo4j async driver wrapper
│   │   └── llm.py                       # get_llm(): Gemini(ChatGoogleGenerativeAI) 전용 팩토리
│   ├── graph/
│   │   ├── workflow.py                  # LangGraph Runner
│   │   ├── state.py                     # GraphState
│   │   ├── models.py                    # pydantic models (Entity, Triple, ...)
│   │   ├── nodes/
│   │   │   ├── gazetteer.py               # gazetteer matching: normalization + entity extraction
│   │   │   ├── entity_extractor.py        # KPF-BERT-NER extractor (not in current workflow)
│   │   │   ├── relation_extractor.py      # relation extraction (LLM)
│   │   │   └── triplet_builder.py         # assembles final (s, p, o) triplets
│   │   ├── ontology/
│   │   │   ├── predicate_dict.py          # predicate whitelist
│   │   │   └── gazetteers/                # entity dictionaries (KRX, US, country, commodity, product)
│   │   ├── prompts/
│   │   │   └── relation_extraction.py     # LLM prompt for relation extraction
│   │   └── utils/
│   │       └── kpf.py                     # KPF label constants and tag mapping
│   └── scripts/
│       └── seed_db.py                   # seeds Neo4j when empty
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

**Run tests**:

```bash
uv run pytest
```

## Environment Configuration

Copy `.env.example` to `.env` and fill in every value — `Settings` in `app/core/config.py` has no defaults, so a missing variable fails fast with a `ValidationError` at startup, even for variables you don't think you need (e.g. LangSmith tracing, or the API key of the LLM provider you aren't using).

## Install HuggingFace Model Locally

The KPF-BERT-NER extractor (`app/graph/nodes/entity_extractor.py`) loads [KPF-BERT-NER](https://huggingface.co/KPF/KPF-bert-ner) from the local `./KPF-bert-ner` directory, so the model must be cloned into the project root before using it (the current workflow uses gazetteer-based extraction instead, so this is only needed if you wire the NER extractor back in):

```bash
# install git-xet to pull the large model files
brew install git-xet
git xet install

# clone the model from the project root
git clone https://huggingface.co/KPF/KPF-bert-ner
```

KPF-BERT-NER bundles both the NER model and its KPF-BERT tokenizer, so cloning this one repo is enough — no separate tokenizer download needed.
