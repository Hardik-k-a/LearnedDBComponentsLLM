# LearnedDBComponentsLLM

A unified framework for **learned cardinality estimation** with LLM-based query generation and **active learning** strategies.

This project merges two complementary systems:
- **MSCN-based Cardinality Estimation** with Active Learning (random, uncertainty, MC Dropout)
- **LLM-based SQL Query Generation** using LangGraph + Ollama

## Project Structure

```
LearnedDBComponentsLLM/
├── config/                  # Unified DB config & settings
│   ├── db_config.py         # DB connection, count_rows, load_column_stats
│   └── settings.py          # Shared constants & env-var defaults
├── schema/                  # Database DDL & schema files
├── mscn/                    # MSCN model architecture
│   ├── model.py             # SetConv neural network
│   ├── data.py              # Data loading & encoding
│   └── util.py              # Encoding utilities
├── generation/              # Query generation
│   ├── query_generator.py   # LLM & synthetic query generation
│   ├── format_converter.py  # SQL ↔ MSCN CSV conversion
│   └── langraph_ollama/     # LangGraph pipeline (generate, fix, calculate)
├── labeling/                # Database labeling & bitmap generation
│   ├── db_labeler.py        # Execute COUNT(*) for true cardinalities
│   └── bitmap_utils.py      # Materialized sample bitmaps
├── training/                # Model training & active learning
│   ├── pipeline.py          # Complete end-to-end pipeline
│   ├── train.py             # MSCN training with AL strategies
│   └── experiment.py        # Quick experiment runner
├── evaluation/              # Benchmarking & strategy comparison
│   ├── compare_strategies.py
│   ├── run_benchmarks.py
│   └── pipeline_graphs.py   # 14-graph pipeline visualization
├── metrics/                 # LLM query quality metrics
│   ├── main.py              # Metrics orchestrator
│   ├── compare_models.py    # Cross-model comparison
│   └── ...                  # Validity, complexity, selectivity, etc.
├── utils/                   # Shared utilities
│   ├── io_utils.py          # JSON/JSONL/Excel I/O
│   ├── sql_utils.py         # SQL normalization & parsing
│   ├── session_utils.py     # Run directory management
│   └── logger.py            # Centralized logging
├── tools/                   # Standalone utilities
│   ├── merge_sessions.py    # Merge previous sessions
│   └── get_stats/           # Column statistics collector
├── prompts/                 # LLM prompt templates
├── docs/                    # Project documentation
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your database credentials and Ollama URL
```

### 3. Run the Training Pipeline
```bash
# Synthetic queries (no Ollama needed)
python -m training.pipeline job-light --synthetic --total-queries 500 --rounds 5

# LLM-generated queries
python -m training.pipeline job-light --total-queries 500 --model-name llama3.2
```

### 4. Compare Active Learning Strategies
```bash
python -m evaluation.compare_strategies job-light --queries 1000 --epochs 10 --rounds 5
```

### 5. Run LLM Query Generation Pipeline
```bash
python -m generation.langraph_ollama.main
```

## Key Features

| Feature | Module |
|---|---|
| MSCN Cardinality Estimation | `mscn/` |
| Active Learning (Random, Uncertainty, MC Dropout) | `training/` |
| LLM Query Generation (Ollama) | `generation/langraph_ollama/` |
| Synthetic Query Generation | `generation/query_generator.py` |
| Automated Benchmarking | `evaluation/run_benchmarks.py` |
| Query Quality Metrics | `metrics/` |
| 14-Graph Pipeline Visualization | `evaluation/pipeline_graphs.py` |

## Database Setup

This project requires a PostgreSQL database with the IMDB dataset. See `schema/` for DDL files and `schema/setup_imdb.py` for the data loader.

## Configuration

All runtime parameters can be set via environment variables (`.env` file) or CLI arguments. See `.env.example` for the full list.
