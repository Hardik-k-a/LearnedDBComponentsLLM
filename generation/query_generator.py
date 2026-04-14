"""
query_generator.py
Generates unlabeled SQL queries using Ollama LLM and converts them
to MSCN-compatible structured format.

Does NOT execute any query on the database — only generates query structures.
"""

import json
import re
import time
import requests
from typing import List, Dict, Optional


# ── Dynamic Schema Prompt ────────────────────────────────────────────────

GENERATION_PROMPT = """You are a SQL query workload generator for a database.

Here is the database schema:
{schema}

Here are the column statistics (min, max, cardinality):
{stats}

Generate exactly {batch_size} SQL queries. Each query MUST follow these STRICT rules:
1. Use SELECT COUNT(*) FROM ... WHERE ... format
2. Use 1-3 tables from the schema above
3. Use proper table aliases (e.g., table_name t)
4. Include join conditions using Primary Key / Foreign Key relationships defined in the schema
5. Include 0-4 filter predicates using ONLY =, <, > operators with NUMERIC values
6. Use realistic filter values within the column ranges specified in the stats
7. ALL conditions in WHERE must be connected with AND only
8. Do NOT use OR, IN, LIKE, IS NULL, IS NOT NULL, BETWEEN, NOT, subqueries, or any other complex SQL
9. Each predicate must be a simple: column operator value (e.g., t.col > 100)

IMPORTANT: Use ONLY the columns and tables listed above. Do NOT invent columns.
IMPORTANT: Do NOT use string values in predicates. Use only numeric integer values.
IMPORTANT: Do NOT combine multiple conditions with OR. Use ONLY AND.

Vary the complexity: some single-table queries, some 2-table joins, some 3-table joins.
Mix of different filter counts and operator types.

{diversity_hint}

Return ONLY a JSON array of objects, each with a "sql" field:
[
  {{"sql": "SELECT COUNT(*) FROM table1 t1 WHERE t1.col1 = 1 AND t1.col2 > 2000"}},
  {{"sql": "SELECT COUNT(*) FROM table1 t1, table2 t2 WHERE t1.id = t2.fk_id AND t2.col < 5"}},
  ...
]
"""


def validate_sql(sql: str) -> bool:
    """
    Basic validation to reject obviously malformed SQL before sending to DB.
    Returns True if the query looks safe to attempt.
    """
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT COUNT"):
        return False

    bad_patterns = [' OR ', ' IN (', ' LIKE ', ' BETWEEN ', ' NOT ',
                    ' IS NULL', ' IS NOT NULL', ' EXISTS',
                    ' UNION ', ' HAVING ', ' GROUP BY ']
    where_idx = sql_upper.find('WHERE')
    if where_idx > 0:
        where_clause = sql_upper[where_idx:]
        for pattern in bad_patterns:
            if pattern in where_clause:
                return False

    if sql_upper.count('SELECT') > 1:
        return False

    return True


def call_ollama(prompt: str,
                model_name: str = "llama3.2",
                ollama_url: str = "http://localhost:11434",
                temperature: float = 0.8) -> Optional[str]:
    """Call Ollama API to generate a response."""
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 4096,
                }
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"[query_generator] Ollama error: {e}")
        return None


def extract_json_array(text: str) -> Optional[List[Dict]]:
    """Extract a JSON array from LLM output text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def generate_queries_batch(batch_size: int,
                            schema_text: str,
                            stats_text: str,
                            model_name: str = "llama3.2",
                            ollama_url: str = "http://localhost:11434",
                            existing_count: int = 0,
                            max_retries: int = 3) -> List[str]:
    """Generate a batch of SQL queries using Ollama."""
    diversity_hints = [
        "Focus on single-table queries with various filters.",
        "Focus on 2-table join queries with 1-2 predicates.",
        "Focus on 3-table join queries with multiple predicates.",
        "Mix simple and complex queries. Include point lookups (=) and range scans (<, >).",
        "Generate queries with extreme selectivities: some returning very few rows, some returning millions.",
    ]

    hint_idx = (existing_count // batch_size) % len(diversity_hints)
    diversity_hint = f"Hint: {diversity_hints[hint_idx]}"

    prompt = GENERATION_PROMPT.format(
        schema=schema_text,
        stats=stats_text or "No specific stats available.",
        batch_size=batch_size,
        diversity_hint=diversity_hint,
    )

    for attempt in range(max_retries):
        raw_response = call_ollama(prompt, model_name, ollama_url)
        if raw_response is None:
            time.sleep(2)
            continue

        queries_json = extract_json_array(raw_response)
        if queries_json is None:
            print(f"[query_generator] Failed to parse JSON (attempt {attempt + 1})")
            time.sleep(1)
            continue

        sqls = []
        for q in queries_json:
            if isinstance(q, dict) and "sql" in q:
                sqls.append(q["sql"])

        if sqls:
            return sqls

    print(f"[query_generator] All {max_retries} attempts failed for batch")
    return []


def generate_all_queries(total_queries: int,
                          schema_text: str,
                          stats_text: str,
                          batch_size: int = 20,
                          model_name: str = "llama3.2",
                          ollama_url: str = "http://localhost:11434") -> List[str]:
    """Generate the full set of unlabeled SQL queries."""
    all_sqls = []
    num_batches = (total_queries + batch_size - 1) // batch_size

    print(f"[query_generator] Generating {total_queries} queries in {num_batches} batches...")

    for b in range(num_batches):
        remaining = total_queries - len(all_sqls)
        current_batch_size = min(batch_size, remaining)

        if current_batch_size <= 0:
            break

        print(f"[query_generator] Batch {b + 1}/{num_batches} (have {len(all_sqls)}, need {remaining} more)")

        sqls = generate_queries_batch(
            batch_size=current_batch_size,
            schema_text=schema_text,
            stats_text=stats_text,
            model_name=model_name,
            ollama_url=ollama_url,
            existing_count=len(all_sqls),
        )

        all_sqls.extend(sqls)
        print(f"[query_generator] Got {len(sqls)} queries (total: {len(all_sqls)})")
        time.sleep(0.5)

    all_sqls = all_sqls[:total_queries]
    print(f"[query_generator] Generation complete: {len(all_sqls)} queries")
    return all_sqls


def generate_synthetic_queries(num_queries: int, seed: int = 42) -> List[Dict]:
    """
    Generate synthetic queries programmatically (no LLM needed).
    Useful for testing or when Ollama is not available.
    """
    import random
    import numpy as np

    random.seed(seed)
    np_rng = np.random.RandomState(seed)

    table_defs = {
        "title t": {
            "columns": [
                ("t.kind_id", 1, 7),
                ("t.production_year", 1880, 2019),
            ],
        },
        "cast_info ci": {
            "columns": [
                ("ci.person_id", 1, 4061926),
                ("ci.role_id", 1, 11),
            ],
        },
        "movie_info mi": {
            "columns": [
                ("mi.info_type_id", 1, 110),
            ],
        },
        "movie_info_idx mi_idx": {
            "columns": [
                ("mi_idx.info_type_id", 99, 113),
            ],
        },
        "movie_companies mc": {
            "columns": [
                ("mc.company_id", 1, 234997),
                ("mc.company_type_id", 1, 2),
            ],
        },
        "movie_keyword mk": {
            "columns": [
                ("mk.keyword_id", 1, 134170),
            ],
        },
    }

    join_map = {
        "cast_info ci": "t.id=ci.movie_id",
        "movie_info mi": "t.id=mi.movie_id",
        "movie_info_idx mi_idx": "t.id=mi_idx.movie_id",
        "movie_companies mc": "t.id=mc.movie_id",
        "movie_keyword mk": "t.id=mk.movie_id",
    }

    all_tables = list(table_defs.keys())
    other_tables = [t for t in all_tables if t != "title t"]
    operators = ["=", "<", ">"]

    queries = []

    for _ in range(num_queries):
        num_tables = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]

        if num_tables == 1:
            chosen = [random.choice(all_tables)]
        else:
            extra = random.sample(other_tables, min(num_tables - 1, len(other_tables)))
            chosen = ["title t"] + extra

        joins = []
        for t in chosen:
            if t in join_map and "title t" in chosen:
                joins.append(join_map[t])

        num_preds = random.randint(0, min(4, sum(len(table_defs[t]["columns"]) for t in chosen)))
        predicates = []

        available_cols = []
        for t in chosen:
            available_cols.extend(table_defs[t]["columns"])

        if available_cols and num_preds > 0:
            chosen_cols = random.sample(available_cols, min(num_preds, len(available_cols)))
            for col_name, min_val, max_val in chosen_cols:
                op = random.choice(operators)
                val = np_rng.randint(min_val, max_val + 1)
                predicates.append((col_name, op, str(val)))

        queries.append({
            "tables": chosen,
            "joins": joins,
            "predicates": predicates,
            "cardinality": None,
        })

    return queries
