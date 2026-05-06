import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.format_converter import parse_sql_to_mscn

EPSILON = 1e-8


def _load_sql_queries(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sqls: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                sqls.append(item)
            elif isinstance(item, dict):
                if "sql" in item and isinstance(item["sql"], str):
                    sqls.append(item["sql"])
                elif all(k in item for k in ("tables", "joins", "predicates")):
                    # If structured without original SQL, skip for KL over SQL-derived features.
                    continue
    return sqls


def _extract_feature_vectors(sqls: List[str]) -> Dict[str, List[int]]:
    features = {
        "tables": [],
        "joins": [],
        "predicates": [],
    }

    for sql in sqls:
        parsed = parse_sql_to_mscn(sql)
        if not parsed:
            continue
        features["tables"].append(len(parsed.get("tables", [])))
        features["joins"].append(len(parsed.get("joins", [])))
        features["predicates"].append(len(parsed.get("predicates", [])))

    return features


def _pmf(values: List[int], bins: List[int]) -> np.ndarray:
    if not bins:
        return np.array([], dtype=np.float64)

    counts = np.zeros(len(bins), dtype=np.float64)
    index = {b: i for i, b in enumerate(bins)}
    for v in values:
        if v in index:
            counts[index[v]] += 1.0

    counts += EPSILON
    counts /= counts.sum()
    return counts


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # D_KL(P || Q)
    return float(np.sum(p * np.log(p / q)))


def _latest_generated_query_file(generated_dir: Path) -> Optional[Path]:
    candidates = sorted(generated_dir.glob("queries_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def build_kl_convergence(
    reference_sqls: List[str],
    generated_sqls: List[str],
    step: int,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    ref_features = _extract_feature_vectors(reference_sqls)
    gen_features = _extract_feature_vectors(generated_sqls)

    ref_count = len(ref_features["joins"])
    gen_count = len(gen_features["joins"])

    bins = {
        name: sorted(set(ref_features[name]) | set(gen_features[name]))
        for name in ("tables", "joins", "predicates")
    }

    ref_p = {name: _pmf(ref_features[name], bins[name]) for name in bins}

    rows: List[Dict[str, float]] = []
    checkpoints = list(range(step, gen_count + 1, step))
    if not checkpoints or checkpoints[-1] != gen_count:
        checkpoints.append(gen_count)

    for t in checkpoints:
        r: Dict[str, float] = {"checkpoint": float(t)}
        kl_values = []

        for name in ("tables", "joins", "predicates"):
            q_t = _pmf(gen_features[name][:t], bins[name])
            if ref_p[name].size == 0 or q_t.size == 0:
                kl = 0.0
            else:
                kl = _kl_divergence(ref_p[name], q_t)
            r[f"kl_{name}"] = kl
            kl_values.append(kl)

        r["kl_mean"] = float(np.mean(kl_values)) if kl_values else 0.0
        rows.append(r)

    meta = {
        "reference_queries_parsed": ref_count,
        "generated_queries_parsed": gen_count,
    }
    return rows, meta


def save_outputs(rows: List[Dict[str, float]], out_dir: Path, prefix: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_kl_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["checkpoint", "kl_tables", "kl_joins", "kl_predicates", "kl_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = [int(r["checkpoint"]) for r in rows]
    ax.plot(x, [r["kl_tables"] for r in rows], label="KL Tables", linewidth=2)
    ax.plot(x, [r["kl_joins"] for r in rows], label="KL Joins", linewidth=2)
    ax.plot(x, [r["kl_predicates"] for r in rows], label="KL Predicates", linewidth=2)
    ax.plot(x, [r["kl_mean"] for r in rows], label="KL Mean", linewidth=2.5, linestyle="--")

    ax.set_xlabel("Generated Queries Seen")
    ax.set_ylabel("KL Divergence (lower is better)")
    ax.set_title("KL Convergence of Generated Workload")
    ax.grid(True, alpha=0.3)
    ax.legend()

    png_path = out_dir / f"{prefix}_kl_convergence.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create KL convergence plot for generated SQL workload")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference workload JSON (real workload)")
    parser.add_argument("--generated", type=str, default="", help="Path to generated workload JSON (default: latest generated_queries/queries_*.json)")
    parser.add_argument("--generated-dir", type=str, default="generated_queries", help="Generated queries folder")
    parser.add_argument("--step", type=int, default=100, help="Checkpoint interval (default: 100)")
    parser.add_argument("--out-dir", type=str, default="generated_queries", help="Output directory")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    if args.generated:
        gen_path = Path(args.generated)
    else:
        latest = _latest_generated_query_file(Path(args.generated_dir))
        if latest is None:
            raise FileNotFoundError("No generated queries file found.")
        gen_path = latest

    if not gen_path.exists():
        raise FileNotFoundError(f"Generated file not found: {gen_path}")

    ref_sqls = _load_sql_queries(ref_path)
    gen_sqls = _load_sql_queries(gen_path)
    if not ref_sqls:
        raise RuntimeError("Reference file contains no SQL strings.")
    if not gen_sqls:
        raise RuntimeError("Generated file contains no SQL strings.")

    rows, meta = build_kl_convergence(ref_sqls, gen_sqls, step=max(args.step, 1))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = f"kl_{gen_path.stem}_vs_{ref_path.stem}_{timestamp}"
    csv_path, png_path = save_outputs(rows, Path(args.out_dir), prefix)

    print(f"Reference file: {ref_path}")
    print(f"Generated file: {gen_path}")
    print(f"Parsed reference queries: {meta['reference_queries_parsed']}")
    print(f"Parsed generated queries: {meta['generated_queries_parsed']}")
    print(f"Saved KL convergence CSV: {csv_path}")
    print(f"Saved KL convergence plot: {png_path}")


if __name__ == "__main__":
    main()
