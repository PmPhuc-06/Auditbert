#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import chi2


DEFAULT_PREDICTIONS_DIR = "paper_artifacts"
OUTPUT_JSON_NAME = "mcnemar_results.json"
OUTPUT_MD_NAME = "table_mcnemar.md"
MODEL_ORDER = ["baseline", "phobert", "mfinbert", "auditbert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run McNemar's test from the prediction artifacts generated for Table 3.",
    )
    parser.add_argument(
        "--predictions-dir",
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory containing predictions_<model>.json files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the test.",
    )
    return parser.parse_args()


def load_predictions(predictions_dir: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    y_true: list[int] | None = None
    predictions: dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        path = predictions_dir / f"predictions_{model_name}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        labels = [int(row["label"]) for row in rows]
        y_pred = np.asarray([int(row["predicted_label"]) for row in rows], dtype=int)
        if y_true is None:
            y_true = labels
        elif y_true != labels:
            raise ValueError(f"Prediction file {path} is not aligned with the shared evaluation set.")
        predictions[model_name] = y_pred
    if y_true is None or len(predictions) < 2:
        raise RuntimeError("Need at least two aligned prediction files to run McNemar's test.")
    return y_true, predictions


def run_mcnemar(
    y_true: list[int],
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=int)
    correct_a = pred_a == y_true_arr
    correct_b = pred_b == y_true_arr

    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))
    discordant = n01 + n10

    if discordant == 0:
        return {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "method": "degenerate",
            "interpretation": "Two models make identical decisions.",
        }

    statistic = float((abs(n01 - n10) - 1) ** 2 / discordant)
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "method": "chi2 with continuity correction",
        "interpretation": (
            f"p={p_value:.6f} {'<' if p_value < alpha else '>='} alpha={alpha:.2f} "
            f"-> {'significant' if p_value < alpha else 'not significant'}"
        ),
    }


def write_outputs(payload: dict[str, object], predictions_dir: Path) -> None:
    json_path = predictions_dir / OUTPUT_JSON_NAME
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# McNemar Test",
        "",
        "| Pair | n01 | n10 | Statistic | p-value | Significant |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for pair_name, result in dict(payload.get("pairwise_tests", {})).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    pair_name,
                    str(result["n01"]),
                    str(result["n10"]),
                    f"{float(result['statistic']):.6f}",
                    f"{float(result['p_value']):.6f}",
                    "yes" if bool(result["significant"]) else "no",
                ]
            )
            + " |"
        )
    (predictions_dir / OUTPUT_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    y_true, predictions = load_predictions(predictions_dir)

    pairwise: dict[str, dict[str, object]] = {}
    for model_a, model_b in combinations(predictions.keys(), 2):
        pair_name = f"{model_a} vs {model_b}"
        pairwise[pair_name] = run_mcnemar(y_true, predictions[model_a], predictions[model_b], args.alpha)

    payload = {
        "alpha": args.alpha,
        "predictions_dir": str(predictions_dir.resolve()),
        "n_samples": len(y_true),
        "n_fraud": int(sum(y_true)),
        "n_non_fraud": int(len(y_true) - sum(y_true)),
        "models_tested": list(predictions.keys()),
        "pairwise_tests": pairwise,
    }
    write_outputs(payload, predictions_dir)

    print("=" * 84)
    print("MCNEMAR TEST")
    print("=" * 84)
    print(f"Samples={len(y_true)} | Fraud={sum(y_true)} | Non-fraud={len(y_true) - sum(y_true)}")
    print("-" * 84)
    for pair_name, result in pairwise.items():
        print(
            f"{pair_name:<26} "
            f"n01={int(result['n01']):>3} "
            f"n10={int(result['n10']):>3} "
            f"stat={float(result['statistic']):.6f} "
            f"p={float(result['p_value']):.6f} "
            f"significant={'yes' if bool(result['significant']) else 'no'}"
        )
    print(f"Saved McNemar artifacts to: {predictions_dir.resolve()}")


if __name__ == "__main__":
    main()
