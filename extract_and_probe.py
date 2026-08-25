"""
Pilot probing harness: rolls each dialogue through a HookedTransformer,
extracts residual-stream activations at every turn boundary, then trains a
linear probe (low- vs high-competence) per (condition, turn, layer) to trace
how decodable the competence signal is before/after the reveal turn.

Usage:
    python extract_and_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 8

Output:
    results/decodability.csv
    results/decodability_curve.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from dialogue_utils import rollout_dialogue

REVEAL_CONDITIONS = ["reveal_mid", "reveal_early", "reveal_late"]


def load_dialogues(path, conditions, limit_per_condition_trait=None):
    by_key = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] not in conditions:
                continue
            key = (r["condition"], r["trait"])
            by_key.setdefault(key, []).append(r)
    if limit_per_condition_trait:
        by_key = {k: v[:limit_per_condition_trait] for k, v in by_key.items()}
    return [r for v in by_key.values() for r in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dialogues", type=Path, default=Path(__file__).parent / "data" / "dialogues.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--limit", type=int, default=None, help="dialogues per (condition, trait) — use ~8 for a smoke test")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--no-processing", action="store_true", help="use from_pretrained_no_processing — lower peak memory, needed for larger models on limited VRAM")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ({args.dtype}, no_processing={args.no_processing}) ...")
    dtype = getattr(torch, args.dtype)
    if args.no_processing:
        model = HookedTransformer.from_pretrained_no_processing(args.model, device=args.device, dtype=dtype)
    else:
        model = HookedTransformer.from_pretrained(args.model, device=args.device, dtype=dtype)

    dialogues = load_dialogues(args.dialogues, REVEAL_CONDITIONS, args.limit)
    print(f"Rolling out {len(dialogues)} dialogues ...")

    n_layers = model.cfg.n_layers
    layers = sorted(set([n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]))

    # rows: condition, trait, turn_index, layer -> activation vector
    rows = []
    for d_i, d in enumerate(dialogues):
        records = rollout_dialogue(model, d["turns"], max_new_tokens=args.max_new_tokens, layers=layers)
        for rec in records:
            for layer, act in rec.resid_by_layer.items():
                rows.append({
                    "dialogue_id": d["dialogue_id"],
                    "condition": d["condition"],
                    "trait": d["trait"],
                    "turn_index": rec.turn_index,
                    "layer": layer,
                    "activation": act.numpy(),
                })
        if (d_i + 1) % 10 == 0:
            print(f"  {d_i + 1}/{len(dialogues)} dialogues done")

    df = pd.DataFrame(rows)

    results = []
    for condition in REVEAL_CONDITIONS:
        for turn_index in sorted(df["turn_index"].unique()):
            for layer in layers:
                sub = df[(df.condition == condition) & (df.turn_index == turn_index) & (df.layer == layer)]
                if sub["trait"].nunique() < 2 or len(sub) < 6:
                    continue
                X = np.stack(sub["activation"].values)
                y = (sub["trait"] == "high_competence").astype(int).values
                clf = LogisticRegression(max_iter=1000)
                n_folds = min(5, min(np.bincount(y)))
                if n_folds < 2:
                    continue
                scores = cross_val_score(clf, X, y, cv=n_folds)
                results.append({
                    "condition": condition,
                    "turn_index": turn_index,
                    "layer": layer,
                    "accuracy": scores.mean(),
                    "n": len(sub),
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out_dir / "decodability.csv", index=False)
    print(f"Saved {args.out_dir / 'decodability.csv'}")

    # best-layer-per-turn curve, one line per condition
    best = res_df.loc[res_df.groupby(["condition", "turn_index"])["accuracy"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in REVEAL_CONDITIONS:
        sub = best[best.condition == condition].sort_values("turn_index")
        ax.plot(sub.turn_index + 1, sub.accuracy, marker="o", label=condition)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Best-layer probe accuracy")
    ax.set_title("Competence decodability across dialogue turns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "decodability_curve.png", dpi=150)
    print(f"Saved {args.out_dir / 'decodability_curve.png'}")


if __name__ == "__main__":
    main()
