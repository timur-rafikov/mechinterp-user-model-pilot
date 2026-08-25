"""
Pilot behavioral-adaptation check: rolls out reveal_mid (low/high competence)
and distractor dialogues, then measures whether the model's replies AFTER the
reveal turn actually differ in complexity/length/hedging — i.e. whether the
competence signal has any observable behavioral effect at all before
investing in the full causal-patching sweep.

Uses simple, Russian-appropriate metrics rather than English-calibrated
readability formulas (Flesch-Kincaid assumes English syllable structure and
is only mildly informative here — kept as an auxiliary signal).

Usage:
    python behavior_check.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 8
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from dialogue_utils import rollout_dialogue

HEDGES = [
    "возможно", "наверное", "скорее всего", "советую проконсультироваться",
    "обратитесь к специалисту", "если не ошибаюсь", "может быть", "стоит уточнить",
    "имейте в виду, что это упрощённо", "если коротко",
]


def metrics_for_reply(text):
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "n_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "n_sentences": max(len(sentences), 1),
        "n_hedges": sum(text.lower().count(h) for h in HEDGES),
    }


def load_dialogues(path, limit_per_key=None):
    by_key = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] not in ("reveal_mid", "distractor"):
                continue
            key = (r["condition"], r["trait"])
            by_key.setdefault(key, []).append(r)
    if limit_per_key:
        by_key = {k: v[:limit_per_key] for k, v in by_key.items()}
    return [r for v in by_key.values() for r in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dialogues", type=Path, default=Path(__file__).parent / "data" / "dialogues.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
    dtype = getattr(torch, args.dtype)
    model = HookedTransformer.from_pretrained(args.model, device=args.device, dtype=dtype)

    dialogues = load_dialogues(args.dialogues, args.limit)
    print(f"Rolling out {len(dialogues)} dialogues ...")

    rows = []
    for d_i, d in enumerate(dialogues):
        records = rollout_dialogue(model, d["turns"], max_new_tokens=args.max_new_tokens, layers=[0])
        post_reveal_start = d["reveal_turn_index"] + 1 if d["reveal_turn_index"] is not None else 3
        for rec in records:
            if rec.turn_index < post_reveal_start:
                continue
            m = metrics_for_reply(rec.assistant_reply)
            m.update({"dialogue_id": d["dialogue_id"], "condition": d["condition"], "trait": d["trait"], "turn_index": rec.turn_index})
            rows.append(m)
        if (d_i + 1) % 10 == 0:
            print(f"  {d_i + 1}/{len(dialogues)} dialogues done")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "behavior_raw.csv", index=False)

    summary = df.groupby(["condition", "trait"])[["n_words", "avg_word_len", "n_sentences", "n_hedges"]].mean()
    summary.to_csv(args.out_dir / "behavior_summary.csv")
    print("\nPost-reveal reply metrics by condition/trait:")
    print(summary.round(2).to_string())
    print(f"\nSaved raw data to {args.out_dir / 'behavior_raw.csv'}")
    print(f"Saved summary to {args.out_dir / 'behavior_summary.csv'}")


if __name__ == "__main__":
    main()
