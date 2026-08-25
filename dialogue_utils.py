"""
Shared helpers for rolling a synthetic dialogue through a HookedTransformer
one turn at a time, caching residual-stream activations at each turn boundary
and generating a short assistant reply to extend the context for the next turn.

Both extract_and_probe.py and behavior_check.py reuse this rollout so the
probing data and the behavioral generations come from the exact same
conversational trajectories.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class TurnRecord:
    turn_index: int
    user_text: str
    assistant_reply: str
    resid_by_layer: dict = field(default_factory=dict)  # layer -> torch.Tensor [d_model]


def format_prefix(turns_so_far, replies_so_far):
    s = ""
    for u, a in zip(turns_so_far[:-1], replies_so_far):
        s += f"Пользователь: {u}\nАссистент: {a}\n"
    s += f"Пользователь: {turns_so_far[-1]}\nАссистент:"
    return s


@torch.no_grad()
def rollout_dialogue(model, turns, max_new_tokens=40, layers=None):
    """
    Runs the dialogue turn by turn. At each turn boundary (right after the
    "Ассистент:" cue, before any reply tokens), caches resid_post activations
    for the requested layers, then generates a short reply to extend context.

    Returns: list[TurnRecord]
    """
    if layers is None:
        n_layers = model.cfg.n_layers
        layers = sorted(set([n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]))

    names_filter = lambda name: name.endswith("resid_post")

    replies = []
    records = []

    for i in range(len(turns)):
        prefix = format_prefix(turns[: i + 1], replies)
        tokens = model.to_tokens(prefix)

        _, cache = model.run_with_cache(tokens, names_filter=names_filter)
        last_pos = tokens.shape[1] - 1

        resid_by_layer = {}
        for layer in layers:
            act = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos, :].detach().float().cpu()
            resid_by_layer[layer] = act

        gen_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )
        reply_tokens = gen_tokens[0, tokens.shape[1]:]
        reply_text = model.to_string(reply_tokens)
        reply_text = reply_text.split("Пользователь:")[0].strip()

        records.append(TurnRecord(
            turn_index=i,
            user_text=turns[i],
            assistant_reply=reply_text,
            resid_by_layer=resid_by_layer,
        ))
        replies.append(reply_text)

    return records
