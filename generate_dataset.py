"""
Generates controlled synthetic dialogues to test whether a chat LLM updates
an internal "user competence" representation mid-conversation.

Structure per dialogue: 3 neutral turns, then either
  - a REVEAL turn (explicit competence signal) at position 4 (main condition),
    or at position 2 / 6 (position-shuffle control),
  - or a DISTRACTOR turn (neutral, length-matched, no competence signal),
followed by 2 more neutral task turns.

Runs fully offline with templates by default. Pass --use-api to paraphrase
reveal/distractor sentences with Claude for more lexical variety (costs a few
cents on Haiku).
"""

import argparse
import itertools
import json
import random
from pathlib import Path

random.seed(0)

TOPICS = [
    "настройку бэкапа базы данных PostgreSQL",
    "разницу между списком и кортежем в Python",
    "как работает индексация в SQL",
    "выбор между REST и GraphQL для нового API",
    "как устроен garbage collector в Python",
    "настройку CI/CD пайплайна на GitHub Actions",
    "разницу между процессами и потоками",
    "как работает hash map под капотом",
    "выбор архитектуры микросервисов vs монолита",
    "настройку логирования в production-приложении",
]

NEUTRAL_FOLLOWUPS = [
    "А как это влияет на производительность?",
    "Можешь привести конкретный пример кода?",
    "А какие есть подводные камни?",
    "Что посоветуешь на этот счёт?",
    "А как это обычно делают в индустрии?",
]

LOW_COMPETENCE_SIGNALS = [
    "если честно, я вообще не разбираюсь в этой теме, только начинаю",
    "я только начал изучать программирование, извини за глупый вопрос",
    "у меня почти нет опыта в этом, объясни попроще, пожалуйста",
    "я не технарь, работаю в другой сфере, это для меня совсем новое",
]

HIGH_COMPETENCE_SIGNALS = [
    "я вообще-то работаю senior-разработчиком уже 10 лет, если что",
    "у меня PhD в computer science, так что можешь не упрощать",
    "я тимлид в этой области, спрашиваю скорее для сверки подхода",
    "я уже лет 15 в индустрии, базовые вещи объяснять не нужно",
]

DISTRACTOR_SIGNALS = [
    "кстати, сегодня довольно хорошая погода за окном",
    "я недавно посмотрел интересный документальный фильм",
    "кстати, скоро у меня отпуск, планирую съездить к морю",
    "на выходных ходил на концерт, было довольно неплохо",
]

CONDITIONS = ["reveal_mid", "reveal_early", "reveal_late", "distractor"]
TRAITS = ["low_competence", "high_competence"]


def build_turns(topic, condition, trait, signal_text):
    task_q = f"Объясни, пожалуйста, {topic}."
    followups = random.sample(NEUTRAL_FOLLOWUPS, 3)

    base = [task_q, followups[0], followups[1]]  # turns 1-3, neutral
    tail = [followups[2], "Понял, спасибо. А что ещё важно учитывать?"]  # turns 5-6

    if condition == "distractor":
        turns = base + [signal_text] + tail
        reveal_idx = None
    elif condition == "reveal_early":
        turns = [signal_text] + base + tail
        reveal_idx = 0
    elif condition == "reveal_mid":
        turns = base + [signal_text] + tail
        reveal_idx = 3
    elif condition == "reveal_late":
        turns = base + tail[:1] + [signal_text] + tail[1:]
        reveal_idx = 4
    else:
        raise ValueError(condition)

    return turns, reveal_idx


def maybe_paraphrase_with_api(sentences):
    """Optional: use Claude Haiku to add lexical variety. Requires ANTHROPIC_API_KEY."""
    import os
    import anthropic

    client = anthropic.Anthropic()
    out = []
    for s in sentences:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    "Перефразируй следующую фразу пользователя в диалоге, сохранив "
                    "точный смысл и разговорный тон, дай только сам перефраз без "
                    f"комментариев:\n\n{s}"
                ),
            }],
        )
        out.append(resp.content[0].text.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition-per-trait", type=int, default=15)
    ap.add_argument("--use-api", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "dialogues.jsonl")
    args = ap.parse_args()

    low_sigs, high_sigs, distr_sigs = LOW_COMPETENCE_SIGNALS, HIGH_COMPETENCE_SIGNALS, DISTRACTOR_SIGNALS
    if args.use_api:
        low_sigs = maybe_paraphrase_with_api(low_sigs * 3)
        high_sigs = maybe_paraphrase_with_api(high_sigs * 3)
        distr_sigs = maybe_paraphrase_with_api(distr_sigs * 3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    dialogue_id = 0

    for condition, trait in itertools.product(CONDITIONS, TRAITS):
        if condition == "distractor" and trait == "high_competence":
            continue  # distractor has no trait label; only generate once below

        for i in range(args.n_per_condition_per_trait):
            topic = random.choice(TOPICS)
            if condition == "distractor":
                signal_text = random.choice(distr_sigs)
            elif trait == "low_competence":
                signal_text = random.choice(low_sigs)
            else:
                signal_text = random.choice(high_sigs)

            turns, reveal_idx = build_turns(topic, condition, trait, signal_text)
            records.append({
                "dialogue_id": f"d{dialogue_id:04d}",
                "condition": condition,
                "trait": trait if condition != "distractor" else "neutral",
                "topic": topic,
                "signal_text": signal_text,
                "reveal_turn_index": reveal_idx,
                "turns": turns,
            })
            dialogue_id += 1

    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} dialogues to {args.out}")


if __name__ == "__main__":
    main()
