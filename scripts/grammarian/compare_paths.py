# ruff: noqa: E501
"""Cross-check of the two paths into the tree format (stage 4.25, task 1).

Path A: UD annotation -> ``ud_to_tree.convert``.   Path B: the same TEXT -> the bot's own readers
(``analyze_structured_utterance`` -> ``parse_proposition(tree=True)`` -> ``thought_to_tree``), as in
``own_stream.build_item``. Sentences accepted by BOTH are compared by a signature of the tree
(role, lemma, lemma of the head, preposition, marker, elided per node; move, negation, mood, tense,
predicate class and the focus lemma per clause). The report lists the agreement and the discrepancy
categories with examples — the places where the converter and the runtime disagree about the format.

    PYTHONPATH=<repo>/src python scripts/grammarian/compare_paths.py [--per-file 3000] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import own_stream  # noqa: E402
import ud_to_tree as u  # noqa: E402

ROOT = HERE.parents[1]
RAW = ROOT / ".codex_work/grammarian_data/raw"
FILES = [
    ("UD_Russian-SynTagRus", "ru_syntagrus-ud-dev"),
    ("UD_Russian-SynTagRus", "ru_syntagrus-ud-test"),
    ("UD_Russian-GSD", "ru_gsd-ud-dev"),
    ("UD_Russian-GSD", "ru_gsd-ud-test"),
    ("UD_Russian-GSD", "ru_gsd-ud-train"),
    ("UD_Russian-Taiga", "ru_taiga-ud-dev"),
    ("UD_Russian-Taiga", "ru_taiga-ud-test"),
    ("UD_Russian-Taiga", "ru_taiga-ud-train-a"),
]


def signature(tree: dict) -> dict:
    nodes = {n["id"]: n for n in tree["nodes"]}
    items = Counter()
    for n in tree["nodes"]:
        head = nodes[n["head"]]["lemma"] if n["head"] is not None else None
        items[
            (
                n["role"],
                n["lemma"],
                head,
                n.get("preposition"),
                n.get("marker"),
                bool(n.get("elided")),
            )
        ] += 1
    clause = tree["clause"]
    focus = nodes[clause["focus"]]["lemma"] if clause["focus"] is not None else None
    features = {
        "move": clause["move"],
        "negated": clause["negated"],
        "mood": clause["mood"],
        "tense": clause["tense"],
        "predicate_class": clause["predicate_class"],
        "focus": focus,
    }
    return {"nodes": items, "clause": features}


def categorize(a: dict, b: dict) -> list[str]:
    """Discrepancy categories between the UD signature ``a`` and the readers' signature ``b``."""
    out = []
    for name, value in a["clause"].items():
        if b["clause"][name] != value:
            out.append(f"clause:{name}")
    only_a, only_b = a["nodes"] - b["nodes"], b["nodes"] - a["nodes"]
    used_b: set = set()
    for key in only_a:
        role, lemma, head, prep, marker, elided = key
        twin = next((k for k in only_b if k[1] == lemma and k not in used_b), None)
        if twin is None:
            out.append(f"node:only_ud:{role}")
            continue
        used_b.add(twin)
        if twin[0] != role:
            out.append(f"node:role:{role}->{twin[0]}")
        elif twin[2] != head:
            out.append(f"node:head:{role}")
        else:
            out.append("node:preposition_or_marker")
    out.extend(f"node:only_readers:{k[0]}" for k in only_b if k not in used_b)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-file", type=int, default=3000)
    parser.add_argument(
        "--out", type=Path, default=ROOT / ".codex_work/grammarian_data/cross_check.json"
    )
    args = parser.parse_args()
    both = same = 0
    only_ud = only_own = neither = 0
    categories: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    for directory, name in FILES:
        taken = 0
        for sentence in u.read_conllu(RAW / directory / f"{name}.conllu"):
            words = sentence.text.split()
            if not 2 <= len(words) <= 9 or re.search(r"[A-Za-z0-9@#:/,;()\"«»]", sentence.text):
                continue
            taken += 1
            if taken > args.per_file:
                break
            try:
                ud = u.convert(sentence)
            except u.Reject:
                ud = None
            try:
                own = own_stream.build_item(sentence.text)
            except u.Reject:
                own = None
            except Exception:  # noqa: BLE001
                own = None
            if ud is None and own is None:
                neither += 1
            elif ud is None:
                only_own += 1
            elif own is None:
                only_ud += 1
            elif ud.tree["kind"] != "clause":
                continue  # multi-clause UD trees have no counterpart in the one-clause readers
            else:
                both += 1
                found = categorize(signature(ud.tree), signature(own.tree))
                if not found:
                    same += 1
                for category in set(found):
                    categories[category] += 1
                    if len(examples[category]) < 6:
                        examples[category].append(
                            {
                                "text": sentence.text,
                                "ud": ud.target[:12],
                                "readers": own.target[:12],
                            }
                        )
    report = {
        "both_accepted": both,
        "identical_trees": same,
        "identical_share": round(same / max(1, both), 3),
        "only_ud": only_ud,
        "only_readers": only_own,
        "neither": neither,
        "discrepancy_categories": dict(categories.most_common()),
        "examples": examples,
        "per_file_limit": args.per_file,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print({k: v for k, v in report.items() if k != "examples"})


if __name__ == "__main__":
    main()
