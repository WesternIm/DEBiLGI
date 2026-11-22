# ruff: noqa: E501
"""Export the tag masks: for every (lemma, pos) of the dataset the morphological tags its pymorphy
paradigm can realize (LAST_MESSAGE39 §3: «маска допустимых клеток»). The Colab notebook has no pymorphy;
this file carries the paradigm to it, so the tag head can only choose a cell that exists.

    PYTHONPATH=<repo>/src python scripts/grammarian/export_paradigms.py [DATA_DIR]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import own_stream  # noqa: E402
import ud_to_tree as u  # noqa: E402

ROOT = HERE.parents[1]
SPLITS = ("train", "train_wide", "dev", "heldout_lemma", "heldout_construction", "bench_ud")


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".codex_work/grammarian_data"
    lemmas: dict[tuple[str, str], int] = defaultdict(int)
    observed: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def visit(tree: dict, target: list | None) -> None:
        """Lemmas of every row (a mask is needed for any lemma the model may meet); observed tags of
        TRAINING rows only (the tag vocabulary is a closed class of the model, never read off evaluation)."""
        clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
        flat = [n for c in clauses for n in c["nodes"]]
        for node in flat:
            lemmas[(node["lemma"], node["pos"])] += 1
        for action in target or ():
            if action[0] == "C":
                observed[flat[action[1]]["pos"]][action[2]] += 1

    for name in SPLITS:
        with (data / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                visit(row["tree"], row["target"] if name in {"train", "train_wide"} else None)
    for item in json.loads((ROOT / ".codex_work/grammarian_bench/bench.json").read_text())["items"]:
        visit(item["tree"], None)
    bench_v2 = ROOT / ".codex_work/grammarian_bench/bench_v2.json"
    if bench_v2.exists():
        for item in json.loads(bench_v2.read_text())["items"]:
            visit(item["tree"], None)
    candidates = {
        pos: sorted(tag for tag, n in tags.items() if n >= 1) for pos, tags in observed.items()
    }
    tag_vocab = sorted({tag for tags in candidates.values() for tag in tags})
    index = {tag: i for i, tag in enumerate(tag_vocab)}
    upos = own_stream._UPOS
    paradigm: dict[str, list[int]] = {}
    for (lemma, pos), _ in sorted(lemmas.items()):
        allowed = []
        for tag in candidates.get(pos, ["-"]):
            form = u.inflect(lemma, upos.get(pos, "NOUN"), tag)
            if form is not None and (
                tag != "-" or pos in {"adverb", "particle", "pronoun", "number"}
            ):
                allowed.append(index[tag])
        paradigm[f"{lemma}|{pos}"] = allowed
    payload = {
        "version": 1,
        "tags": tag_vocab,
        "paradigm": paradigm,
        "pos_tags": {pos: [index[t] for t in tags] for pos, tags in candidates.items()},
    }
    (data / "paradigms.json").write_text(json.dumps(payload, ensure_ascii=False))
    empty = sum(1 for v in paradigm.values() if not v)
    print(f"lemmas={len(paradigm)} tags={len(tag_vocab)} empty_paradigms={empty}")


if __name__ == "__main__":
    main()
