# ruff: noqa: E501
"""Stage D (4.30): the stand «own» through the REAL adapter (``realize_thought`` with ``DEBILGI_GRAMMARIAN=1``) and without it.

Every stand phrase has a plan (``thought``) and a corpus reference; the same plan is worded twice — by the deterministic realizer
(flag 0) and by ``realize_thought`` with the flag on (the network first, the deterministic surfaces after it, or the fallback). The
words are compared by ``alt_orders.words`` with the reference and its alternative orders. Reported: shares of the network / the
fallback with the reasons, ``exact_2ref`` and the set of words on all phrases and on the phrases the network took, speed.

    PYTHONPATH=<repo>/src .venv/bin/python scripts/grammarian/runtime_stand.py [--no-case] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / ".codex_work/grammarian_bench"))

import alt_orders  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-case", action="store_true", help="the plan fixes no case of its own (a new thought)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bench", type=Path, default=ROOT / ".codex_work/grammarian_bench/bench_v2.json")
    args = parser.parse_args()
    from debilgi.language import grammarian
    from debilgi.language.dialogue_obligations import thought_from_data
    from debilgi.language.productive_nlg import productive_pattern_curriculum, realize_thought

    patterns = productive_pattern_curriculum()
    words = alt_orders.words
    items = [i for i in json.loads(args.bench.read_text())["items"] if i["slice"] == "own" and i["thought"]]
    rows, reasons = [], Counter()
    t_off = t_on = 0.0
    for item in items:
        thought = thought_from_data(item["thought"])
        if thought is None:
            continue
        if args.no_case:
            thought = replace(thought, dependencies=tuple(
                replace(d, argument=replace(d.argument, grammatical_case=None)) for d in thought.dependencies))
        os.environ["DEBILGI_GRAMMARIAN"] = "0"
        t0 = time.perf_counter()
        before = realize_thought(thought, patterns, limit=6)
        t_off += time.perf_counter() - t0
        os.environ["DEBILGI_GRAMMARIAN"] = "1"
        with grammarian.capture() as records:
            t0 = time.perf_counter()
            after = realize_thought(thought, patterns, limit=6)
            t_on += time.perf_counter() - t0
        record = records[-1] if records else None
        # the references are written by people of both genders, the bot speaks of itself in the masculine: a second run with
        # the other self-gender tells a difference of the speaker's gender from a difference of the wording
        os.environ["DEBILGI_SELF_GENDER"] = "feminine"
        alt_gender = realize_thought(thought, patterns, limit=6)
        os.environ.pop("DEBILGI_SELF_GENDER")
        reference, alts = words(item["reference"]), [words(a) for a in item["references_alt"]]
        top_b = words(before[0].text) if before else []
        top_a = words(after[0].text) if after else []
        top_f = words(alt_gender[0].text) if alt_gender else []
        network = bool(after and after[0].source == "grammarian")
        reasons["network" if network else (record.reason if record else "none")] += 1
        rows.append({
            "id": item["id"], "reference": item["reference"], "before": before[0].text if before else None,
            "after": after[0].text if after else None, "network": network,
            "reason": None if network or record is None else record.reason,
            "confidence": None if record is None else record.confidence,
            "before_ok": top_b == reference or top_b in alts, "after_ok": top_a == reference or top_a in alts,
            "before_words": sorted(top_b) == sorted(reference), "after_words": sorted(top_a) == sorted(reference),
            "after_ok_any_gender": top_a == reference or top_a in alts or top_f == reference or top_f in alts,
            "after_words_any_gender": sorted(top_a) == sorted(reference) or sorted(top_f) == sorted(reference),
        })
    n = len(rows)
    taken = [r for r in rows if r["network"]]

    def share(sub: list[dict], key: str) -> float | None:
        return round(sum(r[key] for r in sub) / len(sub), 4) if sub else None

    summary = {
        "n": n, "no_case": args.no_case, "network": len(taken), "fallback": n - len(taken),
        "reasons": dict(reasons.most_common()),
        "exact_2ref_before": share(rows, "before_ok"), "exact_2ref_after": share(rows, "after_ok"),
        "words_before": share(rows, "before_words"), "words_after": share(rows, "after_words"),
        "on_network_phrases": {"n": len(taken), "exact_2ref_before": share(taken, "before_ok"),
                               "exact_2ref_after": share(taken, "after_ok"),
                               "words_before": share(taken, "before_words"), "words_after": share(taken, "after_words")},
        "exact_2ref_after_any_self_gender": share(rows, "after_ok_any_gender"),
        "words_after_any_self_gender": share(rows, "after_words_any_gender"),
        "ms_per_call_flag0": round(1000 * t_off / n, 2), "ms_per_call_flag1": round(1000 * t_on / n, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
