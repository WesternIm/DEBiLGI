# ruff: noqa: E501
"""Local evaluation of a Colab export of the grammarian (stages 4.27 → 4.29): fidelity + every gate, CPU only.

Stage 4.29: the MAIN measures are (a) the words of the output are the reference's words (forms + closed class; the order is free) and
(b) the **plan fidelity** (``plan_fidelity.py``): the output says nothing the plan did not say — measurable on 100 % of the rows,
no reader needed. ``exact_2ref`` against one corpus reference stays as a reference figure. The dash is not a word (``alt_orders.words``).
The stand «own» runs in two honest modes: the case as the reader wrote it, and NO case (what a new thought has).
A v2 export is re-evaluated with the same measure: ``--notebook`` is the notebook (v2 snapshot) that trained it, ``--data`` its data.

The Colab notebook cannot call the bot, so what needs the bot is measured here, from the export files
(``grammarian_v2.npz``, ``featurizer.json``; the data dir is the one the notebook trained on):

* the network's output is decoded (the notebook's own model and masks: the notebook source is executed in
  ``GRAMMARIAN_EVAL_ONLY`` mode), turned into words (prepositions put back, ``ud_to_tree.realize``) and compared
  by ONE function, ``words()``, with the reference and the alternative references (other orders the readers read
  into the same tree) — and, on the same phrases, with the current ``realize_thought`` (``baseline_v2.json``);
* **fidelity** (the main metric): the realized text is read back by the bot's own readers
  (``own_stream.build_item``); it is faithful when the reader's tree has the signature of the plan AND the word forms
  are the reference's. Only phrases whose reference the readers read into the plan are measured (a plan with
  roles the readers never produce — UD ``oblique``/``particle`` ... — cannot be re-read even when it is realized perfectly);
  «forms are right» = the bag of surface words equals the reference's (order free; a case-syncretic form is not an error);
* the refusal threshold is calibrated on fidelity (dev), then checked on the stand;
* the seen→unseen drop is reweighted to the length of the held-out sentences.

    # a venv with torch (CPU) + the project's pure-python deps on the path (the project .venv has no torch):
    PYTHONPATH=<repo>/src /tmp/debilgi-427/venv/bin/python scripts/grammarian/evaluate_export.py \
        --data <data dir> --out <export dir> [--fidelity-limit 1000]

Thresholds of the gates are those of LAST_MESSAGE39 §5 (not changed): tag 0.97, drop 3 pp, held-out construction 0.90,
challenge better than the deterministic ``realize_thought``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
NOTEBOOK = ROOT / ".codex_work/grammarian_colab/train_grammarian.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="export dir (grammarian_v3.npz, featurizer.json)")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK, help="the training notebook (.py) the export belongs to")
    parser.add_argument("--baseline-nocase", type=Path, default=None, help="baseline_v2_nocase.json (realize_thought without the case)")
    parser.add_argument("--fidelity-limit", type=int, default=1000, help="rows per split re-read by the readers")
    parser.add_argument("--extra-site", default=str(ROOT / ".venv/lib/python3.13/site-packages"))
    parser.add_argument("--result", type=Path, default=None)
    args = parser.parse_args()
    if args.extra_site and Path(args.extra_site).exists():
        sys.path.append(args.extra_site)  # pymorphy & co from the project venv, AFTER the venv's own numpy/torch
    os.environ.update(GRAMMARIAN_EVAL_ONLY="1", GRAMMARIAN_DATA=str(args.data), GRAMMARIAN_OUT=str(args.out))
    # the model is rebuilt with the configuration the export was trained with (a local experiment may use a smaller one)
    os.environ["GRAMMARIAN_CFG"] = json.dumps(json.loads((args.out / "featurizer.json").read_text()).get("cfg") or {})

    started = time.time()
    ns: dict = {"__name__": "__main__"}
    exec(compile(args.notebook.read_text(encoding="utf-8"), str(args.notebook), "exec"), ns)  # noqa: S102 - loads data, model, weights, action-level metrics
    meta = json.loads((args.out / "featurizer.json").read_text())
    same_fz = json.loads(json.dumps(ns["FZ"], ensure_ascii=False)) == meta["fz"]
    print("featurizer rebuilt from the data == featurizer.json:", same_fz)
    if not same_fz:
        print("WARNING: the data dir differs from the one the export was trained on; the numbers below are not comparable")

    import alt_orders
    import compare_paths
    import own_stream
    import plan_fidelity
    import ud_to_tree as u

    words = alt_orders.words
    decode, collate, model = ns["decode"], ns["collate"], ns["model"]
    tags, func_words, marks = ns["TAGS"], ns["FUNC_WORDS"], ns["PUNCT_MARKS"]
    threshold = meta.get("threshold")

    def run_decode(examples: list, no_case: bool = False) -> list:
        rows = []
        for i in range(0, len(examples), 64):
            chunk = examples[i : i + 64]
            out = decode(model, collate(chunk, with_targets=False, no_case=no_case))
            steps, conf = out[0], out[1]  # v3: the refusal confidence (tags + closed-class choice); v2: the minimum over every step
            conf_all = out[2] if len(out) > 2 else conf
            rows.extend(zip(chunk, steps, conf, conf_all, strict=True))
        return rows

    def surface(node: dict, tag: str) -> str | None:
        """The word of a node in a tag (the realizer of the data path): lets the plan-fidelity check see case syncretism."""
        return u.inflect(node["lemma"], u.UPOS_OF_POS.get(node["pos"], "NOUN"), tag, bool(node.get("preposition")))

    def to_target(steps: list) -> list:
        out: list = []
        for kind, idx, tag in steps:
            if kind == "N":
                out.append(["C", idx, tags[tag]])
            elif kind == "F":
                phrase_kind, _, word = func_words[idx].partition("|")
                out.append(["F", word, f"{phrase_kind}:placed"])
            elif kind == "P":
                out.append(["P", marks[idx]])
        return out

    def realized(tree: dict, steps: list) -> tuple[list, str]:
        target = to_target(steps)
        return target, u.target_text(target, tree)

    # ------------------------------------------------------------------ one record per phrase
    def records(examples: list, key_tree: str, ref_key: str, alt_key: str, limit_fidelity: int, no_case: bool = False) -> list[dict]:
        out = []
        for n, (e, steps, conf, conf_all) in enumerate(run_decode(examples, no_case)):
            item = e["item"]
            tree = item[key_tree]
            target, text = realized(tree, steps)
            explain: dict = {}
            violations = plan_fidelity.check(tree, target, item.get("target"), explain, fixed_case=not no_case, surface=surface)
            w = words(text)
            ref = words(item[ref_key])
            alts = [words(a) for a in item.get(alt_key) or []]
            rec = {
                "id": item.get("id"),
                "slice": item.get("slice"),
                "n_nodes": e["N"],
                "conf": conf,
                "conf_all": conf_all,
                "plan_violations": sorted(violations),
                "plan_explain": {k: v[:2] for k, v in explain.items()},
                "plan_clean": not violations,
                "plan_clean_without_cc": not (violations - {"cc"}),
                "exact": w == ref,
                "exact_2ref": w == ref or w in alts,
                "has_alt": bool(alts),
                "same_bag": sorted(w) == sorted(ref),  # right words and forms, any order
                "main_ok": sorted(w) == sorted(ref) and not violations,  # (a) and (b) together
                "text": text,
                "reference": item[ref_key],
                "heldout_lemma": item.get("heldout_lemma"),
            }
            if n < limit_fidelity and tree["kind"] == "clause":
                read_ref = _read(item[ref_key])
                rec["measurable"] = read_ref is not None and compare_paths.signature(read_ref.tree) == compare_paths.signature(tree)
                if rec["measurable"]:
                    rec["faithful_signature"], rec["faithful"] = _fidelity(tree, text, item[ref_key])
            out.append(rec)
        return out

    def _read(text: str):
        try:
            return own_stream.build_item(text)
        except Exception:  # noqa: BLE001 - an unreadable phrase is a counted outcome, not a crash
            return None

    def _fidelity(tree: dict, text: str, reference: str) -> tuple[bool, bool]:
        """(the reader's tree has the plan's signature, ... and every word form is the reference's form).

        Forms are compared as the bag of surface words with the reference (the order is free: a free word order
        is not an error): the tag ids of the reader and of the plan may differ for one and the same surface
        (case syncretism: «человека» is a genitive and an accusative), which is not a wrong form."""
        read = _read(text)
        if read is None or compare_paths.signature(read.tree) != compare_paths.signature(tree):
            return False, False
        return True, sorted(words(text)) == sorted(words(reference))

    def mean(rows: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows if key in r]
        return round(sum(values) / len(values), 4) if values else None

    def summary(rows: list[dict]) -> dict:
        measurable = [r for r in rows if r.get("measurable")]
        failures = [r for r in rows if not r["exact_2ref"]]
        return {
            "n": len(rows),
            "same_bag_of_words": mean(rows, "same_bag"),
            "plan_fidelity": mean(rows, "plan_clean"),
            "plan_fidelity_without_closed_class": mean(rows, "plan_clean_without_cc"),
            "words_and_plan_fidelity": mean(rows, "main_ok"),
            "plan_violations_by_kind": dict(collections.Counter(k for r in rows for k in r["plan_violations"])),
            "failures_that_are_order_only": round(sum(r["same_bag"] for r in failures) / len(failures), 4) if failures else None,
            "exact_words": mean(rows, "exact"),
            "exact_2ref_words": mean(rows, "exact_2ref"),
            "n_measurable_fidelity": len(measurable),
            "fidelity": mean(measurable, "faithful"),
            "fidelity_signature_only": mean(measurable, "faithful_signature"),
        }

    result: dict = {"featurizer_matches_data": same_fz, "export_epoch": meta.get("epoch"), "threshold_in_export": threshold}
    limit = args.fidelity_limit
    by_split: dict[str, list[dict]] = {}
    for name in ("dev", "heldout_lemma", "heldout_construction"):
        examples = [e for e in ns["ENC"][name] if "steps" in e]
        by_split[name] = records(examples, "tree", "text", "alt_texts", limit)
        by_split[name + ":no_case"] = records(examples, "tree", "text", "alt_texts", 0, no_case=True)
    own = [e for e in ns["ENC"]["bench"] if e["item"]["slice"] == "own" and "steps" in e]
    by_split["bench:own"] = records(own, "tree", "reference", "references_alt", 10**9)
    own_runtime = [e for e in ns["ENC"]["bench_runtime"] if "steps" in e]
    by_split["bench:own_runtime"] = records(own_runtime, "tree_runtime", "reference", "references_alt", 10**9)
    by_split["bench:own_runtime:no_case"] = records(own_runtime, "tree_runtime", "reference", "references_alt", 0, no_case=True)
    for sl in ("wh", "comparison", "nested"):
        examples = [e for e in ns["ENC"]["bench"] if e["item"]["slice"] == sl and "steps" in e]
        by_split["bench:" + sl] = records(examples, "tree", "reference", "references_alt", 0)
    modal = [e for e in ns["ENC"]["heldout_construction"] if "modal_be_predicate" in e["item"]["constructions"] and "steps" in e]
    by_split["heldout_construction:modal_be_predicate"] = records(modal, "tree", "text", "alt_texts", limit)
    compar = [e for e in ns["ENC"]["heldout_construction"] if "comparison" in e["item"]["constructions"] and "steps" in e]
    by_split["heldout_construction:comparison"] = records(compar, "tree", "text", "alt_texts", limit)
    result["splits"] = {k: summary(v) for k, v in by_split.items()}

    # ------------------------------------------------------------------ the deterministic realize_thought on the SAME phrases
    bench_items = {item["id"]: item for item in ns["BENCH"]}

    def baseline_rows(path: Path) -> dict[str, dict]:
        """Rows of ``baseline_v2*.json`` re-scored with the CURRENT ``words()`` (the files carry the realized texts)."""
        out = {}
        for r in json.loads(path.read_text())["rows"]:
            item = bench_items.get(r["id"])
            if item is None:
                continue
            ref, alts = words(item["reference"]), [words(a) for a in item.get("references_alt") or []]
            top = words(r["realized"][0]) if r["realized"] else []
            out[r["id"]] = {"empty": not r["realized"], "exact": top == ref, "exact_2ref": top == ref or top in alts, "same_bag": bool(r["realized"]) and sorted(top) == sorted(ref)}
        return out

    def challenge(mine: list[dict], base_rows: dict[str, dict]) -> dict:
        net = [r for r in mine if r["id"] in base_rows]
        base = [base_rows[r["id"]] for r in net]
        rate = lambda rows, key: round(sum(r[key] for r in rows) / len(rows), 4)  # noqa: E731
        return {
            "n_same_phrases": len(net),
            "network_exact_words": rate(net, "exact"),
            "network_exact_2ref_words": rate(net, "exact_2ref"),
            "network_same_bag": rate(net, "same_bag"),
            "network_plan_fidelity": rate(net, "plan_clean"),
            "realize_thought_exact_words": rate(base, "exact"),
            "realize_thought_exact_2ref_words": rate(base, "exact_2ref"),
            "realize_thought_same_bag": rate(base, "same_bag"),
            "realize_thought_empty": rate(base, "empty"),
            "network_wins": sum(1 for n_, b_ in zip(net, base, strict=True) if n_["exact_2ref"] and not b_["exact_2ref"]),
            "network_loses": sum(1 for n_, b_ in zip(net, base, strict=True) if b_["exact_2ref"] and not n_["exact_2ref"]),
        }

    result["challenge_own"] = {}
    baseline_path = args.data / "baseline_v2.json"
    nocase_path = args.baseline_nocase or (ROOT / ".codex_work/grammarian_bench/baseline_v2_nocase.json")
    if baseline_path.exists():
        result["challenge_own"]["case_as_read_by_the_reader"] = challenge(by_split["bench:own_runtime"], baseline_rows(baseline_path))
        result["challenge_own"]["case_as_read_by_the_reader"]["network_exact_2ref_words_v2_input"] = round(
            sum(r["exact_2ref"] for r in by_split["bench:own"] if r["id"] in {x["id"] for x in by_split["bench:own_runtime"]}) / max(1, len(by_split["bench:own_runtime"])), 4)
    if nocase_path.exists():
        result["challenge_own"]["no_case_honest"] = challenge(by_split["bench:own_runtime:no_case"], baseline_rows(nocase_path))

    # ------------------------------------------------------------------ seen -> unseen, reweighted to the held-out length
    def length_table(rows: list[dict], key: str) -> dict[int, tuple[int, float]]:
        cells: dict[int, list[float]] = collections.defaultdict(list)
        for r in rows:
            cells[ns["len_bin"](r["n_nodes"])].append(r[key])
        return {k: (len(v), sum(v) / len(v)) for k, v in cells.items()}

    for key in ("exact", "exact_2ref", "main_ok"):
        seen, unseen = length_table(by_split["dev"], key), length_table(by_split["heldout_lemma"], key)
        total = sum(n for n, _ in unseen.values())
        weight = sum(n / total for k, (n, _) in unseen.items() if k in seen)
        reweighted = sum((n / total) * seen[k][1] for k, (n, _) in unseen.items() if k in seen) / weight
        plain_unseen = sum(r[key] for r in by_split["heldout_lemma"]) / len(by_split["heldout_lemma"])
        plain_seen = sum(r[key] for r in by_split["dev"]) / len(by_split["dev"])
        result.setdefault("drop_seen_unseen", {})[key] = {
            "dev_plain": round(plain_seen, 4),
            "dev_reweighted_to_heldout_length": round(reweighted, 4),
            "heldout_lemma": round(plain_unseen, 4),
            "drop_pp_reweighted": round(100 * (reweighted - plain_unseen), 1),
            "drop_pp_plain": round(100 * (plain_seen - plain_unseen), 1),
        }

    # ------------------------------------------------------------------ refusal threshold (dev), checked on the stand
    def auc(pairs: list[tuple[float, bool]]) -> float | None:
        positive, negative = [c for c, ok in pairs if ok], [c for c, ok in pairs if not ok]
        if not positive or not negative:
            return None
        return round(sum((p > q) + 0.5 * (p == q) for p in positive for q in negative) / (len(positive) * len(negative)), 3)

    def calibrate(rows: list[dict], level: float, label: str, conf_key: str = "conf") -> dict:
        """The highest coverage with the share of ``label`` >= ``level`` among the accepted (confidence >= threshold)."""
        pairs = sorted(((r[conf_key], r[label]) for r in rows if label in r), reverse=True)
        best, hits = {"threshold": 1.01, "coverage": 0.0, "n_accepted": 0}, 0
        for k, (conf, ok) in enumerate(pairs, 1):
            hits += ok
            if hits / k >= level and k / len(pairs) >= best["coverage"]:
                best = {"threshold": round(conf, 4), "coverage": round(k / len(pairs), 4), "n_accepted": k}
        best["n"] = len(pairs)
        return best

    def applied(rows: list[dict], t: float, label: str, conf_key: str = "conf") -> dict:
        accepted = [r for r in rows if label in r and r[conf_key] >= t]
        return {
            "n": len(rows),
            "accepted": len(accepted),
            "coverage": round(len(accepted) / len(rows), 4) if rows else None,
            "share_among_accepted": mean(accepted, label),
        }

    # AUC: does the confidence separate the outputs faithful to the plan from the others?
    result["auc_confidence_vs_plan_fidelity"] = {
        name: {"conf": auc([(r["conf"], r["plan_clean"]) for r in by_split[name]]), "conf_all_steps": auc([(r["conf_all"], r["plan_clean"]) for r in by_split[name]])}
        for name in ("dev", "heldout_lemma", "bench:own_runtime", "bench:own_runtime:no_case")
    }
    # the refusal threshold by PLAN fidelity (100 % of the rows), then the reader fidelity as the second, narrow check
    result["threshold_by_plan_fidelity_on_dev"] = {}
    for level in (0.99, 0.95, 0.90):  # 0.99 is the level of v1 (§5 of LAST_MESSAGE39); the others show the trade-off
        cal = calibrate(by_split["dev"], level, "plan_clean")
        result["threshold_by_plan_fidelity_on_dev"][str(level)] = {
            **cal,
            "stand_own_runtime": applied(by_split["bench:own_runtime"], cal["threshold"], "plan_clean"),
            "stand_own_runtime_no_case": applied(by_split["bench:own_runtime:no_case"], cal["threshold"], "plan_clean"),
            "heldout_lemma": applied(by_split["heldout_lemma"], cal["threshold"], "plan_clean"),
        }
    # for information (NOT a threshold: it is read off the very rows it is applied to): where 99 % plan fidelity would hold on the unseen lemmas
    result["heldout_lemma_own_calibration_for_information"] = calibrate(by_split["heldout_lemma"], 0.99, "plan_clean")
    result["threshold_by_reader_fidelity_on_dev"] = {}
    for level in (0.99, 0.95, 0.90):
        measurable_dev = [r for r in by_split["dev"] if r.get("measurable")]
        cal = calibrate(measurable_dev, level, "faithful")
        result["threshold_by_reader_fidelity_on_dev"][str(level)] = {
            **cal,
            "stand_own_runtime": applied([r for r in by_split["bench:own_runtime"] if r.get("measurable")], cal["threshold"], "faithful"),
            "heldout_lemma": applied([r for r in by_split["heldout_lemma"] if r.get("measurable")], cal["threshold"], "faithful"),
        }

    # ------------------------------------------------------------------ gates (thresholds of LAST_MESSAGE39, unchanged)
    action = ns["RESULTS"]
    tag_unseen = action["heldout_lemma"].get("tag")
    drop = result["drop_seen_unseen"]["exact_2ref"]["drop_pp_reweighted"]
    drop_main = result["drop_seen_unseen"]["main_ok"]["drop_pp_reweighted"]
    modal_s = result["splits"]["heldout_construction:modal_be_predicate"]
    gates = {
        "tag_unseen_lemmas>=0.97": {"value": tag_unseen, "pass": (tag_unseen or 0) >= 0.97, "no_case": action.get("heldout_lemma:no_case", {}).get("tag")},
        "exact_drop_unseen<=3pp (exact_2ref, reweighted by length)": {"value": drop, "pass": drop <= 3.0},
        "drop_unseen<=3pp (words + plan fidelity, reweighted by length)": {"value": drop_main, "pass": drop_main <= 3.0},
        "exact_modal_be>=0.90 (exact_2ref, words; reference figure — not blocking, the order of one corpus reference)": {"value": modal_s["exact_2ref_words"], "pass": (modal_s["exact_2ref_words"] or 0) >= 0.90},
        "modal_be>=0.90 (words equal AND plan fidelity; not blocking)": {"value": modal_s["words_and_plan_fidelity"], "pass": (modal_s["words_and_plan_fidelity"] or 0) >= 0.90},
    }
    for mode, c in result["challenge_own"].items():
        gates[f"challenge_better_than_realize_thought (own, {mode}, exact_2ref, words)"] = {
            "value": [c["network_exact_2ref_words"], c["realize_thought_exact_2ref_words"]],
            "pass": c["network_exact_2ref_words"] > c["realize_thought_exact_2ref_words"],
        }
    result["gates"] = gates
    result["seconds"] = round(time.time() - started)

    # where the stand's runtime input loses fidelity: the bag of forms differs from the reference (human look, not a metric)
    result["forms_failures_own_runtime"] = [
        {"reference": r["reference"], "network": r["text"]}
        for r in by_split["bench:own_runtime"]
        if r.get("faithful_signature") and not r.get("faithful")
    ][:40]
    result["signature_failures_own_runtime"] = [
        {"reference": r["reference"], "network": r["text"]}
        for r in by_split["bench:own_runtime"]
        if r.get("measurable") and not r.get("faithful_signature")
    ][:20]
    result["examples_plan_violations"] = {
        name: [
            {"reference": r["reference"], "network": r["text"], "why": r["plan_explain"]}
            for r in by_split[name]
            if r["plan_violations"]
        ][:25]
        for name in ("dev", "heldout_lemma", "heldout_lemma:no_case", "bench:own_runtime:no_case", "heldout_construction:modal_be_predicate")
    }
    result["examples_modal_be_failures"] = [
        {"reference": r["reference"], "network": r["text"]}
        for r in by_split["heldout_construction:modal_be_predicate"]
        if not r["exact_2ref"]
    ][:15]
    # examples for a human look (not for the report's numbers): the first disagreements on the stand
    result["examples_own_runtime"] = [
        {"reference": r["reference"], "network": r["text"], "ok_2ref": r["exact_2ref"]}
        for r in by_split["bench:own_runtime"][:25]
    ]
    out_path = args.result or (args.out / "evaluation.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    printable = {k: v for k, v in result.items() if not k.startswith(("examples", "forms_failures", "signature_failures"))}
    print(json.dumps(printable, ensure_ascii=False, indent=1))
    print("written to", out_path)


if __name__ == "__main__":
    main()
