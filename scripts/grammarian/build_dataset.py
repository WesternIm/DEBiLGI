# ruff: noqa: E501
"""Build the grammarian dataset (stage 4.25): UD stream + «own readers» stream, splits, statistics.

    PYTHONPATH=<repo>/src python scripts/grammarian/build_dataset.py [--out DIR] [--own-share 0.25]

Inputs  : .codex_work/grammarian_data/raw/UD_Russian-{SynTagRus,Taiga,GSD}/*.conllu (not in git),
          .codex_work/grammarian_bench/bench.json (excluded from every training split).
Outputs : DIR/{train,dev,heldout_lemma,heldout_construction,bench_ud}.jsonl, vocab.json, stats.json.

Rules (LAST_MESSAGE39 §4/§5, 40 §«План этапа B»):
* deduplication by the delexicalised skeleton (``ud_to_tree._skeleton``), at most ``--cap`` (30) per skeleton;
* held-out lemmas: a sentence with a NOUN or VERB lemma whose ``digest("L:"+lemma) % 10 == 0`` never reaches
  ``train`` (pronouns/adjectives are not held out: «мы», «вы» are such lemmas and the grammar of a pronoun
  is not what is tested) — it goes to ``heldout_lemma``;
* held-out constructions: ``comparison`` («как X») and ``modal_be_predicate`` (modal + быть/стать +
  nominal) go to ``heldout_construction`` whole;
* UD test files feed only ``bench_ud`` (the slices of the stand); UD dev files feed ``dev``;
* the own-readers stream is 20–30 % of the training rows and shares the skeleton cap;
* nothing from ``bench.json`` (same normalised text, or the same lemma set) is in any training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ud_to_tree as u  # noqa: E402

ROOT = HERE.parents[1]
RAW = ROOT / ".codex_work/grammarian_data/raw"
BENCH = ROOT / ".codex_work/grammarian_bench/bench.json"
CAP = 30
OWN_CAP = 300
HELDOUT_CONSTRUCTIONS = {"comparison", "modal_be_predicate"}
COMPARISON_TO_TRAIN = 5  # stage 4.27: 4 of every 5 comparison rows (by hash of the text) train, one stays in evaluation
EVAL_CAP = 3000  # the notebook evaluates at most this many rows of a split: only those get alternative references
SOURCES = {  # corpus -> (quality, files)
    "syntagrus": (
        "gold",
        [
            "ru_syntagrus-ud-train-a",
            "ru_syntagrus-ud-train-b",
            "ru_syntagrus-ud-train-c",
            "ru_syntagrus-ud-dev",
            "ru_syntagrus-ud-test",
        ],
    ),
    "gsd": ("gold", ["ru_gsd-ud-train", "ru_gsd-ud-dev", "ru_gsd-ud-test"]),
    # Taiga: dev/test (and the few hundred sentences of the 2.x train) are hand-checked; the files
    # train-a..e (~119k sentences, no enhanced deps) are far larger than the documented gold train
    # (3 138 sentences in README) — treated as silver, an inference to be confirmed (see README).
    "taiga": (
        "silver",
        [
            "ru_taiga-ud-train-a",
            "ru_taiga-ud-train-b",
            "ru_taiga-ud-train-c",
            "ru_taiga-ud-train-d",
            "ru_taiga-ud-train-e",
            "ru_taiga-ud-dev",
            "ru_taiga-ud-test",
        ],
    ),
}
DIR_OF = {"syntagrus": "UD_Russian-SynTagRus", "gsd": "UD_Russian-GSD", "taiga": "UD_Russian-Taiga"}


def digest(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", text.casefold().replace("ё", "е")).split())


def heldout_lemma(tree: dict) -> bool:
    clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
    return any(
        n["pos"] in {"noun", "verb"} and not n.get("elided") and digest("L:" + n["lemma"]) % 10 == 0
        for c in clauses
        for n in c["nodes"]
    )


def moves_to_train(row: dict) -> bool:
    """Stage 4.27: ``comparison`` rows are mostly training material (the construction was absent from
    run1 and could not be learned); ``modal_be_predicate`` stays whole in evaluation (the construction gate).
    Rows with a held-out lemma and rows of dev/test files never train."""
    constructions = set(row["constructions"])
    return (
        "comparison" in constructions
        and "modal_be_predicate" not in constructions
        and not row["heldout_lemma"]
        and row["file_split"] == "train"
        and digest("C:" + row["text"]) % COMPARISON_TO_TRAIN != 0
    )


def role_of_file(name: str) -> str:
    return "test" if name.endswith("-test") else "dev" if name.endswith("-dev") else "train"


def attachment_edges(tree: dict) -> frozenset[tuple[str, str]]:
    """(lemma, lemma of its head) of every spoken node: what the two paths (UD annotation, the bot's readers)
    must agree on. The names of roles are not compared (the converter has roles the readers never produce)."""
    nodes = tree["nodes"]
    return frozenset(
        (n["lemma"], nodes[n["head"]]["lemma"])
        for n in nodes
        if n["head"] is not None and not n.get("elided")
    )


def convert_file(job: tuple[str, str]) -> tuple[str, list[dict], Counter, list[tuple[str, frozenset | None]]]:
    """Worker: every sentence of one file -> accepted rows, reject counter, plain texts for the own stream."""
    corpus, name = job
    quality, _ = SOURCES[corpus]
    if corpus == "taiga" and name.endswith(("-dev", "-test")):
        quality = "gold"
    rows, rejects, texts = [], Counter(), []
    split = role_of_file(name)
    for sentence in u.read_conllu(RAW / DIR_OF[corpus] / f"{name}.conllu"):
        wanted = (
            split == "train"
            and 2 <= len(sentence.text.split()) <= 11
            and not re.search(r"[A-Za-z0-9@#:/,;()\"«»]", sentence.text)
        )
        try:
            result = u.convert(sentence)
            if wanted:  # the UD reading of the same phrase: the own-readers stream is checked against it (stage 4.29)
                texts.append((sentence.text, attachment_edges(result.tree) if result.tree["kind"] == "clause" else None))
        except Exception as reason:  # noqa: BLE001 - counted below; a bug must show up as a reason
            if wanted:
                texts.append((sentence.text, None))
            if not isinstance(reason, u.Reject):
                reason = u.Reject("internal:" + type(reason).__name__)
            rejects[
                str(reason).split(":")[0]
                + (
                    ":" + str(reason).split(":")[1]
                    if str(reason).startswith(
                        (
                            "np:",
                            "clause:",
                            "unlicensed_func",
                            "roundtrip",
                            "internal",
                            "advcl:",
                            "punct:",
                            "preposition:",
                            "inconsistent",
                            "marker",
                        )
                    )
                    and ":" in str(reason)
                    else ""
                )
            ] += 1
            continue
        rows.append(
            {
                "id": f"{corpus}:{sentence.sent_id}",
                "source": corpus,
                "quality": quality,
                "stream": "ud",
                "text": sentence.text,
                "tree": result.tree,
                "target": result.target,
                **result.meta,
                "file_split": split,
            }
        )
    return corpus, rows, rejects, texts


def seed_texts(bench_items: list[dict]) -> list[str]:  # noqa: C901
    """Chat phrases of the seed dialogues (the bot's own corpus) for the readers stream. A dialogue that
    gave any phrase to the stand is excluded whole, so the stand's neighbours never train."""
    stand_dialogues = {
        item["source"].split(":")[1] for item in bench_items if item["source"].startswith("seed:")
    }
    out: list[str] = []
    path = ROOT / ".codex_work/toloka_seed.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            dialogue = json.loads(line)
            if str(dialogue["dialogue"]) in stand_dialogues:
                continue
            for turn in dialogue["turns"]:
                out.extend(p for p in re.split(r"(?<=[.!?])\s+", turn["text"].strip()) if p)
    return out


def alt_worker(job: tuple) -> tuple[str, int, list]:
    import alt_orders

    name, index, tree, target, text = job
    try:
        return name, index, alt_orders.alternative_orders(tree, target, text)
    except Exception:  # noqa: BLE001 - a reader failure means: no alternatives for this row
        return name, index, []


def own_worker(job: tuple[str, frozenset | None]) -> dict | str:
    import own_stream

    text, ud_edges = job
    try:
        result = own_stream.build_item(text)
    except u.Reject as reason:
        return str(reason)
    except Exception as error:  # noqa: BLE001 - a reader failure on one phrase must not stop the build
        return "own:exception:" + type(error).__name__
    if ud_edges is not None and attachment_edges(result.tree) != ud_edges:
        return "own:contradicts_ud"  # both paths read the phrase and disagree about who depends on whom
    return {
        "text": text,
        "tree": result.tree,
        "target": result.target,
        **{k: v for k, v in result.meta.items() if k not in {"thought", "tree_runtime"}},
    }


def cap_by_skeleton(
    rows: list[dict], cap: int, counts: Counter, key: str = "skeleton"
) -> list[dict]:
    kept = []
    for row in sorted(rows, key=lambda r: digest(r["text"])):
        if counts[row[key]] < cap:
            counts[row[key]] += 1
            kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / ".codex_work/grammarian_data")
    parser.add_argument("--own-share", type=float, default=0.25)
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument(
        "--own-cap",
        type=int,
        default=OWN_CAP,
        help="per-skeleton cap of the readers stream (the target distribution: short chat phrases repeat skeletons)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--own-candidates", type=int, default=400_000)
    parser.add_argument("--limit-files", type=int, default=0, help="smoke: first N files only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bench = json.loads(BENCH.read_text())["items"]
    bench_texts = {normalize(item["reference"]) for item in bench}
    bench_lemmas = {frozenset(item["lemmas"]) for item in bench}

    jobs = [(corpus, name) for corpus, (_, files) in SOURCES.items() for name in files]
    if args.limit_files:
        jobs = jobs[: args.limit_files]
    all_rows: list[dict] = []
    rejects: dict[str, Counter] = defaultdict(Counter)
    own_texts: dict[str, frozenset | None] = {}  # phrase -> its UD attachments (None: no UD reading, e.g. a seed phrase)
    with Pool(args.workers) as pool:
        for corpus, rows, counter, texts in pool.imap_unordered(convert_file, jobs):
            all_rows.extend(rows)
            rejects[corpus].update(counter)
            for text, edges in texts:
                if own_texts.get(text) is None:
                    own_texts[text] = edges
        print(
            f"ud: accepted={len(all_rows)} rejected={sum(sum(c.values()) for c in rejects.values())}",
            flush=True,
        )

        # ---- classify the UD rows
        pools: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            row["heldout_lemma"] = heldout_lemma(row["tree"])
            row["heldout_construction"] = bool(HELDOUT_CONSTRUCTIONS & set(row["constructions"]))
            if row["file_split"] == "test":
                pools["bench_ud"].append(row)
            elif row["heldout_construction"]:
                if moves_to_train(row):
                    row["heldout_construction"] = False
                    pools["train"].append(row)
                else:
                    pools["heldout_construction"].append(row)
            elif row["heldout_lemma"]:
                pools["heldout_lemma"].append(row)
            else:
                pools["dev" if row["file_split"] == "dev" else "train"].append(row)
        skeletons: Counter = Counter()
        train_ud = cap_by_skeleton(pools["train"], args.cap, skeletons)
        # train_wide: the same pool capped by the finer (order) skeleton — a bigger, still
        # deduplicated variant for when the coarse cap leaves too little for the model
        wide_skeletons: Counter = Counter()
        wide_ud = cap_by_skeleton(pools["train"], args.cap, wide_skeletons, "skeleton_order")
        splits = {"train": train_ud}
        for name in ("dev", "heldout_lemma", "heldout_construction", "bench_ud"):
            splits[name] = cap_by_skeleton(pools[name], args.cap, Counter())
        print(
            {k: len(v) for k, v in splits.items()},
            "capped from",
            {k: len(v) for k, v in pools.items()},
            flush=True,
        )

        # ---- own readers stream
        seeds = seed_texts(bench)
        for text in seeds:
            own_texts.setdefault(text, None)
        print(f"own pool: {len(set(own_texts))} phrases, of them from seed dialogues {len(set(seeds))}", flush=True)
        # A phrase the UD stream already holds may also enter through the readers (the same
        # sentence by the two paths: a deliberate augmentation); a phrase of any EVALUATION split may not.
        seen: set[str] = {
            normalize(r["text"]) for name, rows in splits.items() if name != "train" for r in rows
        }
        candidates = []
        for text in sorted(set(own_texts), key=digest):
            key = normalize(text)
            if key in seen or key in bench_texts or key.count(" ") + 1 > 12:
                continue
            seen.add(key)
            candidates.append((text, own_texts[text]))
            if len(candidates) >= args.own_candidates:
                break
        wanted = int(len(train_ud) * args.own_share / (1 - args.own_share))
        wanted_wide = int(len(wide_ud) * args.own_share / (1 - args.own_share))
        own_wide: list[dict] = []
        own_skeletons: Counter = Counter()
        own_wide_skeletons: Counter = Counter()
        own_rejects: Counter = Counter()
        own_rows, held_own = [], []
        for result in pool.imap(own_worker, candidates, chunksize=64):
            if isinstance(result, str):
                own_rejects[result] += 1
                continue
            result["heldout_lemma"] = heldout_lemma(result["tree"])
            result["heldout_construction"] = bool(
                HELDOUT_CONSTRUCTIONS & set(result["constructions"])
            )
            if result["heldout_construction"] and moves_to_train(
                {**result, "file_split": "train"}
            ):
                result["heldout_construction"] = False
            result.update(
                {
                    "id": f"own:{digest(result['text']):x}",
                    "source": "own_readers",
                    "quality": "own",
                    "stream": "own",
                    "file_split": "train",
                }
            )
            if result["heldout_construction"] or result["heldout_lemma"]:
                held_own.append(result)
            else:
                # the cap of the readers stream is its own (the UD stream already fills the common
                # skeletons; a shared counter would leave the readers no room at all)
                if len(own_rows) < wanted and own_skeletons[result["skeleton"]] < args.own_cap:
                    own_skeletons[result["skeleton"]] += 1
                    own_rows.append(result)
                if (
                    len(own_wide) < wanted_wide
                    and own_wide_skeletons[result["skeleton_order"]] < args.own_cap
                ):
                    own_wide_skeletons[result["skeleton_order"]] += 1
                    own_wide.append(result)
            if len(own_rows) >= wanted and len(own_wide) >= wanted_wide and len(held_own) >= 300:
                break
        print(
            f"own: wanted={wanted} got={len(own_rows)} heldout={len(held_own)} "
            f"rejects={sum(own_rejects.values())}",
            flush=True,
        )
    splits["train"] = train_ud + own_rows
    splits["train_wide"] = wide_ud + own_wide
    for row in held_own:
        splits["heldout_construction" if row["heldout_construction"] else "heldout_lemma"].append(
            row
        )

    # ---- bench exclusion (text and lemma set) from every training split
    dropped = Counter()
    for name in ("train", "train_wide", "dev"):
        kept = []
        for row in splits[name]:
            if normalize(row["text"]) in bench_texts:
                dropped[f"{name}:text"] += 1
            elif frozenset(row["lemmas"]) in bench_lemmas:
                dropped[f"{name}:lemmas"] += 1
            else:
                kept.append(row)
        splits[name] = kept
    leaks = sum(
        normalize(r["text"]) in bench_texts
        for n in ("train", "train_wide", "dev")
        for r in splits[n]
    )

    # ---- alternative references of the evaluation rows (other orders the readers read into the same tree)
    for name in ("dev", "heldout_lemma", "heldout_construction"):
        splits[name].sort(key=lambda r: r["id"])
    jobs_alt = [
        (name, i, row["tree"], row["target"], row["text"])
        for name in ("dev", "heldout_lemma", "heldout_construction")
        for i, row in enumerate(splits[name][:EVAL_CAP])
        if row["tree"]["kind"] == "clause" and row["n_words"] <= 12
    ]
    with Pool(args.workers) as pool_alt:
        for name, i, accepted in pool_alt.imap_unordered(alt_worker, jobs_alt, chunksize=16):
            if accepted:
                splits[name][i]["alt_texts"] = [text for text, _ in accepted]
                splits[name][i]["alt_targets"] = [actions for _, actions in accepted]
    print(
        "alt references:",
        {
            name: sum(bool(r.get("alt_targets")) for r in splits[name][:EVAL_CAP])
            for name in ("dev", "heldout_lemma", "heldout_construction")
        },
        flush=True,
    )

    # ---- write
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda r: r["id"]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = statistics(splits, rejects, own_rejects, dropped, leaks, args)
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    (args.out / "vocab.json").write_text(
        json.dumps(vocabulary(splits), ensure_ascii=False, indent=1)
    )
    print(json.dumps({k: stats[k] for k in ("sizes", "bench_overlap")}, ensure_ascii=False))


def clauses_of(row: dict) -> list[dict]:
    tree = row["tree"]
    return tree["clauses"] if tree["kind"] == "message" else [tree]


def statistics(splits, rejects, own_rejects, dropped, leaks, args) -> dict:
    out: dict = {
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "by_stream": {},
        "by_source": {},
        "by_quality": {},
        "roles": {},
        "constructions": {},
        "moves": {},
        "tags": Counter(),
        "func_words": Counter(),
        "lemmas": {},
        "skeletons": {},
    }
    for name, rows in splits.items():
        out["by_stream"][name] = dict(Counter(r["stream"] for r in rows))
        out["by_source"][name] = dict(Counter(r["source"] for r in rows))
        out["by_quality"][name] = dict(Counter(r["quality"] for r in rows))
        out["roles"][name] = dict(
            Counter(
                n["role"] for r in rows for c in clauses_of(r) for n in c["nodes"]
            ).most_common()
        )
        out["constructions"][name] = dict(
            Counter(k for r in rows for k in r["constructions"]).most_common()
        )
        out["moves"][name] = dict(Counter(c["clause"]["move"] for r in rows for c in clauses_of(r)))
        out["lemmas"][name] = len({lemma for r in rows for lemma in r["lemmas"]})
        sk = Counter(r["skeleton"] for r in rows)
        out["skeletons"][name] = {
            "unique": len(sk),
            "max_per_skeleton": max(sk.values(), default=0),
        }
        out["words_mean"] = out.get("words_mean", {})
        out["words_mean"][name] = round(sum(r["n_words"] for r in rows) / max(1, len(rows)), 2)
    for r in splits["train"]:
        for action in r["target"]:
            if action[0] == "C":
                out["tags"][action[2]] += 1
            elif action[0] == "F":
                out["func_words"][action[1]] += 1
    out["tags"] = dict(out["tags"].most_common())
    out["func_words"] = dict(out["func_words"].most_common())

    def noun_verb_lemmas(rows):
        return {
            n["lemma"]
            for r in rows
            for c in clauses_of(r)
            for n in c["nodes"]
            if n["pos"] in {"noun", "verb"} and not n.get("elided")
        }

    held = {x for x in noun_verb_lemmas(splits["heldout_lemma"]) if digest("L:" + x) % 10 == 0}
    out["heldout_lemma_check"] = {
        "heldout_noun_verb_lemmas_in_eval": len(held),
        "seen_in_train": len(held & noun_verb_lemmas(splits["train"])),
        "seen_in_train_wide": len(held & noun_verb_lemmas(splits["train_wide"])),
        "seen_in_dev": len(held & noun_verb_lemmas(splits["dev"])),
    }
    out["heldout_construction_check"] = {
        "train_with_modal_be_predicate": sum(
            "modal_be_predicate" in r["constructions"] for r in splits["train"]
        ),
        "train_with_comparison": sum("comparison" in r["constructions"] for r in splits["train"]),
        "heldout_construction_split": dict(
            Counter(
                k
                for r in splits["heldout_construction"]
                for k in r["constructions"]
                if k in HELDOUT_CONSTRUCTIONS
            )
        ),
    }
    out["bench_overlap"] = {
        "dropped": dict(dropped),
        "train_dev_rows_with_bench_text": leaks,
        "bench_items": len(json.loads(BENCH.read_text())["items"]),
    }
    out["rejects_ud"] = {
        corpus: {
            **dict(counter.most_common(40)),
            **{k: v for k, v in counter.items() if k.startswith(("inconsistent", "marker"))},
        }
        for corpus, counter in rejects.items()
    }
    out["rejects_own"] = dict(own_rejects.most_common(20))
    out["params"] = {
        "cap": args.cap,
        "own_share": args.own_share,
        "own_share_actual": round(
            out["by_stream"]["train"].get("own", 0) / max(1, out["sizes"]["train"]), 3
        ),
    }
    return out


def vocabulary(splits) -> dict:
    """Closed vocabularies the notebook masks with."""
    func: dict[str, set[str]] = defaultdict(set)
    tags: Counter = Counter()
    pos: Counter = Counter()
    lexical: Counter = Counter()
    roles: Counter = Counter()
    for name in ("train", "train_wide"):  # closed vocabularies come from training rows only
        for r in splits[name]:
            for action in r["target"]:
                if action[0] == "F":
                    func[action[2].split(":")[0]].add(action[1])
                elif action[0] == "C":
                    tags[action[2]] += 1
            for c in clauses_of(r):
                for n in c["nodes"]:
                    pos[n["pos"]] += 1
                    roles[n["role"]] += 1
                    for k, v in n["lex"].items():
                        lexical[f"{k}={v}"] += 1
    return {
        "version": 2,
        "tree_version": u.TREE_VERSION,
        "func_words_by_licence": {k: sorted(v) for k, v in sorted(func.items())},
        "tags": [t for t, _ in tags.most_common()],
        "pos": sorted(pos),
        "roles": sorted(roles),
        "lex": sorted(lexical),
        "moves": ["statement", "polar_question", "slot_question", "command"],
        "moods": ["indicative", "conditional", "imperative"],
        "tenses": ["past", "present", "future"],
        "links": [
            "main",
            "cause",
            "condition",
            "coordination",
            "embedded_polar",
            "embedded_wh",
            "complement",
        ],
        "punct": sorted(u.PUNCT_OK - u.TERMINALS),  # inner marks only: the terminal mark is the clause feature ``end``
        "ends": [*sorted(u.TERMINALS), "none"],
    }


if __name__ == "__main__":
    main()
