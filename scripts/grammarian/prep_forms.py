# ruff: noqa: E501
"""Allomorphs of prepositions (в/во, с/со, к/ко, из/изо ...) read from the UD corpora -> ``prep_forms.json``.

A preposition is a property of its node in the v2 format; its surface depends on the word that follows it.
The rule is learned from observation (no hand-written list): for every preposition lemma with a second
surface form, a decision list over the first 1-4 letters of the next word (a prefix is kept when it was
seen >= 3 times and one form covers >= 90 % of them; a rule that the shorter rules already give is pruned).

    PYTHONPATH=<repo>/src python scripts/grammarian/prep_forms.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ud_to_tree as u  # noqa: E402

RAW = HERE.parents[1] / ".codex_work/grammarian_data/raw"
MIN_SEEN, MIN_SHARE = 3, 0.9


def main() -> None:
    seen: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))  # lemma -> prefix -> forms
    total: dict[str, Counter] = defaultdict(Counter)
    for path in sorted(RAW.glob("*/*.conllu")):
        for sentence in u.read_conllu(path):
            toks = sentence.toks
            for i, tok in enumerate(toks[:-1]):
                nxt = toks[i + 1]
                if tok.upos != "ADP" or nxt.upos == "PUNCT":
                    continue
                lemma, form, after = u.norm(tok.lemma), u.norm(tok.form), u.norm(nxt.form)
                if not (form.isalpha() and lemma.isalpha() and after.isalpha()) or form[:1] != lemma[:1]:
                    continue
                total[lemma][form] += 1
                for k in range(1, 5):
                    seen[lemma][after[:k]][form] += 1
    rules: dict[str, list[list[str]]] = {}
    for lemma, forms in total.items():
        if len(forms) < 2 or sorted(forms.values())[-2] < MIN_SEEN:
            continue
        candidates = []
        for prefix, counts in seen[lemma].items():
            n = sum(counts.values())
            form, top = counts.most_common(1)[0]
            if n >= MIN_SEEN and top / n >= MIN_SHARE:
                candidates.append((prefix, form))
        kept: list[tuple[str, str]] = []
        for prefix, form in sorted(candidates, key=lambda c: (len(c[0]), c[0])):
            given = lemma
            for p, f in sorted(kept, key=lambda c: -len(c[0])):
                if prefix.startswith(p):
                    given = f
                    break
            if given != form:
                kept.append((prefix, form))
        if kept:
            rules[lemma] = [[p, f] for p, f in sorted(kept, key=lambda c: (-len(c[0]), c[0]))]
    (HERE / "prep_forms.json").write_text(json.dumps(rules, ensure_ascii=False, indent=0))
    print({lemma: len(r) for lemma, r in rules.items()})


if __name__ == "__main__":
    main()
