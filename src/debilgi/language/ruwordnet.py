"""Read-only RuWordNet senses and typed links; never a response generator."""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from debilgi.language.content import content_lemmas
from debilgi.language.morphology import POS_NAMES
from debilgi.language.tokenizer import tokenize

logger = logging.getLogger(__name__)
_POS = {"noun": "N", "verb": "V", "adjective": "A", "participle": "A"}
# Table identifiers are constants, never supplied by input text.
_LINKS = (
    ("hypernym_relation", "hyponym_id", "hypernym_id", "is_a"),
    ("meronymy_relation", "meronym_id", "holonym_id", "part_of"),
    ("meronymy_relation", "holonym_id", "meronym_id", "has_part"),
    ("domain_relation", "domain_item_id", "domain_id", "domain"),
    ("instance_relation", "instance_id", "class_id", "instance_of"),
    ("antonymy_relation", "left_id", "right_id", "antonym"),
    ("antonymy_relation", "right_id", "left_id", "antonym"),
    ("related_relation", "left_id", "right_id", "related"),
    ("related_relation", "right_id", "left_id", "related"),
)


@dataclass(frozen=True, slots=True)
class LexicalLink:
    relation: str
    target_id: str
    target: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LexicalSense:
    synset_id: str
    title: str
    definition: str
    part_of_speech: str
    synonyms: tuple[str, ...]
    links: tuple[LexicalLink, ...]

    @property
    def meaning(self) -> str:
        return self.definition or self.title


@dataclass(frozen=True, slots=True)
class SenseSelection:
    senses: tuple[LexicalSense, ...]
    selected: LexicalSense | None
    evidence: tuple[str, ...] = ()


class RuWordNet:
    """A bounded view of the published SQLite file, without importing into live memory."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.uri = self.path.as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(self.uri, uri=True)) as db:
            # Missing files/tables fail at configuration, not on every word.
            db.execute("SELECT id, name, lemma, synset_id FROM sense LIMIT 1").fetchone()
            db.execute(
                "SELECT id, title, definition, part_of_speech FROM synset LIMIT 1"
            ).fetchone()
            self.tables = frozenset(row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ))
            # The published database has no index on sense.synset_id. Load
            # this finite alias mapping once instead of scanning 154k senses
            # for every neighbor of every input word.
            aliases = defaultdict(list)
            for sid, name in db.execute("SELECT synset_id, name FROM sense ORDER BY name"):
                if name.casefold() not in aliases[sid]:
                    aliases[sid].append(name.casefold())
            self.aliases = {sid: tuple(names) for sid, names in aliases.items()}
        self.senses = lru_cache(maxsize=2048)(self._senses)
        self.related_forms = lru_cache(maxsize=1024)(self._related_forms)

    def _aliases(self, synset_id: str) -> tuple[str, ...]:
        return self.aliases.get(synset_id, ())[:64]

    def _senses(self, lemma: str, part: str) -> tuple[LexicalSense, ...]:
        mapped = _POS.get(POS_NAMES.get(part, part))
        if mapped is None or not lemma or len(lemma) > 100:
            return ()
        names = (lemma.upper(), lemma.upper().replace("Ё", "Е"))
        with closing(sqlite3.connect(self.uri, uri=True)) as db:
            rows = db.execute(
                """SELECT DISTINCT s.id, s.title, s.definition, s.part_of_speech
                   FROM sense w JOIN synset s ON s.id = w.synset_id
                   WHERE (w.name IN (?, ?) OR w.lemma IN (?, ?)) AND s.part_of_speech = ?
                   ORDER BY s.id LIMIT 33""", (*names, *names, mapped),
            ).fetchall()
            if len(rows) > 32:
                return ()  # never select a sense from a silently truncated set
            result = []
            for sid, title, definition, pos in rows:
                links = []
                for table, source, target, relation in _LINKS:
                    if table not in self.tables:
                        continue
                    targets = db.execute(
                        f"SELECT s.id, s.title FROM {table} r JOIN synset s ON s.id=r.{target} "
                        f"WHERE r.{source}=? ORDER BY s.id LIMIT 64", (sid,),
                    ).fetchall()
                    for tid, target_title in targets:
                        aliases = self._aliases(tid)
                        # A canonical lexical item is safer to inflect than a
                        # thesaurus heading containing commas or parentheses.
                        name = min(aliases, key=lambda a: (len(a.split()), len(a), a),
                                   default=target_title.casefold())
                        links.append(LexicalLink(relation, tid, name, aliases))
                result.append(LexicalSense(
                    sid, title.casefold(), (definition or "").strip(), pos,
                    self._aliases(sid), tuple(dict.fromkeys(links)),
                ))
            return tuple(result)

    def lookup(self, lemma: str, part: str, context: str = "") -> SenseSelection:
        try:
            senses = self.senses(lemma, part)
            selection = select_sense(lemma, senses, context)
            if selection.selected is not None or len(senses) < 2:
                return selection
            derived = tuple((term, form)
                            for term in sorted(_terms(context[:2000]) - _terms(lemma))[:8]
                            for form in self.related_forms(term))
            return select_sense(lemma, senses, context, derived=derived)
        except sqlite3.Error:
            logger.exception("RuWordNet lookup failed; retaining existing lexical analysis")
            return SenseSelection((), None)

    def _related_forms(self, lemma: str) -> tuple[str, ...]:
        if "derivation_relation" not in self.tables:
            return ()
        names = (lemma.upper(), lemma.upper().replace("Ё", "Е"))
        with closing(sqlite3.connect(self.uri, uri=True)) as db:
            return tuple(row[0].casefold() for row in db.execute(
                """SELECT DISTINCT d.name FROM sense s
                   JOIN derivation_relation r ON r.source_id=s.id
                   JOIN sense d ON d.id=r.derivative_id
                   WHERE s.name IN (?, ?) OR s.lemma IN (?, ?) ORDER BY d.name LIMIT 32""",
                (*names, *names),
            ))


@lru_cache(maxsize=4096)
def _terms(text: str) -> frozenset[str]:
    return frozenset(content_lemmas(text))


def select_sense(
    lemma: str, senses: tuple[LexicalSense, ...], context: str,
    *, derived: tuple[tuple[str, str], ...] = (),
) -> SenseSelection:
    if len(senses) == 1:
        return SenseSelection(senses, senses[0], (f"ruwordnet:unique:{senses[0].synset_id}",))
    if not senses:
        return SenseSelection((), None)
    if set(tokenize(context[:2000])) & {"не", "ни", "без"}:
        # Until scope can bind exclusions to individual senses, negated
        # context must not become positive disambiguating evidence.
        return SenseSelection(senses, None)
    terms = _terms(context[:2000]) - _terms(lemma)
    scored = []
    for sense in senses:
        cues = _terms(" ".join((sense.title, sense.definition, *sense.synonyms))) - _terms(lemma)
        evidence = {term: 3 for term in terms & cues}
        for source, form in derived:
            if _terms(form) & cues:
                evidence[f"derivation:{source}"] = 2
        for link in sense.links:
            if link.relation not in {"is_a", "part_of", "has_part", "domain", "related"}:
                continue
            matches = terms & (_terms(" ".join(link.aliases)) - _terms(lemma))
            for term in matches:
                evidence[term] = max(evidence.get(term, 0), 1 if link.relation == "related" else 2)
        scored.append((sum(evidence.values()), sense, tuple(sorted(evidence))))
    scored.sort(key=lambda item: (-item[0], item[1].synset_id))
    best, runner = scored[0], scored[1]
    if best[0] < 2 or best[0] - runner[0] < 2:
        return SenseSelection(senses, None)
    return SenseSelection(senses, best[1], tuple(
        f"ruwordnet:context:{best[1].synset_id}:{term}" for term in best[2]
    ))


def open_ruwordnet(path: Path | None) -> RuWordNet | None:
    if path is None:
        return None
    try:
        store = RuWordNet(path)
    except (OSError, sqlite3.Error):
        logger.exception("RuWordNet unavailable; existing dictionaries remain active")
        return None
    logger.info("Local RuWordNet enabled: %s", path)
    return store
