"""Optional local Slovnet dependency parser. No text generation or network I/O."""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

from debilgi.language.syntax_models import verify_models

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyntaxEdge:
    head: int
    dependent: int
    relation: str


class SlovnetParser:
    def __init__(self, model_dir: Path):
        verify_models(model_dir)
        from navec import Navec
        from slovnet import Syntax

        vectors = Navec.load(str(model_dir / "navec_news_v1_1B_250K_300d_100q.tar"))
        self.model = Syntax.load(str(model_dir / "slovnet_syntax_news_v1.tar"))
        self.model.navec(vectors)
        self._lock = Lock()
        self.failed = False
        self.parse = lru_cache(maxsize=512)(self._parse)

    def _parse(self, tokens: tuple[str, ...]) -> tuple[SyntaxEdge, ...]:
        if self.failed or not 2 <= len(tokens) <= 128:
            return ()
        try:
            with self._lock:
                markup = self.model(list(tokens))
            if tuple(t.text for t in markup.tokens) != tokens:
                logger.debug("Slovnet token alignment mismatch; using local grammar")
                return ()
            edges = tuple(SyntaxEdge(int(t.head_id) - 1, int(t.id) - 1, t.rel)
                          for t in markup.tokens)
            if any(not -1 <= e.head < len(tokens) or not 0 <= e.dependent < len(tokens)
                   or e.head == e.dependent for e in edges):
                logger.debug("Invalid Slovnet dependency indices; using local grammar")
                return ()
            if sum(e.head == -1 for e in edges) != 1:
                return ()
            return edges
        except Exception:
            self.failed = True
            logger.exception("Slovnet disabled after inference failure; using local grammar")
            return ()


_parser: SlovnetParser | None = None
_analysis_identity = "local"


def configure_syntax_backend(model_dir: Path | None) -> bool:
    global _parser
    _parser = None
    if model_dir is None:
        _configure_analysis_identity("local")
        return False
    try:
        _parser = SlovnetParser(model_dir)
    except Exception:
        logger.exception("Slovnet unavailable; using local grammar")
        _configure_analysis_identity("local")
        return False
    digest = hashlib.sha256()
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    _configure_analysis_identity(digest.hexdigest())
    logger.info("Local Slovnet syntax enabled: %s", model_dir)
    return True


def _configure_analysis_identity(key):
    global _analysis_identity
    if key == _analysis_identity:
        return
    _analysis_identity = key
    module = sys.modules.get("debilgi.language.dialogue_learning")
    if module is not None:
        module.configure_analysis_cache(key)
    structure = sys.modules.get("debilgi.language.structured_utterance")
    if structure is not None:
        structure._cached_structure.cache_clear()


def syntax_edges(tokens: tuple[str, ...]) -> tuple[SyntaxEdge, ...]:
    return _parser.parse(tokens) if _parser is not None else ()


def enrich_structure(structure):
    """Use aligned UD subjects/objects inside simple, explicit lexical clauses.

    News-trained predictions cannot erase quoted scope, modality, negation or
    cross-clause links. Complex structures retain the existing semantic parser.
    """
    from dataclasses import replace

    from debilgi.language.structured_utterance import DependencyArc

    if len(structure.clauses) != 1 or structure.clause_links:
        return structure
    frame = structure.clauses[0]
    if (frame.predicate_index is None or frame.modal_lemmas or frame.vocative_indices
            or frame.coordinations or frame.action_complement_indices
            or any(t.semantic_role == "quoted_content" for t in structure.tokens)):
        return structure
    if structure.tokens[frame.predicate_index].part_of_speech != "verb":
        return structure
    edges = syntax_edges(tuple(t.surface for t in structure.tokens))
    root = next((e.dependent for e in edges if e.head == -1), None)
    if root != frame.predicate_index:
        return structure
    subjects = tuple(e.dependent for e in edges if e.head == root and e.relation == "nsubj")
    objects = tuple(e.dependent for e in edges if e.head == root and e.relation == "obj")
    if len(subjects) > 1 or len(objects) > 1:
        return structure
    # Require an explicit subject and compatible nominal arguments. Do not
    # interpret a passive subject as an agent or a question word as an entity.
    if not subjects or any(e.relation.startswith("nsubj:") for e in edges):
        return structure
    selected = (*subjects, *objects)
    if any(structure.tokens[i].part_of_speech not in {"noun", "pronoun"}
           or i == frame.question_target_index for i in selected):
        return structure
    if any(e.head == i and e.relation == "case" for e in edges for i in selected):
        return structure
    updated = replace(frame, subject_indices=subjects,
                      direct_object_indices=objects or tuple(
                          i for i in frame.direct_object_indices if i not in subjects))
    # Replace only the two roles for which aligned parser evidence exists.
    cards = []
    for card in structure.tokens:
        role = card.semantic_role
        if role in {"subject", "direct_object"}:
            role = "unassigned_content"
        if card.index in updated.subject_indices:
            role = "subject"
        elif card.index in updated.direct_object_indices:
            role = "direct_object"
        cards.append(replace(card, semantic_role=role))
    deps = tuple(e for e in structure.dependencies if e.relation not in {"nsubj", "obj"})
    deps += tuple(DependencyArc(frame.clause_index, root, i, rel)
                  for rel, indices in (("nsubj", updated.subject_indices),
                                       ("obj", updated.direct_object_indices)) for i in indices)
    return replace(structure, tokens=tuple(cards), clauses=(updated,), dependencies=deps,
                   niche_signals=(*structure.niche_signals, "syntax_backend:slovnet"))
