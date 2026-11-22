# ruff: noqa: E501
"""The «grammarian» (stage 4.30, stage D of LAST_MESSAGE39): a small personal network that only *words* a finished plan.

The plan (``ClauseThought``) is turned into the tree of ``plan_tree.thought_to_tree`` (format v3 of LAST_MESSAGE45), the
network chooses an order and the morphological tag of every input word, and ``pymorphy`` builds the forms. The network
cannot write a word of its own: its output vocabulary is the input nodes, the closed-class phrases the PLAN gave
(negation, conditional particle, a link word, a comparison marker, a preposition) and inner punctuation. It adds nothing
to the meaning, writes nothing to memory, and when it is not sure it refuses: ``realize_thought`` then answers as before.

Everything here is optional and off by default (``DEBILGI_GRAMMARIAN=1`` turns it on); the numpy inference needs no torch.

Contract of the input (LAST_MESSAGE45 §6):

* the closed-class words are given by the plan (``не`` of a negated clause, ``бы`` of a conditional predicate, the
  comparison marker the plan names — ``как`` when it names none, exactly as ``realize_thought`` does);
* a case enters only as ``ConceptArgument.grammatical_case`` (the plan fixed it) — never ``_case_for_role``; the nominal
  part of a clause is the one exception: it is nominative for a present «быть» and instrumental otherwise;
* the gender of «я» is the bot's own (a fact about itself: ``DEBILGI_SELF_GENDER``), of «ты» — the interlocutor's (an explicit
  feature of the lexeme or the ``interlocutor_gender`` the caller passes); an unknown gender that agreement needs is an
  INCOMPLETE plan, i.e. a refusal, never a guess;
* degree / short form come from the observed ``ConceptLexeme.surface`` by morphology; ``given`` is off.

The path is traced: every call leaves a ``GrammarianRecord`` (chosen path, confidence, the reason of a refusal).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import math
import os
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.plan_tree import thought_to_tree
from debilgi.language.productive_nlg import (
    _QUESTION_FOCUS_LEMMAS,
    _nominal_complement_case,
    preposition_allomorph,
)

ENV_FLAG = "DEBILGI_GRAMMARIAN"
ENV_DIR = "DEBILGI_GRAMMARIAN_DIR"
ENV_THRESHOLD = "DEBILGI_GRAMMARIAN_THRESHOLD"
ENV_SELF_GENDER = "DEBILGI_SELF_GENDER"
DEFAULT_SELF_GENDER = "masculine"  # the owner's decision of 2026-11-21: the bot speaks of itself in the masculine
_TRUE = {"1", "true", "yes", "on"}
_NEG_WORD, _COND_WORD = "не", "бы"  # the closed-class words the plan licenses for a negated clause / a conditional mood
_COMPARISON_DEFAULT = "как"  # as in ``realize_thought``: the likeness marker when the plan names none
_SPEAKERS = frozenset({"я", "ты", "вы"})
_ROOT = Path(__file__).resolve().parents[3]

_UPOS = {
    "noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "participle": "VERB", "adverb": "ADV",
    "number": "NUM", "pronoun": "PRON", "gerund": "VERB", "particle": "PART",
}
_POS_OK = {
    "NOUN": {"NOUN"}, "PROPN": {"NOUN"}, "ADJ": {"ADJF", "ADJS", "COMP", "PRTF", "PRTS", "NPRO"},
    "DET": {"ADJF", "NPRO", "NUMR", "ADJS"}, "PRON": {"NPRO", "ADJF", "NOUN"}, "NUM": {"NUMR", "ADJF", "NOUN"},
    "VERB": {"VERB", "INFN", "PRTF", "PRTS", "GRND"}, "AUX": {"VERB", "INFN"}, "ADV": {"ADVB", "COMP", "PRED", "NPRO"},
}
_INVARIABLE_POS = {"adverb", "particle", "pronoun", "number"}  # a "-" tag is a cell of these, and only of these
_LONG_CASE_SHORT = {
    "nominative": "nomn", "genitive": "gent", "dative": "datv", "accusative": "accs", "instrumental": "ablt",
    "prepositional": "loct",
}
_MARKS_LEFT = frozenset({",", ";", ":"})  # marks glued to the word before them


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in _TRUE


def self_gender() -> str:
    return os.environ.get(ENV_SELF_GENDER, DEFAULT_SELF_GENDER).strip().lower() or DEFAULT_SELF_GENDER


# ---------------------------------------------------------------------------------------------------- the journal
@dataclass(slots=True)
class GrammarianRecord:
    """One call of the adapter: which path answered and why (the trace of LAST_MESSAGE39 §5)."""

    path: str  # "network" | "fallback"
    reason: str | None  # why the network refused (None when it answered)
    confidence: float | None
    threshold: float
    milliseconds: float
    predicate: str
    text: str | None = None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path": self.path, "reason": self.reason,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "threshold": self.threshold, "milliseconds": round(self.milliseconds, 2),
            "predicate": self.predicate, "text": self.text, **self.details,
        }


_collector: contextvars.ContextVar[list[GrammarianRecord] | None] = contextvars.ContextVar(
    "grammarian_collector", default=None
)
_lock = threading.Lock()
STATS: Counter = Counter()  # process-wide: "network", "fallback:<reason>" ...
RECENT: deque[GrammarianRecord] = deque(maxlen=50_000)


@contextlib.contextmanager
def capture() -> Iterator[list[GrammarianRecord]]:
    """Collect the records of every call made inside the block (per context, so concurrent replies do not mix)."""
    records: list[GrammarianRecord] = []
    token = _collector.set(records)
    try:
        yield records
    finally:
        _collector.reset(token)


def _journal(record: GrammarianRecord) -> None:
    with _lock:
        STATS["network" if record.path == "network" else f"fallback:{record.reason}"] += 1
        STATS["calls"] += 1
        RECENT.append(record)
    sink = _collector.get()
    if sink is not None:
        sink.append(record)


def note_fallback(thought, reason: str) -> None:
    """Journal a call the network did not even try (the caller fixed the order)."""
    _journal(GrammarianRecord("fallback", reason, None, 0.0, 0.0,
                              getattr(getattr(thought, "predicate", None), "lemma", "") or ""))


def reset_stats() -> None:
    _cache.clear()
    with _lock:
        STATS.clear()
        RECENT.clear()


# ---------------------------------------------------------------------------------------------------- morphology
def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def _analyzer():
    return DEFAULT_MORPHOLOGY._analyzer


@functools.lru_cache(maxsize=400_000)
def inflect(lemma: str, upos: str, tag: str, prosthetic: bool = False) -> str | None:
    """The pymorphy form of ``lemma`` for ``tag`` (the same rule as the data path ``ud_to_tree.inflect``; the form keeps
    its «ё»: the corpus normalised it away, a reply should not)."""
    plain = lemma.lower()
    if tag == "-":
        return plain
    wanted = _POS_OK.get(upos)
    grams = set(tag.split(","))
    key = _norm(plain)
    readings = _analyzer().parse(plain)
    own = [r for r in readings if _norm(r.normal_form) == key] or list(readings)
    own = [r for r in own if wanted is None or str(r.tag.POS) in wanted] or own
    for reading in own:
        try:
            inflected = reading.inflect(grams)
        except Exception:  # noqa: BLE001 - pymorphy rejects unknown grammemes for a POS
            continue
        if inflected is not None:
            word = str(inflected.word).lower()
            if prosthetic and key in {"он", "она", "оно", "они"} and word[:1] in "еёи":
                word = "н" + word
            return word
    return None


# ---------------------------------------------------------------------------------------------------- numpy layers
def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU like torch's default; erf by Abramowitz–Stegun 7.1.26 refined with one Newton step."""
    z = x * np.float32(0.70710678118654752)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    erf = sign * (1.0 - poly * t * np.exp(-a * a))
    return (0.5 * x * (1.0 + erf)).astype(np.float32)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + 1e-5) * w + b).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- the model
class GrammarianModel:
    """Numpy inference of the v3 export (``grammarian_v3.npz`` + ``featurizer.json``), batch of one."""

    def __init__(self, directory: Path) -> None:
        meta = json.loads((directory / "featurizer.json").read_text(encoding="utf-8"))
        self.cfg: dict = meta["cfg"]
        self.fz: dict = meta["fz"]
        self.tags: list[str] = meta["tags"]
        self.n_other: int = meta["n_other"]
        self.w = {k: v.astype(np.float32) for k, v in np.load(directory / "grammarian_v3.npz").items()}
        side = json.loads((directory / "paradigm_tags.json").read_text(encoding="utf-8"))
        if side["tags"] != self.tags:
            raise ValueError("paradigm_tags.json does not belong to this export")
        self.pos_tags: dict[str, list[int]] = side["pos_tags"]
        prep = directory / "prep_forms.json"
        self.prep_forms: dict[str, list[list[str]]] = (
            json.loads(prep.read_text(encoding="utf-8")) if prep.exists() else {}
        )
        refusal = directory / "refusal.json"
        self.threshold: float = (
            float(json.loads(refusal.read_text(encoding="utf-8"))["threshold"]) if refusal.exists()
            else float(meta.get("threshold", 1.0))
        )
        override = os.environ.get(ENV_THRESHOLD)
        if override:
            self.threshold = float(override)
        self.directory = directory
        self.func_words: dict[int, str] = {i: p for p, i in self.fz["func"].items()}
        self.punct_marks: dict[int, str] = {i: p for p, i in self.fz["punct"].items()}
        self.n_func, self.n_punct, self.n_tags = len(self.fz["func"]), len(self.fz["punct"]), len(self.tags)
        self.lex_dim = len(self.fz["lex"])
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        self.d = self.cfg["d"]
        # the conv bank as (filters, k·cd) matrices per kernel width
        self._convs = [(self.w[f"convs.{i}.weight"], self.w[f"convs.{i}.bias"], k) for i, k in enumerate((2, 3, 4))]

    # ------------------------------------------------------------------------------------------ features (``encode``)
    @functools.lru_cache(maxsize=100_000)  # noqa: B019 - one model per process
    def paradigm_row(self, lemma: str, pos: str) -> tuple[int, ...]:
        """The tags a lemma's paradigm can realize (the mask of the tag head; ``export_paradigms.py`` rule)."""
        allowed = []
        for index in self.pos_tags.get(pos, [self.tag_index.get("-", 0)]):
            tag = self.tags[index]
            form = inflect(lemma, _UPOS.get(pos, "NOUN"), tag)
            if form is not None and (tag != "-" or pos in _INVARIABLE_POS):
                allowed.append(index)
        return tuple(allowed or self.pos_tags.get(pos) or range(self.n_tags))

    def punct_caps(self, clauses: list[dict], flat: list[dict], heads: list[int | None]) -> np.ndarray:
        siblings = Counter(h for n, h in zip(flat, heads, strict=True) if h is not None and n["role"] == "modifier")
        runs = sum(max(0, v - 1) for v in siblings.values())
        inner = len(clauses) - 1
        by_mark = {",": 2 * inner + sum(n["role"] == "conjunct" for n in flat) + runs + 1,
                   "-": 2, "—": 2, ":": 1, ";": inner + 1}
        caps = np.zeros(self.n_punct, dtype=np.int64)
        for mark, i in self.fz["punct"].items():
            caps[i] = by_mark.get(mark, 0)
        return caps

    def licences(self, clauses: list[dict]) -> tuple[np.ndarray, int]:
        need = np.zeros(self.n_func, dtype=np.int64)
        unknown = 0
        pairs = [pair for c in clauses for pair in c["clause"].get("func", [])]
        pairs += [pair for c in clauses for n in c["nodes"] for pair in n.get("func", [])]
        for kind, word in pairs:
            j = self.fz["func"].get(f"{kind}|{word}")
            if j is None:
                unknown += 1
            else:
                need[j] += 1
        return need, unknown

    @staticmethod
    def nominal_groups(nodes: list[dict], heads: list[int | None], silent: np.ndarray) -> np.ndarray:
        n = len(nodes)
        grp = np.zeros((n, n), dtype=bool)
        for h, node in enumerate(nodes):
            if node["pos"] == "verb" or node["role"] in {"predicate", "predicate_complement", "action_complement"}:
                continue
            for g in range(n):
                k, steps = g, 0
                while k is not None and steps < 60:
                    if k == h:
                        grp[h, g] = not silent[g]
                        break
                    k, steps = heads[k], steps + 1
        return grp

    def encode(self, tree: dict) -> dict | None:
        """One tree -> arrays; the same computation as the training notebook's ``encode`` (``use_given`` off)."""
        fz, cfg = self.fz, self.cfg
        clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
        nodes: list[tuple[int, dict]] = []
        base: list[int] = []
        for ci, c in enumerate(clauses):
            base.append(len(nodes))
            nodes.extend((ci, n) for n in c["nodes"])
        if not nodes or len(nodes) > cfg["max_nodes"]:
            return None
        count = len(nodes)
        end = fz["end"].get(clauses[0]["clause"].get("end", "none"), 0)
        cat = np.zeros((count, 18), dtype=np.int64)
        flags = np.zeros((count, 8), dtype=np.float32)
        lex = np.zeros((count, self.lex_dim), dtype=np.float32)
        chars = np.zeros((count, cfg["max_chars"]), dtype=np.int64)
        verb = np.zeros(count, dtype=np.int64)
        silent = np.zeros(count, dtype=bool)
        head_g = np.full(count, -1)
        clause_of = np.zeros(count, dtype=np.int64)
        for g, (ci, n) in enumerate(nodes):
            c = clauses[ci]["clause"]
            link = clauses[ci].get("link", {}).get("type", "main")
            head = None if n["head"] is None else base[ci] + n["head"]
            head_g[g], clause_of[g] = -1 if head is None else head, ci
            depth, h = 0, head
            while h is not None and depth < 4:
                depth += 1
                h = None if nodes[h][1]["head"] is None else base[nodes[h][0]] + nodes[h][1]["head"]
            is_silent = bool(n.get("elided")) or (
                n["id"] == 0 and c["predicate_class"] == "copular" and c["tense"] == "present")
            silent[g] = is_silent
            fphrase = fz["fphrase"]
            func = c.get("func", [])
            link_p = [f"{k}|{w}" for k, w in func if k == "link"]
            neg_p = [f"{k}|{w}" for k, w in func if k == "particle_negation"]
            conj_p = [f"{k}|{w}" for k, w in n.get("func", []) if k == "conjunction"]
            cat[g] = [
                fz["role"].get(n["role"], 0), fz["pos"].get(n["pos"], 0), fz["prep"].get(n.get("preposition"), 0),
                fz["marker"].get(n.get("marker"), 0), fz["case"].get(n.get("case_constraint"), 0),
                0 if head is None else fz["role"].get(nodes[head][1]["role"], 0),
                fz["move"].get(c["move"], 0), fz["mood"].get(c["mood"], 0), fz["tense"].get(c["tense"], 0),
                fz["pclass"].get(c["predicate_class"], 0), fz["link"].get(link, 0), end,
                fz["marker_kind"].get(n.get("marker_kind"), 0),
                fz["rank"].get(None if n.get("conj_rank") is None else str(n["conj_rank"]), 0), 0,
                fphrase.get(link_p[0], 0) if link_p else 0, fphrase.get(neg_p[0], 0) if neg_p else 0,
                fphrase.get(conj_p[0], 0) if conj_p else 0,
            ]
            flags[g] = [float(bool(n.get("elided"))), float(c["negated"]), float(c["negation_node"] == n["id"]),
                        float(c["focus"] == n["id"]), float(c["subject_elided"]), float(is_silent), depth / 4.0,
                        float(n["lemma"] in _SPEAKERS and n["pos"] == "pronoun")]
            for k, v in n["lex"].items():
                j = fz["lex"].get(f"{k}={v}")
                if j is not None:
                    lex[g, j] = 1.0
            ids = [fz["chars"].get(ch, 1) for ch in n["lemma"][-cfg["max_chars"]:]]
            chars[g] = ids + [0] * (cfg["max_chars"] - len(ids))
            verb[g] = fz["verbs"].get(n["lemma"], 0) if n["pos"] == "verb" else 0
        rel = np.full((count, count), 6, dtype=np.int64)
        same_clause = clause_of[:, None] == clause_of[None, :]
        rel[same_clause] = 5
        sibling = (head_g[:, None] == head_g[None, :]) & (head_g[:, None] >= 0)
        rel[sibling] = 4
        rel[head_g[None, :] == np.arange(count)[:, None]] = 3  # rel[i, j] = 3: j has head i
        rel[head_g[:, None] == np.arange(count)[None, :]] = 2  # rel[i, j] = 2: i has head j
        np.fill_diagonal(rel, 1)
        para = np.zeros((count, self.n_tags), dtype=bool)
        for g, (_, n) in enumerate(nodes):
            para[g, list(self.paradigm_row(n["lemma"], n["pos"]))] = True
        need, need_unknown = self.licences(clauses)
        flat = [n for _, n in nodes]
        head_list = [None if h < 0 else int(h) for h in head_g]
        return {
            "cat": cat, "flags": flags, "lex": lex, "chars": chars, "verb": verb, "rel": rel, "silent": silent,
            "para": para, "need": need, "need_unknown": need_unknown,
            "grp": self.nominal_groups(flat, head_list, silent),
            "punct_cap": self.punct_caps(clauses, flat, head_list), "N": count,
        }

    # ------------------------------------------------------------------------------------------ encoder
    def _conv_pool(self, ch: np.ndarray) -> np.ndarray:
        """Three 1-d convolutions (kernel 2/3/4, padding k//2) over the letters of every lemma, max over time."""
        pooled = []
        for weight, bias, k in self._convs:
            pad = k // 2
            padded = np.pad(ch, ((0, 0), (pad, pad), (0, 0)))  # N, C+2p, cd
            windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=1)  # N, L, cd, k
            out = np.einsum("nlck,fck->nlf", windows, weight, optimize=True) + bias
            pooled.append(out.max(1))
        return np.concatenate(pooled, -1)

    def encode_nodes(self, e: dict) -> np.ndarray:
        w, d = self.w, self.d
        cat = np.concatenate([w[f"cat_emb.{i}.weight"][e["cat"][:, i]] for i in range(18)], -1)
        x_in = np.concatenate([cat, e["lex"] @ w["lex_proj.weight"].T + w["lex_proj.bias"],
                               e["flags"] @ w["flag_proj.weight"].T + w["flag_proj.bias"]], -1)
        x = x_in @ w["in_proj.weight"].T + w["in_proj.bias"]
        ch = w["char_emb.weight"][e["chars"]]  # N, C, cd
        lemma = _gelu(self._conv_pool(ch)) @ w["char_out.weight"].T + w["char_out.bias"]
        lengths = np.maximum((e["chars"] != 0).sum(-1), 1)
        idx = np.maximum(lengths[:, None] - 4 + np.arange(4)[None, :], 0)
        last = np.take_along_axis(e["chars"], idx, 1)
        suffix = w["char_emb.weight"][last].reshape(e["N"], -1) @ w["suffix.weight"].T + w["suffix.bias"]
        x = x + lemma + suffix + w["verb_emb.weight"][e["verb"]]
        heads, dk = self.cfg["heads"], d // self.cfg["heads"]
        count = e["N"]
        for li in range(self.cfg["layers"]):
            p = f"layers.{li}."
            h = _layer_norm(x, w[p + "n1.weight"], w[p + "n1.bias"])
            qkv = (h @ w[p + "qkv.weight"].T + w[p + "qkv.bias"]).reshape(count, 3, heads, dk)
            q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))  # heads, N, dk
            scores = q @ k.transpose(0, 2, 1) / math.sqrt(dk) + w[p + "bias.weight"][e["rel"]].transpose(2, 0, 1)
            attended = (_softmax(scores) @ v).transpose(1, 0, 2).reshape(count, d)
            x = x + attended @ w[p + "out.weight"].T + w[p + "out.bias"]
            h = _layer_norm(x, w[p + "n2.weight"], w[p + "n2.bias"])
            x = x + _gelu(h @ w[p + "ff.0.weight"].T + w[p + "ff.0.bias"]) @ w[p + "ff.3.weight"].T + w[p + "ff.3.bias"]
        return _layer_norm(x, w["enc_norm.weight"], w["enc_norm.bias"])

    # ------------------------------------------------------------------------------------------ decoder
    def _gru(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        w = self.w
        gi = x @ w["gru.weight_ih_l0"].T + w["gru.bias_ih_l0"]
        gh = h @ w["gru.weight_hh_l0"].T + w["gru.bias_hh_l0"]
        hs = gi.shape[-1] // 3
        r = _sigmoid(gi[..., :hs] + gh[..., :hs])
        z = _sigmoid(gi[..., hs:2 * hs] + gh[..., hs:2 * hs])
        n = np.tanh(gi[..., 2 * hs:] + r * gh[..., 2 * hs:])
        return ((1 - z) * n + z * h).astype(np.float32)

    def step_logits(self, state: np.ndarray, enc: np.ndarray, prev_type: int, prev_vec: np.ndarray,
                    prev_other: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One decoder step: (new state, ``comb``, raw [other | pointer] scores)."""
        w, d = self.w, self.d
        other_in = w["other_emb.weight"][prev_other]
        inp = w["type_emb.weight"][prev_type] + (prev_vec if prev_type == 3 else other_in)
        state = self._gru(inp @ w["dec_in.weight"].T + w["dec_in.bias"], state)
        ctx_scores = _softmax((enc @ (state @ w["q.weight"].T + w["q.bias"])) / math.sqrt(d))
        ctx = ctx_scores @ enc
        comb = np.tanh(np.concatenate([state, ctx]) @ w["comb.weight"].T + w["comb.bias"]).astype(np.float32)
        ptr = (enc @ (comb @ w["ptr_q.weight"].T + w["ptr_q.bias"])) / math.sqrt(d)
        other = comb @ w["other_head.weight"].T + w["other_head.bias"]
        return state, comb, np.concatenate([other, ptr]).astype(np.float32)

    def mask_actions(self, scores: np.ndarray, free: np.ndarray, punct_left: np.ndarray, need_left: np.ndarray,
                     prev_is_mark: bool, started: bool, used: np.ndarray, grp: np.ndarray) -> np.ndarray:
        """The masks of the training notebook (``mask_actions``) for one step."""
        n_p = self.n_punct
        out = scores.copy()
        remaining = int(free.sum())
        if not (remaining == 0 and int(need_left.sum()) == 0):
            out[0] = -1e4
        mark_ok = (punct_left > 0) & (not prev_is_mark and started and remaining > 0)
        out[1:1 + n_p] = np.where(mark_ok, out[1:1 + n_p], -1e4)
        out[1 + n_p:self.n_other] = np.where(need_left > 0, out[1 + n_p:self.n_other], -1e4)
        taken = (grp & used[None, :]).sum(-1)
        opened = (taken > 0) & (taken < grp.sum(-1))
        blocked = (opened[:, None] & ~grp).any(0)
        ptr = out[self.n_other:]
        out[self.n_other:] = np.where(~free | blocked, -1e4, ptr)
        return out

    def tag_logits(self, comb: np.ndarray, vec: np.ndarray, para_row: np.ndarray) -> np.ndarray:
        w = self.w
        h = _gelu(np.concatenate([comb, vec]) @ w["tag_head.0.weight"].T + w["tag_head.0.bias"])
        logits = h @ w["tag_head.2.weight"].T + w["tag_head.2.bias"]
        return np.where(para_row, logits, -1e4).astype(np.float32)

    def decode(self, e: dict, verbatim: frozenset[int] = frozenset()) -> tuple[list[tuple[str, int, int]], float, bool]:
        """Greedy decoding with the masks. Returns (steps, confidence, finished). A step is ("N", node, tag) |
        ("F", phrase, -1) | ("P", mark, -1) | ("EOS", 0, -1). The confidence is the minimum over the tag choices and the
        choices among closed-class phrases (never over the choice of the next node, i.e. the order), as in the notebook;
        the tag of a node the plan wants copied verbatim does not count."""
        enc = self.encode_nodes(e)
        count = e["N"]
        valid = ~e["silent"]
        used = np.zeros(count, dtype=bool)
        need_left = e["need"].copy()
        punct_left = e["punct_cap"].copy()
        state = np.tanh(enc.mean(0) @ self.w["init_h.weight"].T + self.w["init_h.bias"]).astype(np.float32)
        prev_type, prev_vec, prev_other = 4, np.zeros(self.d, np.float32), 0
        steps: list[tuple[str, int, int]] = []
        conf, finished = 1.0, False
        n_other, n_p = self.n_other, self.n_punct
        for step in range(self.cfg["max_steps"]):
            state, comb, scores = self.step_logits(state, enc, prev_type, prev_vec, prev_other)
            logits = self.mask_actions(scores, valid & ~used, punct_left, need_left, prev_type == 1, step > 0, used, e["grp"])
            best = int(np.argmax(logits))
            if 1 + n_p <= best < n_other:
                conf = min(conf, float(_softmax(logits[1 + n_p:n_other]).max()))
            if best == 0:
                steps.append(("EOS", 0, -1))
                finished = True
                break
            if best < 1 + n_p:
                steps.append(("P", best - 1, -1))
                punct_left[best - 1] -= 1
                prev_type, prev_vec, prev_other = 1, prev_vec, 1 + best
            elif best < n_other:
                steps.append(("F", best - 1 - n_p, -1))
                need_left[best - 1 - n_p] -= 1
                prev_type, prev_vec, prev_other = 2, prev_vec, 1 + best
            else:
                node = best - n_other
                vec = enc[node]
                tag_scores = self.tag_logits(comb, vec, e["para"][node])
                probs = _softmax(tag_scores)
                tag = int(np.argmax(tag_scores))
                if node not in verbatim:
                    conf = min(conf, float(probs.max()))
                steps.append(("N", node, tag))
                used[node] = True
                prev_type, prev_vec, prev_other = 3, vec, 0
        return steps, conf, finished

    # ------------------------------------------------------------------------------------------ realization
    def preposition_form(self, preposition: str, next_word: str) -> str:
        """в/во, с/со, к/ко ...: a decision list over the first letters of the next word, read from the corpus."""
        word = _norm(next_word)
        for prefix, form in self.prep_forms.get(preposition) or ():
            if word.startswith(prefix):
                return form
        return preposition_allomorph(preposition, word) or preposition

    def realize(self, tree: dict, steps: list[tuple[str, int, int]], words_of: dict[int, str]) -> list[str]:
        """Tokens of the decoded steps: the node words are in ``words_of``; the preposition and the marker of a node are
        put back in front of the first word of its phrase; inner marks and closed-class phrases stand where decoded."""
        clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
        flat: list[dict] = []
        heads: list[int | None] = []
        base = 0
        for clause in clauses:
            for node in clause["nodes"]:
                flat.append(node)
                heads.append(None if node["head"] is None else base + node["head"])
            base += len(clause["nodes"])
        tokens: list[str] = []
        owner: list[int | None] = []
        for kind, idx, _tag in steps:
            if kind == "N":
                tokens.append(words_of[idx])
                owner.append(idx)
            elif kind == "F":
                tokens.append(self.func_words[idx].partition("|")[2])
                owner.append(None)
            elif kind == "P":
                tokens.append(self.punct_marks[idx])
                owner.append(None)
        inserts: dict[int, list[tuple[str, str]]] = {}
        for k, node in enumerate(flat):
            if not node.get("preposition") and not node.get("marker"):
                continue
            members = _phrase_members(flat, heads, k)
            at = next((i for i, o in enumerate(owner) if o in members), None)
            if at is None:
                continue
            if node.get("marker"):
                inserts.setdefault(at, []).append(("marker", node["marker"]))
            if node.get("preposition"):
                inserts.setdefault(at, []).append(("preposition", node["preposition"]))
        out: list[str] = []
        for i, token in enumerate(tokens):
            for kind, text in inserts.get(i, ()):
                out.append(self.preposition_form(text, token) if kind == "preposition" else text)
            out.append(token)
        return out


def _phrase_members(flat: list[dict], heads: list[int | None], k: int) -> set[int]:
    members, stack = {k}, [k]
    while stack:
        x = stack.pop()
        for j, h in enumerate(heads):
            if (h == x and flat[j]["role"] in {"modifier", "postmodifier"} and not flat[j].get("preposition")
                    and not flat[j].get("marker")):
                members.add(j)
                stack.append(j)
    return members


def join_tokens(tokens: list[str]) -> str:
    """Words and marks into a string: «,;:» stick to the word before, a dash stands in spaces."""
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif token in _MARKS_LEFT:
            text += token
        else:
            text += " " + token
    return text


# ---------------------------------------------------------------------------------------------------- loading
_model_cache: dict[str, GrammarianModel | Exception] = {}


def artifact_dir() -> Path:
    raw = os.environ.get(ENV_DIR)
    return Path(raw) if raw else _ROOT / "data/models/grammarian"


def load_model(directory: Path | None = None) -> GrammarianModel | None:
    """The model of the artifact directory, loaded once; ``None`` (and the reason in ``load_error``) when unavailable."""
    path = directory or artifact_dir()
    key = str(path)
    if key not in _model_cache:
        try:
            _model_cache[key] = GrammarianModel(path)
        except Exception as exc:  # noqa: BLE001 - a missing / broken artifact must never break a reply
            _model_cache[key] = exc
    cached = _model_cache[key]
    return cached if isinstance(cached, GrammarianModel) else None


def warm_up() -> str:
    """Load the weights (and a first plan through the encoder) so that the first reply does not pay for it."""
    started = time.perf_counter()
    model = load_model()
    if model is None:
        return f"unavailable ({load_error()}); every call falls back to the deterministic realizer"
    return (f"{model.directory} loaded in {(time.perf_counter() - started) * 1000:.0f} ms, "
            f"threshold {model.threshold}, self gender {self_gender()}")


def load_error(directory: Path | None = None) -> str | None:
    cached = _model_cache.get(str(directory or artifact_dir()))
    return None if cached is None or isinstance(cached, GrammarianModel) else f"{type(cached).__name__}: {cached}"


# ---------------------------------------------------------------------------------------------------- the adapter
class Refusal(Exception):  # noqa: N818 - a refusal is a result, not an error
    """The plan cannot be given to the network (incomplete or unsupported): the reason is a short code."""


@dataclass(frozen=True, slots=True)
class GrammarianContext:
    """Facts about the speakers that the plan itself does not carry."""

    self_gender: str | None = None
    interlocutor_gender: str | None = None


def _agreement_needs_gender(thought, tree: dict) -> bool:
    """Does a word of this clause agree with its subject in gender (a past or conditional verb, an adjectival nominal
    part)? Only then is the gender of «я»/«ты» needed."""
    clause = tree["clause"]
    if clause["tense"] == "past" or clause["mood"] == "conditional":
        if thought.predicate.part_of_speech == "verb" and clause["predicate_class"] != "copular":
            return True
        if clause["predicate_class"] == "copular":
            return True
    return any(n["role"] == "predicate_complement" and n["pos"] in {"adjective", "participle"} for n in tree["nodes"])


def _surface_features(lexeme, pos: str) -> dict[str, str]:
    """Degree / short form of an adjective or an adverb, read from its OBSERVED surface by morphology."""
    if pos not in {"adjective", "participle", "adverb"} or not lexeme.surface:
        return {}
    surface = lexeme.surface.lower()
    readings = _analyzer().parse(surface)
    own = [r for r in readings if _norm(r.normal_form) == _norm(lexeme.lemma)]
    if not own:
        return {}
    grammemes = set(own[0].tag.grammemes)
    out: dict[str, str] = {}
    if "COMP" in grammemes:
        out["degree"] = "comparative"
    if grammemes & {"ADJS", "PRTS"}:
        out["form"] = "short"
    return out


def plan_to_tree(thought, context: GrammarianContext, model: GrammarianModel) -> tuple[dict, dict]:
    """``ClauseThought`` -> (v3 input tree, side info). Raises ``Refusal`` for an incomplete / unsupported plan."""
    fz = model.fz
    tree = thought_to_tree(thought)
    clause = tree["clause"]
    nodes = tree["nodes"]
    if len(nodes) > model.cfg["max_nodes"]:
        raise Refusal("too_many_nodes")
    if clause["predicate_class"] not in fz["pclass"]:
        raise Refusal(f"predicate_class:{clause['predicate_class']}")
    if clause["predicate_class"] == "copular" and thought.predicate.lemma != "быть":
        raise Refusal("copular_not_be")  # the net drops a copula of the present tense: only «быть» is silent
    if clause["move"] not in fz["move"]:
        raise Refusal(f"move:{clause['move']}")
    func: list[list[str]] = []
    if thought.negated:
        func.append(["particle_negation", _NEG_WORD])
    if clause["mood"] == "conditional":
        func.append(["particle_conditional", _COND_WORD])
    clause["func"] = func
    clause["end"] = "?" if clause["move"].endswith("question") else "."
    verbatim: set[int] = set()
    proper: set[int] = set()
    # lexemes by node: predicate first, then every argument's head, modifiers, post-modifiers in the order of the tree
    lexemes: dict[int, object] = {0: thought.predicate}
    order = sorted(
        enumerate(thought.dependencies),
        key=lambda item: (item[1].head is not None, _role_rank(item[1].role), item[0]),
    )
    cursor = 1
    node_role_arg: dict[int, str] = {}
    for _, dependency in order:
        argument = dependency.argument
        lexemes[cursor] = argument.head
        node_role_arg[cursor] = dependency.role
        cursor += 1
        for mod in argument.modifiers:
            lexemes[cursor], node_role_arg[cursor] = mod, dependency.role
            cursor += 1
        for post in argument.postmodifiers:
            lexemes[cursor], node_role_arg[cursor] = post, dependency.role
            verbatim.add(cursor)  # a post-modifier is a surface the plan already inflected: copied as it stands
            cursor += 1
    if cursor != len(nodes):
        raise Refusal("tree_mismatch")
    be_present = thought.predicate.lemma == "быть" and clause["tense"] == "present"
    for node in nodes:
        lexeme = lexemes[node["id"]]
        if node["role"] not in fz["role"]:
            raise Refusal(f"unknown_role:{node['role']}")
        if node["pos"] not in fz["pos"]:
            raise Refusal(f"unsupported_pos:{node['pos']}")
        if node["id"] not in verbatim and node["role"] != "predicate":
            features = dict(lexeme.features)
            reading = DEFAULT_MORPHOLOGY.best(node["lemma"])
            is_name = bool(reading and reading.is_known and {"Name", "Surn"} & reading.grammemes
                           and not lexeme.surface.startswith("«"))
            if is_name:
                proper.add(node["id"])
            elif lexeme.entity_kind is not None or features.get("indeclinable") == "true":
                verbatim.add(node["id"])  # a quoted name / an indeclinable word: the plan's surface, untouched
        if node["role"] == "comparison":
            node["marker_kind"] = "comparison"
            node.setdefault("marker", _COMPARISON_DEFAULT)
        head_node = nodes[node["head"]] if node["head"] not in (None, 0) else None
        if (
            node["role"] == "direct_object" and head_node is not None and head_node["role"] == "action_complement"
            and _nominal_complement_case(head_node["lemma"]) == "instrumental"
        ):
            # the noun of «хотел бы быть собакой» is the nominal part of an embedded intransitive verb (the corpus
            # tree calls it ``predicate_complement``), whatever the reader named it; its case is the nominal part's
            node["role"] = "predicate_complement"
        if node["role"] == "predicate_complement" and not node.get("case_constraint"):
            # the nominal part: nominative of a present «быть» (the null copula), instrumental otherwise
            node["case_constraint"] = "nominative" if be_present and node["head"] == 0 else "instrumental"
        node["lex"].update(_surface_features(lexeme, node["pos"]))
        if ("elided", "true") in lexeme.features and node["lemma"] not in {"я", "ты", "мы", "вы"}:
            raise Refusal("elided_subject_not_a_speaker")  # the net is trained to drop only a speaker's pronoun
        if node["lemma"] != lexeme.lemma:
            raise Refusal("tree_mismatch")
    # the gender of я / ты: a fact about the speaker
    if _agreement_needs_gender(thought, tree):
        subject = next((n for n in nodes if n["role"] == "subject"), None)
        if subject is not None and subject["lemma"] in {"я", "ты"} and subject["pos"] == "pronoun":
            given = dict(lexemes[subject["id"]].features).get("gender")
            gender = given or (context.self_gender if subject["lemma"] == "я" else context.interlocutor_gender)
            if gender not in {"masculine", "feminine", "neuter"}:
                raise Refusal("speaker_gender_unknown")
            subject["lex"]["gender"] = gender
    return tree, {"verbatim": frozenset(verbatim), "proper": frozenset(proper), "lexemes": lexemes,
                  "node_role_arg": node_role_arg}


_ROLE_RANK = ("subject", "indirect_object", "direct_object", "predicate_complement", "action_complement", "manner",
              "cause", "location", "topic", "comparison")


def _role_rank(role: str) -> int:
    return _ROLE_RANK.index(role) if role in _ROLE_RANK else 99


@dataclass(frozen=True, slots=True)
class GrammarianAnswer:
    body: str  # the words and inner marks, no sentence ends (``productive_nlg`` closes the sentence)
    role_order: tuple[str, ...]
    confidence: float
    record: GrammarianRecord


def _speaker_agreement_broken(tree: dict, steps, model: GrammarianModel) -> bool:
    """Fidelity to the plan: when the plan gave the gender of «я»/«ты», a past verb or an adjectival nominal part that
    agrees with it must carry that gender (the network can pick another one with full confidence)."""
    subject = next((n for n in tree["nodes"] if n["role"] == "subject" and n["lex"].get("gender")), None)
    if subject is None or subject["lemma"] not in {"я", "ты"}:
        return False
    wanted = {"masculine": "masc", "feminine": "femn", "neuter": "neut"}.get(subject["lex"]["gender"])
    for kind, idx, tag in steps:
        node = tree["nodes"][idx] if kind == "N" else None
        if node is None or node["role"] not in {"predicate", "predicate_complement"}:
            continue
        if node["pos"] not in {"verb", "adjective", "participle"}:
            continue
        grams = set(model.tags[tag].split(","))
        genders = grams & {"masc", "femn", "neut"}
        if "sing" in grams and genders and wanted not in genders:
            return True
    return False


def _focus_not_fronted(thought, tree: dict, steps, info: dict) -> bool:
    """``focus_role`` of a question whose focus holds a WH word is an explicit instruction of the plan (the deterministic
    realizer fronts it too): an output that leaves the WH argument in situ is not the plan's wording."""
    if thought.clause_kind != "question" or not thought.focus_role:
        return False
    focus = next((d.argument for d in thought.dependencies if d.role == thought.focus_role), None)
    if focus is None or not any(x.lemma in _QUESTION_FOCUS_LEMMAS for x in (focus.head, *focus.modifiers)):
        return False
    first = steps[0] if steps else None
    if first is None or first[0] != "N":
        return True
    return info["node_role_arg"].get(first[1]) != thought.focus_role


def _role_order(thought, tree: dict, steps, info: dict, model: GrammarianModel) -> tuple[str, ...]:
    """Roles in the order of the output; a silent role (a pro-dropped subject, a null copula) keeps its default place
    among the roles the plan has (the contract of ``ProductiveSurface.role_order``)."""
    roles: list[str] = []
    for kind, idx, _ in steps:
        role = None
        if kind == "N":
            role = "predicate" if idx == 0 else info["node_role_arg"].get(idx, tree["nodes"][idx]["role"])
        elif kind == "F" and model.func_words[idx].startswith("particle_negation|"):
            role = "negation"
        if role is not None and role not in roles:
            roles.append(role)
    present = ["predicate", *(d.role for d in thought.dependencies), *(("negation",) if thought.negated else ())]
    default = ("subject", "negation", "predicate", "action_complement", "predicate_complement", "direct_object",
               "indirect_object", "location", "topic", "manner", "comparison")
    rank = {role: i for i, role in enumerate(default)}
    for role in sorted((r for r in present if r not in roles), key=lambda r: rank.get(r, 99)):
        later = next((i for i, have in enumerate(roles) if rank.get(have, 99) > rank.get(role, 99)), None)
        roles.insert(len(roles) if later is None else later, role)
    return tuple(roles)


_CACHE_SIZE = 4096
_cache: OrderedDict = OrderedDict()


def realize(thought, *, context: GrammarianContext | None = None, model: GrammarianModel | None = None
            ) -> tuple[GrammarianAnswer | None, GrammarianRecord]:
    """Try to word ``thought`` with the network. Returns (answer or None, record). Never raises.

    One reply realizes the same plan many times (hundreds of calls for tens of distinct plans: LAST_MESSAGE40 / ``calls.json``), so the
    answer for a plan is remembered (a small LRU, the plan is a frozen value); a repeated call is journaled with ``cache: true``.
    """
    model = model or load_model()
    context = context or GrammarianContext()
    key = None
    if model is not None:
        try:
            key = (thought, context.self_gender or self_gender(), context.interlocutor_gender, id(model), model.threshold)
            hash(key)
        except TypeError:
            key = None
    if key is not None and key in _cache:
        answer, record = _cache[key]
        _cache.move_to_end(key)
        hit = GrammarianRecord(record.path, record.reason, record.confidence, record.threshold, 0.0, record.predicate,
                               record.text, {**record.details, "cache": True})
        _journal(hit)
        STATS["cache_hit"] += 1
        return answer, hit
    answer, record = _realize(thought, context, model)
    if key is not None:
        _cache[key] = (answer, record)
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return answer, record


def _realize(thought, context: GrammarianContext, model: GrammarianModel | None
             ) -> tuple[GrammarianAnswer | None, GrammarianRecord]:
    started = time.perf_counter()
    threshold = model.threshold if model is not None else 1.0
    predicate = getattr(getattr(thought, "predicate", None), "lemma", "") or ""

    def done(path: str, reason: str | None, confidence: float | None, text: str | None = None,
             **details) -> GrammarianRecord:
        record = GrammarianRecord(path, reason, confidence, threshold, (time.perf_counter() - started) * 1000,
                                  predicate, text, details)
        _journal(record)
        return record

    if model is None:
        return None, done("fallback", "model_unavailable", None, error=load_error())
    if context.self_gender is None:
        context = GrammarianContext(self_gender(), context.interlocutor_gender)
    try:
        tree, info = plan_to_tree(thought, context, model)
        encoded = model.encode(tree)
        if encoded is None:
            return None, done("fallback", "too_many_nodes", None)
        if encoded["need_unknown"]:
            return None, done("fallback", "closed_class_word_not_in_vocabulary", None)
        steps, confidence, finished = model.decode(encoded, info["verbatim"])
        if not finished:
            return None, done("fallback", "no_end_of_sentence", confidence)
        if confidence < threshold:
            return None, done("fallback", "low_confidence", confidence, steps=len(steps))
        words: dict[int, str] = {}
        for kind, idx, tag in steps:
            if kind != "N":
                continue
            node = tree["nodes"][idx]
            lexeme = info["lexemes"][idx]
            if idx in info["verbatim"]:
                word = lexeme.surface
            else:
                word = inflect(node["lemma"], _UPOS.get(node["pos"], "NOUN"), model.tags[tag],
                               bool(node.get("preposition")))
            if not word:
                return None, done("fallback", "form_unavailable", confidence, lemma=node["lemma"], tag=model.tags[tag])
            if idx in info["proper"]:
                word = word[:1].upper() + word[1:]
            words[idx] = word
        tokens = model.realize(tree, steps, words)
        body = join_tokens(tokens)
        if not body:
            return None, done("fallback", "empty", confidence)
        if _speaker_agreement_broken(tree, steps, model):
            return None, done("fallback", "speaker_gender_disagreement", confidence)
        if _focus_not_fronted(thought, tree, steps, info):
            return None, done("fallback", "question_focus_not_fronted", confidence)
        order = _role_order(thought, tree, steps, info, model)
        record = done("network", None, confidence, body)
        return GrammarianAnswer(body, order, confidence, record), record
    except Refusal as refusal:
        return None, done("fallback", str(refusal), None)
    except Exception as exc:  # noqa: BLE001 - the network must never break a reply
        return None, done("fallback", f"error:{type(exc).__name__}", None, message=str(exc)[:200])
