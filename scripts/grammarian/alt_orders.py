# ruff: noqa: E501
"""Alternative references (v2 format): other orders of a phrase's units that the bot's own readers read
back into the SAME tree. Mechanical, no hand-written text: the permutation is accepted only when
``own_stream.build_item(text)`` gives a tree with the signature of the row's tree.

Used for the bench («own» slice) and, since stage 4.27, for dev / held-out rows too (a corpus has one
reference but a free word order has many correct ones). Only one-clause rows without inner marks and with
2-5 movable units are permuted; the target of an alternative is a target in the v2 format.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict

import compare_paths
import ud_to_tree as u

MAX_PERMUTED_UNITS = 5
PERMUTATIONS = 24


def digest(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def words(text: str) -> list[str]:
    """The words of a phrase for every comparison of the grammarian (one function, reference and output alike).

    A dash is punctuation, not a word (stage 4.29: «-» / «--» used to count as words, so a reference with the
    dash of a chat line, or «бизнес - сообщества» against «бизнес-сообщества», failed against itself in 6.8 % of
    dev rows). A hyphenated word is compared by its parts, because the corpus tokenizer splits it («кто - то»)."""
    cleaned = re.sub(r"[^\w\s]|_", " ", text.casefold().replace("ё", "е"))
    return cleaned.split()


def units_of(tree: dict, target: list) -> list[list[int]]:
    """Positions of the target grouped into movable units (a top-level argument with its words)."""
    nodes = {n["id"]: n for n in tree["nodes"]}

    def top(node_id: int) -> int:
        while nodes[node_id]["head"] not in (None, 0):
            node_id = nodes[node_id]["head"]
        return node_id

    groups: dict[int, list[int]] = defaultdict(list)
    for position, action in enumerate(target):
        if action[0] == "C":
            groups[top(action[1])].append(position)
        else:
            kind, _, node = action[2].partition(":")
            groups[0 if kind.startswith("particle") else top(int(node))].append(position)
    return [groups[key] for key in sorted(groups)]


def alternative_orders(tree: dict, target: list, reference: str) -> list[tuple[str, list]]:
    """(text, target) alternatives of one row; ``[]`` when the row cannot be permuted."""
    import own_stream

    if tree["kind"] != "clause" or any(a[0] == "P" for a in target):
        return []
    # a multi-word closed-class phrase or a preposition keeps its place relative to its node: units are
    # built from node groups, the closed-class words travel with their licence owner
    units = units_of(tree, target)
    if not 2 <= len(units) <= MAX_PERMUTED_UNITS:
        return []
    original = tuple(range(len(units)))
    orders = [p for p in itertools.permutations(original) if p != original]
    orders.sort(key=lambda p: digest(reference + str(p)))
    reference_signature = compare_paths.signature(tree)
    accepted: list[tuple[str, list]] = []
    seen = {" ".join(words(reference))}
    for order in orders[:PERMUTATIONS]:
        actions = [target[position] for index in order for position in units[index]]
        text = u.target_text(actions, tree)
        if "?" in text.split()[:-1]:
            continue
        key = " ".join(words(text))
        if key in seen:
            continue
        try:
            read = own_stream.build_item(text)
        except Exception:  # noqa: BLE001 - unreadable order: not a second reference
            continue
        if compare_paths.signature(read.tree) == reference_signature:
            seen.add(key)
            accepted.append((text, actions))
    return accepted
