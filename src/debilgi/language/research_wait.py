"""A pending reference lookup is resumable work, not a failed reply."""

from __future__ import annotations


class LexicalResearchPending(Exception):
    def __init__(self, terms: tuple[str, ...], retry_after: int):
        self.terms = terms
        self.retry_after = max(1, min(300, retry_after))
        super().__init__("reference_pending:" + ",".join(terms))
