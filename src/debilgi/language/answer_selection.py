"""Choose the amount of wording for a selected proposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from debilgi.language.generative_planner import DialogueObjective
from debilgi.language.productive_nlg import ClauseThought


@dataclass(frozen=True, slots=True)
class AnswerShape:
    mode: str
    focus_role: str | None
    reason: tuple[str, ...]


def select_answer_shape(objective: DialogueObjective, thought: ClauseThought) -> AnswerShape:
    """Select sentence vs focused phrase after meaning has been selected.

    The decision uses the reader's slot and proposition complexity. It never
    changes the claim or fabricates a shorter claim when a role is absent.
    """
    if objective.kind != "slot" or objective.question_word not in {"кто", "что", "где", "куда"}:
        return AnswerShape("sentence", None, ("content_requires_context",))
    roles = {dependency.role for dependency in thought.dependencies}
    focus = {"кто": "subject", "что": "direct_object", "где": "location", "куда": "location"}[
        objective.question_word
    ]
    if focus not in roles:
        return AnswerShape("sentence", None, ("answer_role_not_present",))
    if len(thought.dependencies) > 3:
        return AnswerShape("sentence", None, ("proposition_has_multiple_relations",))
    return AnswerShape("fragment", focus, ("reader_slot_is_focused", "claim_can_be_shortened"))
