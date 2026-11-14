"""Turn fully grounded dialogue-task changes into semantic plans.

Control tasks such as evidence search never become user-facing assertions.  They
guide corpus search and arbitration until actual language or a proposition can
support an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.cognition.dialogue_tasks import DialogueTask
from debilgi.language.communication import MessagePlan, ReaderGoal
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)


@dataclass(frozen=True, slots=True)
class TaskResolution:
    plan: MessagePlan
    operation: str
    evidence_text: str


def plan_task_resolution(task: DialogueTask, goal: ReaderGoal) -> TaskResolution | None:
    """Produce a justified semantic move when ordinary evidence search is empty."""

    if task.goal == "perform_utterance":
        # The requested utterance is an action payload rather than a proposition.
        # It is admitted separately with exact task-effect verification.
        return None
    if task.goal == "repair_shared_model":
        revisions = dict(task.objective.revisions)
        predicate = revisions.get("predicate")
        revised_object = revisions.get("object")
        if predicate:
            dependencies = [
                SemanticDependency(
                    "subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))
                )
            ]
            if revised_object:
                dependencies.append(
                    SemanticDependency(
                        "direct_object",
                        ConceptArgument(
                            ConceptLexeme(
                                revised_object,
                                f"«{revised_object}»",
                                "noun",
                                entity_kind="quoted_claim",
                            ),
                            grammatical_case="accusative",
                        ),
                    )
                )
            thought = ClauseThought(
                ConceptLexeme(
                    predicate,
                    predicate,
                    "verb",
                    (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
                ),
                tuple(dependencies),
                confidence=task.confidence,
                evidence=(task.task_id, task.source_event_id),
            )
            retracted = replace(
                thought,
                dependencies=(
                    SemanticDependency(
                        "subject", ConceptArgument(ConceptLexeme("ты", "ты", "pronoun"))
                    ),
                    *thought.dependencies[1:],
                ),
                negated=True,
            )
            plan = MessagePlan(goal, thought, additional_claims=(retracted,))
        else:
            return None
        return TaskResolution(
            plan,
            "task_repair_shared_model",
            "A later turn contradicts the agent's current account of the dialogue.",
        )
    # Internal control states and incomplete directives are search constraints,
    # not conversational claims.  Returning no plan routes them to learned
    # language generation, where a candidate still has to advance the task.
    return None
