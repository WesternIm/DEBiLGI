"""Comparable learning metrics for recent and preceding observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import Any

from debilgi.storage.database import Database


class ProgressVerdict(StrEnum):
    IMPROVING = "improving"
    PLATEAU = "plateau"
    REGRESSING = "regressing"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class FeedbackMetrics:
    samples: int
    average_reward: float | None
    positive_rate: float | None
    negative_rate: float | None
    calibration_error: float | None


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    samples: int
    accuracy: float | None
    average_difference: float | None


@dataclass(frozen=True, slots=True)
class ComprehensionMetrics:
    samples: int
    average_score: float | None
    sufficient_rate: float | None
    average_unknown_terms: float | None


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    samples: int
    intentional_silences: int
    reply_rate: float | None
    technical_failure_rate: float | None


@dataclass(frozen=True, slots=True)
class QualityLoopStats:
    queued_cases: int
    monitoring_cases: int
    resolved_cases: int
    exhausted_cases: int
    applied_actions: int
    queued_actions: int


@dataclass(frozen=True, slots=True)
class MetricComparison:
    current: Any
    previous: Any


@dataclass(frozen=True, slots=True)
class LearningProgressReport:
    window_size: int
    minimum_samples: int
    feedback: MetricComparison
    comprehension: MetricComparison
    predictions: MetricComparison
    pipeline: MetricComparison
    quality_loop: QualityLoopStats
    verdict: ProgressVerdict
    evidence_score: int
    comparable_sections: int


class LearningProgressAnalyzer:
    """Compare equally sized recent windows instead of cumulative counters.

    Cumulative totals demonstrate activity, not improvement. This analyzer
    deliberately keeps feedback, world-model predictions and reply-pipeline
    reliability separate, then emits a conservative operational trend.
    """

    def __init__(self, database: Database, *, minimum_samples: int = 10) -> None:
        if minimum_samples < 2:
            raise ValueError("minimum_samples must be at least 2")
        self._database = database
        self._minimum_samples = minimum_samples

    async def report(self, guild_id: int, *, window_size: int = 25) -> LearningProgressReport:
        window_size = max(self._minimum_samples, min(window_size, 100))
        limit = window_size * 2
        feedback_rows = await self._database.fetchall(
            """SELECT reward, expected_reward
               FROM language_utterance_experiences
               WHERE guild_id = ? AND reward IS NOT NULL
               ORDER BY COALESCE(resolved_at, created_at) DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        prediction_rows = await self._database.fetchall(
            """SELECT predicted_outcome, actual_outcome, difference
               FROM cognition_expectations
               WHERE guild_id = ? AND status = 'resolved'
                 AND actual_outcome IS NOT NULL AND difference IS NOT NULL
               ORDER BY COALESCE(resolved_at, created_at) DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        comprehension_rows = await self._database.fetchall(
            """SELECT comprehension_score, sufficient, unknown_terms_json
               FROM language_utterance_understandings
               WHERE guild_id = ?
               ORDER BY observed_at DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        pipeline_rows = await self._database.fetchall(
            """SELECT outcome_type
               FROM language_processing_outcomes
               WHERE guild_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (guild_id, limit),
        )
        quality_row = await self._database.fetchone(
            """SELECT
                   SUM(status = 'queued') AS queued,
                   SUM(status = 'monitoring') AS monitoring,
                   SUM(status = 'resolved') AS resolved,
                   SUM(status = 'exhausted') AS exhausted
               FROM learning_quality_cases WHERE guild_id = ?""",
            (guild_id,),
        )
        action_row = await self._database.fetchone(
            """SELECT
                   SUM(action.status = 'applied') AS applied,
                   SUM(action.status = 'queued') AS queued
               FROM learning_quality_actions AS action
               JOIN learning_quality_cases AS quality_case
                 ON quality_case.case_id = action.case_id
               WHERE quality_case.guild_id = ?""",
            (guild_id,),
        )

        feedback = _comparison(feedback_rows, window_size, _feedback_metrics)
        comprehension = _comparison(
            comprehension_rows, window_size, _comprehension_metrics
        )
        predictions = _comparison(prediction_rows, window_size, _prediction_metrics)
        pipeline = _comparison(pipeline_rows, window_size, _pipeline_metrics)
        quality_loop = QualityLoopStats(
            queued_cases=int(quality_row["queued"] or 0) if quality_row else 0,
            monitoring_cases=int(quality_row["monitoring"] or 0) if quality_row else 0,
            resolved_cases=int(quality_row["resolved"] or 0) if quality_row else 0,
            exhausted_cases=int(quality_row["exhausted"] or 0) if quality_row else 0,
            applied_actions=int(action_row["applied"] or 0) if action_row else 0,
            queued_actions=int(action_row["queued"] or 0) if action_row else 0,
        )
        verdict, evidence_score, sections = _verdict(
            feedback,
            comprehension,
            predictions,
            pipeline,
            minimum_samples=self._minimum_samples,
        )
        return LearningProgressReport(
            window_size=window_size,
            minimum_samples=self._minimum_samples,
            feedback=feedback,
            comprehension=comprehension,
            predictions=predictions,
            pipeline=pipeline,
            quality_loop=quality_loop,
            verdict=verdict,
            evidence_score=evidence_score,
            comparable_sections=sections,
        )


def _comparison(rows: list[Any], size: int, calculate: Any) -> MetricComparison:
    return MetricComparison(
        current=calculate(rows[:size]),
        previous=calculate(rows[size : size * 2]),
    )


def _feedback_metrics(rows: list[Any]) -> FeedbackMetrics:
    rewards = [float(row["reward"]) for row in rows]
    if not rewards:
        return FeedbackMetrics(0, None, None, None, None)
    calibration = [
        abs(float(row["reward"]) - float(row["expected_reward"])) for row in rows
    ]
    return FeedbackMetrics(
        samples=len(rewards),
        average_reward=fmean(rewards),
        positive_rate=sum(reward > 0.0 for reward in rewards) / len(rewards),
        negative_rate=sum(reward < 0.0 for reward in rewards) / len(rewards),
        calibration_error=fmean(calibration),
    )


def _prediction_metrics(rows: list[Any]) -> PredictionMetrics:
    if not rows:
        return PredictionMetrics(0, None, None)
    return PredictionMetrics(
        samples=len(rows),
        accuracy=(
            sum(str(row["predicted_outcome"]) == str(row["actual_outcome"]) for row in rows)
            / len(rows)
        ),
        average_difference=fmean(float(row["difference"]) for row in rows),
    )


def _comprehension_metrics(rows: list[Any]) -> ComprehensionMetrics:
    if not rows:
        return ComprehensionMetrics(0, None, None, None)
    unknown_counts = []
    for row in rows:
        try:
            unknowns = json.loads(str(row["unknown_terms_json"]))
        except (TypeError, ValueError):
            unknowns = []
        unknown_counts.append(len(unknowns) if isinstance(unknowns, list) else 0)
    return ComprehensionMetrics(
        samples=len(rows),
        average_score=fmean(float(row["comprehension_score"]) for row in rows),
        sufficient_rate=sum(bool(row["sufficient"]) for row in rows) / len(rows),
        average_unknown_terms=fmean(unknown_counts),
    )


def _pipeline_metrics(rows: list[Any]) -> PipelineMetrics:
    outcomes = [str(row["outcome_type"]) for row in rows]
    actionable = [outcome for outcome in outcomes if outcome != "intentional_silence"]
    silences = len(outcomes) - len(actionable)
    if not actionable:
        return PipelineMetrics(0, silences, None, None)
    replied = sum(outcome == "replied" for outcome in actionable)
    return PipelineMetrics(
        samples=len(actionable),
        intentional_silences=silences,
        reply_rate=replied / len(actionable),
        technical_failure_rate=(len(actionable) - replied) / len(actionable),
    )


def _verdict(
    feedback: MetricComparison,
    comprehension: MetricComparison,
    predictions: MetricComparison,
    pipeline: MetricComparison,
    *,
    minimum_samples: int,
) -> tuple[ProgressVerdict, int, int]:
    section_scores: list[int] = []
    if _comparable(feedback, minimum_samples):
        reward_change = feedback.current.average_reward - feedback.previous.average_reward
        calibration_gain = (
            feedback.previous.calibration_error - feedback.current.calibration_error
        )
        section_scores.append(_direction(reward_change, 0.05) + _direction(calibration_gain, 0.05))
    if _comparable(comprehension, minimum_samples):
        score_change = comprehension.current.average_score - comprehension.previous.average_score
        sufficient_change = (
            comprehension.current.sufficient_rate - comprehension.previous.sufficient_rate
        )
        unknown_gain = (
            comprehension.previous.average_unknown_terms
            - comprehension.current.average_unknown_terms
        )
        section_scores.append(
            _direction(score_change, 0.05)
            + _direction(sufficient_change, 0.05)
            + _direction(unknown_gain, 0.20)
        )
    if _comparable(predictions, minimum_samples):
        accuracy_change = predictions.current.accuracy - predictions.previous.accuracy
        difference_gain = (
            predictions.previous.average_difference - predictions.current.average_difference
        )
        section_scores.append(
            _direction(accuracy_change, 0.05) + _direction(difference_gain, 0.05)
        )
    if (
        _comparable(pipeline, minimum_samples)
        and pipeline.current.technical_failure_rate is not None
        and pipeline.previous.technical_failure_rate is not None
    ):
        failure_gain = (
            pipeline.previous.technical_failure_rate
            - pipeline.current.technical_failure_rate
        )
        section_scores.append(_direction(failure_gain, 0.03))

    if len(section_scores) < 2:
        return ProgressVerdict.INSUFFICIENT_DATA, sum(section_scores), len(section_scores)
    score = sum(section_scores)
    if score >= 2:
        return ProgressVerdict.IMPROVING, score, len(section_scores)
    if score <= -2:
        return ProgressVerdict.REGRESSING, score, len(section_scores)
    return ProgressVerdict.PLATEAU, score, len(section_scores)


def _comparable(comparison: MetricComparison, minimum_samples: int) -> bool:
    return (
        comparison.current.samples >= minimum_samples
        and comparison.previous.samples >= minimum_samples
    )


def _direction(change: float, threshold: float) -> int:
    if change >= threshold:
        return 1
    if change <= -threshold:
        return -1
    return 0
