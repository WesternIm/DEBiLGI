import pytest

from debilgi.diagnostics.progress import (
    ComprehensionMetrics,
    MetricComparison,
    PipelineMetrics,
    PredictionMetrics,
    ProgressVerdict,
    _comprehension_metrics,
    _feedback_metrics,
    _pipeline_metrics,
    _prediction_metrics,
    _verdict,
)


def test_feedback_metrics_measure_reward_rates_and_calibration() -> None:
    metrics = _feedback_metrics(
        [
            {"reward": 1.0, "expected_reward": 0.5},
            {"reward": -0.5, "expected_reward": 0.0},
            {"reward": 0.0, "expected_reward": 0.0},
        ]
    )

    assert metrics.samples == 3
    assert metrics.average_reward == pytest.approx(1 / 6)
    assert metrics.positive_rate == pytest.approx(1 / 3)
    assert metrics.negative_rate == pytest.approx(1 / 3)
    assert metrics.calibration_error == pytest.approx(1 / 3)


def test_prediction_metrics_measure_exact_accuracy_and_difference() -> None:
    metrics = _prediction_metrics(
        [
            {"predicted_outcome": "reply", "actual_outcome": "reply", "difference": 0.1},
            {
                "predicted_outcome": "reaction",
                "actual_outcome": "no_response",
                "difference": 0.7,
            },
        ]
    )

    assert metrics.samples == 2
    assert metrics.accuracy == 0.5
    assert metrics.average_difference == pytest.approx(0.4)


def test_comprehension_metrics_measure_coverage_and_unknown_terms() -> None:
    metrics = _comprehension_metrics(
        [
            {
                "comprehension_score": 0.8,
                "sufficient": 1,
                "unknown_terms_json": "[]",
            },
            {
                "comprehension_score": 0.4,
                "sufficient": 0,
                "unknown_terms_json": '["трошка","фиговина"]',
            },
        ]
    )

    assert metrics.samples == 2
    assert metrics.average_score == pytest.approx(0.6)
    assert metrics.sufficient_rate == 0.5
    assert metrics.average_unknown_terms == 1.0


def test_pipeline_metrics_do_not_count_intentional_silence_as_failure() -> None:
    metrics = _pipeline_metrics(
        [
            {"outcome_type": "replied"},
            {"outcome_type": "failed_to_form_plan"},
            {"outcome_type": "intentional_silence"},
        ]
    )

    assert metrics.samples == 2
    assert metrics.intentional_silences == 1
    assert metrics.reply_rate == 0.5
    assert metrics.technical_failure_rate == 0.5


def test_verdict_requires_two_comparable_sections() -> None:
    feedback = MetricComparison(
        current=_feedback_metrics([{"reward": 1.0, "expected_reward": 1.0}] * 10),
        previous=_feedback_metrics([{"reward": 0.0, "expected_reward": 0.5}] * 10),
    )
    empty_predictions = MetricComparison(
        PredictionMetrics(0, None, None),
        PredictionMetrics(0, None, None),
    )
    empty_comprehension = MetricComparison(
        ComprehensionMetrics(0, None, None, None),
        ComprehensionMetrics(0, None, None, None),
    )
    empty_pipeline = MetricComparison(
        PipelineMetrics(0, 0, None, None),
        PipelineMetrics(0, 0, None, None),
    )

    verdict, _, sections = _verdict(
        feedback,
        empty_comprehension,
        empty_predictions,
        empty_pipeline,
        minimum_samples=10,
    )

    assert verdict is ProgressVerdict.INSUFFICIENT_DATA
    assert sections == 1


def test_verdict_detects_improvement_across_independent_sections() -> None:
    feedback = MetricComparison(
        current=_feedback_metrics([{"reward": 0.8, "expected_reward": 0.7}] * 10),
        previous=_feedback_metrics([{"reward": 0.1, "expected_reward": -0.4}] * 10),
    )
    predictions = MetricComparison(
        current=PredictionMetrics(10, 0.8, 0.2),
        previous=PredictionMetrics(10, 0.4, 0.6),
    )
    empty_comprehension = MetricComparison(
        ComprehensionMetrics(0, None, None, None),
        ComprehensionMetrics(0, None, None, None),
    )
    pipeline = MetricComparison(
        current=PipelineMetrics(10, 0, 0.9, 0.1),
        previous=PipelineMetrics(10, 0, 0.6, 0.4),
    )

    verdict, score, sections = _verdict(
        feedback,
        empty_comprehension,
        predictions,
        pipeline,
        minimum_samples=10,
    )

    assert verdict is ProgressVerdict.IMPROVING
    assert score == 5
    assert sections == 3
