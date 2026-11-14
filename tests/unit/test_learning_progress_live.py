from types import SimpleNamespace

import debilgi.discord.commands as commands_module
from debilgi.diagnostics.progress import (
    ComprehensionMetrics,
    FeedbackMetrics,
    LearningProgressReport,
    MetricComparison,
    PipelineMetrics,
    PredictionMetrics,
    ProgressVerdict,
    QualityLoopStats,
)
from debilgi.discord.commands import Diagnostics


def _empty_report() -> LearningProgressReport:
    return LearningProgressReport(
        window_size=25,
        minimum_samples=10,
        feedback=MetricComparison(
            FeedbackMetrics(0, None, None, None, None),
            FeedbackMetrics(0, None, None, None, None),
        ),
        comprehension=MetricComparison(
            ComprehensionMetrics(0, None, None, None),
            ComprehensionMetrics(0, None, None, None),
        ),
        predictions=MetricComparison(
            PredictionMetrics(0, None, None),
            PredictionMetrics(0, None, None),
        ),
        pipeline=MetricComparison(
            PipelineMetrics(0, 0, None, None),
            PipelineMetrics(0, 0, None, None),
        ),
        quality_loop=QualityLoopStats(0, 0, 0, 0, 0, 0),
        verdict=ProgressVerdict.INSUFFICIENT_DATA,
        evidence_score=0,
        comparable_sections=0,
    )


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0

    async def report(self, guild_id: int, *, window_size: int) -> LearningProgressReport:
        assert guild_id == 1
        assert window_size == 25
        self.calls += 1
        return _empty_report()


class _Response:
    def __init__(self) -> None:
        self.deferred = False

    async def defer(self, *, ephemeral: bool, thinking: bool) -> None:
        assert ephemeral and thinking
        self.deferred = True


class _Interaction:
    guild_id = 1
    user = SimpleNamespace(id=2)

    def __init__(self) -> None:
        self.response = _Response()
        self.edits: list[str] = []

    async def edit_original_response(self, *, content: str) -> None:
        self.edits.append(content)


class _Loop:
    def __init__(self) -> None:
        self._times = iter((0.0, 0.0, 61.0))

    def time(self) -> float:
        return next(self._times)


async def test_live_progress_refreshes_one_message_and_cleans_up(monkeypatch) -> None:
    analyzer = _Analyzer()
    diagnostics = SimpleNamespace(
        _learning_progress=analyzer,
        _live_progress_sessions=set(),
    )
    interaction = _Interaction()

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(commands_module.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(commands_module.asyncio, "sleep", no_sleep)

    await Diagnostics.learning_progress_live.callback(
        diagnostics,
        interaction,
        window=25,
        duration_minutes=1,
        refresh_seconds=10,
    )

    assert interaction.response.deferred
    assert analyzer.calls == 2
    assert len(interaction.edits) == 2
    assert interaction.edits[0].startswith("LIVE | update #1")
    assert interaction.edits[1].startswith("FINISHED | update #2")
    assert diagnostics._live_progress_sessions == set()
