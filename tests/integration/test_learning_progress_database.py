from pathlib import Path

from debilgi.diagnostics.progress import LearningProgressAnalyzer, ProgressVerdict
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


async def test_empty_database_returns_an_insufficient_data_report(tmp_path: Path) -> None:
    database = Database(tmp_path / "learning-progress.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        report = await LearningProgressAnalyzer(database).report(1)
    finally:
        await database.close()

    assert report.verdict is ProgressVerdict.INSUFFICIENT_DATA
    assert report.comparable_sections == 0
    assert report.feedback.current.samples == 0
    assert report.comprehension.current.samples == 0
    assert report.predictions.current.samples == 0
    assert report.pipeline.current.samples == 0
    assert report.quality_loop.queued_cases == 0
