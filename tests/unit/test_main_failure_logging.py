import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from debilgi import main as entrypoint
from debilgi.logging import configure_logging


def test_startup_exception_survives_a_closed_console(tmp_path, monkeypatch):
    configure_logging("INFO", tmp_path)
    monkeypatch.setattr(entrypoint, "run", AsyncMock(side_effect=RuntimeError("startup failed")))
    try:
        with pytest.raises(SystemExit) as exc:
            entrypoint.main()
        assert exc.value.code == 1
        text = (tmp_path / "runtime.log").read_text(encoding="utf-8")
        assert "Bot terminated unexpectedly" in text
        assert "Traceback" in text
        assert "RuntimeError: startup failed" in text
    finally:
        for logger in (logging.getLogger("debilgi"), *[
            item for item in logging.Logger.manager.loggerDict.values()
            if isinstance(item, logging.Logger) and item.name.startswith("debilgi.")
        ]):
            for handler in tuple(logger.handlers):
                if Path(getattr(handler, "baseFilename", "")).parent == tmp_path:
                    logger.removeHandler(handler)
                    handler.close()
            logger.propagate = True
