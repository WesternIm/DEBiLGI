import hashlib
from io import BytesIO

import pytest

from debilgi.language import ruwordnet_models
from debilgi.language.ruwordnet import LexicalLink, LexicalSense, open_ruwordnet, select_sense


def test_antonym_does_not_select_a_sense():
    senses = (
        LexicalSense("a", "слово", "", "N", (), (
            LexicalLink("antonym", "b", "холод", ("холод",)),
        )),
        LexicalSense("b", "слово", "", "N", (), ()),
    )
    assert select_sense("слово", senses, "холод").selected is None


def test_derivatives_from_one_context_word_do_not_multiply_evidence():
    senses = (
        LexicalSense("a", "домашний дом", "", "N", (), ()),
        LexicalSense("b", "слово", "", "N", (), (
            LexicalLink("related", "c", "животное", ("животное",)),
        )),
    )
    result = select_sense("слово", senses, "жилище животное", derived=(
        ("жилище", "дом"), ("жилище", "домашний"),
    ))
    assert result.selected is None  # two vs one; insufficient margin


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_invalid_database_falls_back_without_creating_file(tmp_path, exists):
    path = tmp_path / "reference.db"
    if exists:
        path.write_bytes(b"invalid sqlite database")
    assert open_ruwordnet(path) is None
    assert path.exists() is exists
    if exists:
        assert path.read_bytes() == b"invalid sqlite database"


def test_failed_download_preserves_existing_database_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "reference.db"
    path.write_bytes(b"previous database")
    monkeypatch.setattr(ruwordnet_models, "urlopen", lambda *a, **k: BytesIO(b"corrupt download"))
    with pytest.raises(ValueError, match="checksum"):
        ruwordnet_models.install_ruwordnet(path)
    assert path.read_bytes() == b"previous database"
    assert not path.with_suffix(".download").exists()


def test_verified_download_installs_and_next_install_works_offline(tmp_path, monkeypatch):
    payload = b"verified reference database"
    path = tmp_path / "reference.db"
    monkeypatch.setattr(ruwordnet_models, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(ruwordnet_models, "urlopen", lambda *a, **k: BytesIO(payload))
    ruwordnet_models.install_ruwordnet(path)
    assert path.read_bytes() == payload

    def offline(*args, **kwargs):
        raise AssertionError("An installed reference must not need a network request")

    monkeypatch.setattr(ruwordnet_models, "urlopen", offline)
    ruwordnet_models.install_ruwordnet(path)
