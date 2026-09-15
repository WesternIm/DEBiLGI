"""Explicit, checksum-verified installation of the published Slovnet models."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODELS = (
    ("https://storage.yandexcloud.net/natasha-navec/packs/"
     "navec_news_v1_1B_250K_300d_100q.tar",
     "f07270833d78523edc5781538d67038e95b43975e4a7ae757c693b687f9cbfca"),
    ("https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar",
     "0759ce82eaac4430f5b4ac27b2444aa6e4b6501ca38c2f45d62467265383d85e"),
)


def verify_models(directory: Path) -> None:
    for url, expected in MODELS:
        path = directory / url.rsplit("/", 1)[1]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model checksum mismatch: {path.name}")


def install_models(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for url, expected in MODELS:
        path = directory / url.rsplit("/", 1)[1]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(url, timeout=60) as response:
            data = response.read(40_000_001)
        if len(data) > 40_000_000 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded model checksum mismatch: {path.name}")
        temporary = path.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(path)
    verify_models(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/models/slovnet"))
    install_models(parser.parse_args().directory)
