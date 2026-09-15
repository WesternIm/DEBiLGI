"""Explicit installation of the published RuWordNet 2.0 SQLite database."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://github.com/avidale/python-ruwordnet/releases/download/0.0.4/ruwordnet-2021.db"
SHA256 = "584892becc590ab287aa5eb407204962f8ccb58283482fbbbe9e1a75b6d1bd53"


def install_ruwordnet(path: Path) -> None:
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 128_000_000:
                    raise ValueError("RuWordNet download exceeds expected size")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("RuWordNet checksum mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path,
                        default=Path("data/models/ruwordnet/ruwordnet-2021.db"))
    install_ruwordnet(parser.parse_args().path)
