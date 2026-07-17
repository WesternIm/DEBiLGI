import hashlib
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
RELEASED_CHECKSUMS = {
    1: "cc690e7a4e95e3e9de890e8c75ec73b12f8b72390e1abcee8b14315b49e96ac2",
    2: "bdbb5eab8ae8b24291597de351e060a029a68195843e7f1eaa9ab223d3cfeb6d",
    3: "9da6e8e70442f817855bcabc3f0fe7322d5f0ac793c195c9e74c6b77e5f2b095",
    4: "67ce83deec4684a0c54d525a6a6b4e8a01d21af153907d54a7c380af24d36127",
    5: "79c3af84919a8eddc7374b5e8d6cfc242360ece7d104df4439d5bbd54d5b0090",
    6: "244672d02d4f8b653f5f4777e97d367d6c73a163a83e29b66d50a3dc70636bae",
    7: "b1d194060f867ea6f0ca321e4a1ba227a4b61ef3a11a11705ef78cf8624d8c01",
    8: "5685b707aeeeb5d42deb12b980d8adc484936302d83f0f67b7fb43ace2d0609b",
    9: "eaa720bab7cca216a1d6b9dc82d4fff0a09abe8b497e533b43231516862f8334",
    10: "87eda08b03f3c22bc1602b54246d7f1bbd89966e156d2707ca0833b1b7383692",
    11: "dd8f9cf5431297f1698ccbe5c6fca9d38dccc8fea21d9115fe0ea31b62c65b13",
    12: "dcf4303607ed5585a10b7ea6f660a19ca007274904ecc903a196b4061cc0c2df",
    13: "76256f1844c4f7a0d9240fe40a56fc88e9f3c82f9e59328764f22305d01d2fd7",
    14: "34cd751882606a14d3b4e23cdcba4217173fc31764a70dcab90f4110282e43bb",
    15: "f54122e6e1f1eb637a015d7c53777993bece8c13c023950ab5c4362244bf5cf2",
}


def test_released_migrations_are_byte_stable() -> None:
    for version, expected in RELEASED_CHECKSUMS.items():
        matches = tuple(MIGRATIONS.glob(f"{version:03d}_*.sql"))
        assert len(matches) == 1
        sql = matches[0].read_text(encoding="utf-8")
        assert hashlib.sha256(sql.encode("utf-8")).hexdigest() == expected
