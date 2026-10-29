import hashlib
import json
import sqlite3
import time
from pathlib import Path


class ResultCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL)"
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(image_path: Path, namespace: str) -> str:
        digest = hashlib.sha256()
        digest.update(namespace.encode("utf-8"))
        with open(image_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def get(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM results WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, payload: dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO results (key, payload, created) VALUES (?, ?, ?)",
            (key, json.dumps(payload), time.time()),
        )
        self.connection.commit()

    def clear(self) -> None:
        self.connection.execute("DELETE FROM results")
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    def size(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def close(self) -> None:
        self.connection.close()
