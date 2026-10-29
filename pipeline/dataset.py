import json
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import GROUND_TRUTH_DIR, IMAGE_DIR, IMAGE_EXTENSIONS, SYNTHETIC_INDICES

INDEX_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Document:
    index: int
    image_path: Path
    truth_path: Path
    source: str

    @property
    def name(self) -> str:
        return self.image_path.stem

    def load_truth(self) -> dict:
        with open(self.truth_path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def document_index(path: Path) -> int:
    match = INDEX_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot derive document index from {path.name}")
    return int(match.group(1))


def load_documents(limit: int | None = None, source: str | None = None, indices=None) -> list[Document]:
    documents = []
    for image_path in IMAGE_DIR.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = document_index(image_path)
        truth_path = GROUND_TRUTH_DIR / f"{image_path.stem}.json"
        if not truth_path.exists():
            raise FileNotFoundError(f"Missing ground truth for {image_path.name}")
        kind = "synthetic" if index in SYNTHETIC_INDICES else "real"
        documents.append(Document(index, image_path, truth_path, kind))
    documents.sort(key=lambda doc: doc.index)
    if source:
        documents = [doc for doc in documents if doc.source == source]
    if indices is not None:
        wanted = set(indices)
        documents = [doc for doc in documents if doc.index in wanted]
    if limit:
        documents = documents[:limit]
    return documents
