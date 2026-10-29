import torch
from sentence_transformers import SentenceTransformer

from pipeline.config import EMBEDDER_NAME, FIELDS, QUERY_TEXTS, SELECTOR_TOP_K


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class LineSelector:
    def __init__(self, device: str = "cuda", top_k: int = SELECTOR_TOP_K):
        self.device = device
        self.top_k = top_k
        self.model = SentenceTransformer(EMBEDDER_NAME, device=device)
        queries = [QUERY_TEXTS[field] for field in FIELDS]
        self.query_embeddings = self.model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)

    def _pick(self, lines: list[str], embeddings: torch.Tensor) -> str:
        if len(lines) <= self.top_k * len(FIELDS):
            return "\n".join(lines)
        similarity = embeddings @ self.query_embeddings.T
        chosen: set[int] = set()
        k = min(self.top_k, len(lines))
        for column in range(similarity.shape[1]):
            top = torch.topk(similarity[:, column], k=k).indices.tolist()
            for index in top:
                chosen.add(index)
                if index + 1 < len(lines):
                    chosen.add(index + 1)
        return "\n".join(lines[i] for i in sorted(chosen))

    def select(self, text: str) -> str:
        lines = split_lines(text)
        if not lines:
            return ""
        embeddings = self.model.encode(lines, convert_to_tensor=True, normalize_embeddings=True)
        return self._pick(lines, embeddings)

    def select_batch(self, texts: list[str]) -> list[str]:
        per_doc = [split_lines(text) for text in texts]
        flat = [line for lines in per_doc for line in lines]
        if not flat:
            return ["" for _ in texts]
        embeddings = self.model.encode(flat, convert_to_tensor=True, normalize_embeddings=True, batch_size=256)
        outputs = []
        offset = 0
        for lines in per_doc:
            if not lines:
                outputs.append("")
                continue
            outputs.append(self._pick(lines, embeddings[offset : offset + len(lines)]))
            offset += len(lines)
        return outputs
