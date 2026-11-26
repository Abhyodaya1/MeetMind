from sentence_transformers import SentenceTransformer

from app.providers.embeddings import EmbeddingProvider


class HFEmbeddingProvider(EmbeddingProvider):
    """CPU-bound SentenceTransformer embeddings to preserve GPU VRAM."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in vectors]
