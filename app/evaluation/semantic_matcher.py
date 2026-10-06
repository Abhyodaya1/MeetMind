from sentence_transformers import SentenceTransformer


class SemanticMatcher:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:

        embeddings = self.model.encode(
            [text_a, text_b],
            normalize_embeddings=True,
        )

        return float(
            embeddings[0] @ embeddings[1]
        )