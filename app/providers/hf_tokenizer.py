from transformers import AutoTokenizer

from app.providers.tokenizer import TokenizerProvider


class HFTokenizerProvider(TokenizerProvider):

    def __init__(
        self,
        model_name="openai/gpt-oss-120b",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

    def count_tokens(self, text: str) -> int:

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return len(tokens)