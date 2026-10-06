import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from app.providers.llm import LLMProvider


load_dotenv()


class GroqProvider(LLMProvider):

    def __init__(
        self,
        model="openai/gpt-oss-120b",
        max_retries=3,
    ):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model = model
        self.max_retries = max_retries

    def _call(self, prompt):

        last_error = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                response = (
                    self.client
                    .chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an AI meeting "
                                    "intelligence assistant. "
                                    "Return only valid JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.1,
                        response_format={
                            "type": "json_object"
                        },
                    )
                )

                content = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                if not content:
                    raise ValueError(
                        "LLM returned empty content."
                    )

                return json.loads(content)

            except Exception as error:

                last_error = error

                print(
                    f"LLM attempt "
                    f"{attempt}/{self.max_retries} "
                    f"failed: {error}"
                )

                if attempt < self.max_retries:
                    time.sleep(
                        2 ** (attempt - 1)
                    )

        raise RuntimeError(
            f"LLM failed after "
            f"{self.max_retries} attempts."
        ) from last_error

    def analyze(self, prompt):
        return self._call(prompt)

    def synthesize(self, prompt):
        return self._call(prompt)