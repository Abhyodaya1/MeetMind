import json
import os
import time

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-120b"

MAX_RETRIES = 3


def call_llm(prompt):
    """
    Send a prompt to Groq and return parsed JSON.

    Retries temporary failures before giving up.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI meeting intelligence assistant. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("LLM returned empty content.")

            return json.loads(content)

        except Exception as error:

            last_error = error

            print(
                f"LLM attempt {attempt}/{MAX_RETRIES} failed: "
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"LLM failed after {MAX_RETRIES} attempts."
    ) from last_error


def analyze_chunk(prompt):
    return call_llm(prompt)


def synthesize_meeting(prompt):
    return call_llm(prompt)