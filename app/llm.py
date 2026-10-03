import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_minutes(prompt):

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
         messages=[
            {
                "role": "system",
                "content": ("You are an AI meeting intelligence assistant."
                            "Return ONLY valid JSON.")
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    content =  response.choices[0].message.content

    return json.loads(content)  # Parse the JSON string into a Python dictionary


