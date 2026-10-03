import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"  # Specify the model you want to use

def call_llm(prompt):

    response = client.chat.completions.create(
        model=MODEL,
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

    return json.loads(content) 

def analyze_chunk(prompt):
    return call_llm(prompt) 

def synthesize_meeting(prompt):
    return call_llm(prompt)



