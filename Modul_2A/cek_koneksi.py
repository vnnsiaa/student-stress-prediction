import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()                                        # baca GROQ_API_KEY dari .env
client = Groq(api_key=os.environ["GROQ_API_KEY"])

resp = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": "Jelaskan dalam satu kalimat: apa itu klasifikasi teks?"}],
)
print(resp.choices[0].message.content)
