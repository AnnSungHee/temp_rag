import os
from dotenv import load_dotenv
from groq import Groq

# .env 파일 로드
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "안녕하세요"}
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
