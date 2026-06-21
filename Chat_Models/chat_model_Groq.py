
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)
result = model.invoke("What is 81 divided by 9?")
print(result.content)