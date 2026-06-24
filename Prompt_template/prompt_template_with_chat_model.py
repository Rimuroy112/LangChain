from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)

# Part 1: Prompt from template
print("----Prompt from template----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic":"dogs"})
result = model.invoke(prompt)
print(result.content)
