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

# # Part 1: Prompt from template
# print("----Prompt from template----")
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)
# prompt = prompt_template.invoke({"topic":"dogs"})
# result = model.invoke(prompt)
# print(result.content)

# # Part 2: Prompt with multiple placeholders
# template_multiple = """You are a useful assistant
# Human: Tell me a {adjective} short story about {animal}.
# Assistant: """
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective":"funny","animal":"dogs"})
# result = model.invoke(prompt)
# print(result.content)

# Part 3: Prompt with system and Human messages (Using Tuples)
messages = [
    ("system","You are a comedian who tells joke about {topic}."),
    ("human","Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic":"lawyers","joke_count":3})

result = model.invoke(prompt)
print("\n----Prompt with system and Human Messages(Tuple)----\n")
print(result.content)