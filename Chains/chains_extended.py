from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq
import os
load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a comedian who tells joke about {topic}."),
        ("human","Tell me {joke_count} jokes."),
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic":"lawyers","joke_count":3})
print(result)
