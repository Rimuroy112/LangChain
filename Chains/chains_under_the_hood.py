from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY"),
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a comedian who tells jokes about {topic}."),
        ("human","Tell me {joke_count} jokes."),
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)
chain = RunnableSequence(first=format_prompt, middle = [invoke_model], last = parse_output)
response = chain.invoke({"topic":"lawyers","joke_count":3})
print(response)
