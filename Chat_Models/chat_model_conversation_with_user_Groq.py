from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os

load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)

while True:
    query = input("you: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("---Message History---")
print(chat_history)
