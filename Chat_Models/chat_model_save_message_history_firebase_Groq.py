from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq
import os

load_dotenv()

PROJECT_ID = "chat-model-afbfe"
SESSION_ID = "User_session"
COLLECTION_NAME = "chat-history2"

KEY_FILE_PATH = "/mnt/f/LangChain/chat-model-afbfe-firebase-adminsdk-fbsvc-31b94ee124.json"

print("Initializing Firestore client with service account..")
client = firestore.Client.from_service_account_json(KEY_FILE_PATH, project=PROJECT_ID)

print("Initializing Firestore chat message history ")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection = COLLECTION_NAME,
    client = client,

)
print("Chat history initialized ")
print("Current chat history:", chat_history.messages)
print("Initializing Groq..")
model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)
print("Groq model ready")
while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    print(f"AI: {ai_response.content}")
