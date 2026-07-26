import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))

persistent_directory = os.path.join(current_dir,"db","chroma_db")

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)
# Load the existing vector store from the persistent directory
db = Chroma(persist_directory=persistent_directory,
           embedding_function = embeddings
   )

query = "Who is the odysseus wife?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
                "k":3,  # 3 most similar documents
                "score_threshold":0.3
    }
)
relevant_docs = retriever.invoke(query)
print("\n--- Relevant Document ---")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}")

    if doc.metadata:
        print(f"source:{doc.metadata.get('source','Unknown')}\n")