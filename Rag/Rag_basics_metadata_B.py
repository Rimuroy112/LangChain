# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

# current_dir = os.path.dirname(os.path.abspath(__file__))

# db_dir = os.path.join(current_dir,"db")

# persistent_directory = os.path.join(db_dir,"chroma_db_with_metadata")

# embeddings = HuggingFaceEmbeddings(
#     model_name = "all-MiniLM-L6-v2"
# )
# db = Chroma(persist_directory=persistent_directory,
#             embedding_function=embeddings)

# query = "How did juliet die?"

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold" : 0.01},
# )
# relevant_docs = retriever.invoke(query)

# print("\n--- Relevant Document ---")
# for i,doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     print(f"Source: {doc.metadata['source']}\n")

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# YOUR EXISTING CODE
query = "How did juliet die?"

# Fix 1: Use "similarity" instead of threshold (this WILL work)
print("\n=== FIX 1: Using 'similarity' search type ===")
retriever = db.as_retriever(
    search_type="similarity",  # Changed from "similarity_score_threshold"
    search_kwargs={"k": 3},
)
relevant_docs = retriever.invoke(query)

if relevant_docs:
    print(f"Found {len(relevant_docs)} documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
else:
    print("No documents found with 'similarity' search")

# Fix 2: Get scores to see what's happening
print("\n=== FIX 2: Checking similarity scores ===")
results = db.similarity_search_with_relevance_scores(query, k=5)
if results:
    print(f"Found {len(results)} results with scores:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i}: Score = {score:.6f}")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print()
    
    highest_score = max(score for _, score in results)
    print(f"Highest score: {highest_score:.6f}")
    print(f"Your threshold (0.01) is {'HIGHER' if 0.01 > highest_score else 'LOWER'} than highest score")
else:
    print("No results found at all!")
    print("This means the database might be empty or doesn't contain relevant documents.")

# Fix 3: Try with threshold but very low
print("\n=== FIX 3: Using threshold 0.000001 (very low) ===")
retriever2 = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.000001},  # Very low
)
docs2 = retriever2.invoke(query)

if docs2:
    print(f"Found {len(docs2)} documents with very low threshold:")
    for i, doc in enumerate(docs2, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
else:
    print("Still no documents found")

# Check if database has ANY documents
print("\n=== Checking if database has any documents ===")
try:
    # Try to get any document
    all_docs = db.similarity_search("", k=5)
    if all_docs:
        print(f"Database has at least {len(all_docs)} documents")
        print(f"Sample document: {all_docs[0].page_content[:200]}...")
    else:
        print("Database appears to be EMPTY!")
        print("You need to run Rag_basics_metadata_A.py to populate it first.")
except Exception as e:
    print(f"Error checking database: {e}")