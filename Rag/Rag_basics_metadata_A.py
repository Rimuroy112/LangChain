import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir,"4_rag","books")
db_dir = os.path.join(current_dir,"db")
persistent_directory = os.path.join(db_dir,"chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path"
        )
    # Get a list of all files in the 'books' folder that end with .txt
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    documents=[]

    # Loop through each .txt file found
    for book_file in book_files:
        # Build the full file path
        file_path = os.path.join(books_dir,book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # where metadata is added key-value pair
            doc.metadata={"source": book_file}
            documents.append(doc)
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print("\n--- Document chunk information ---")
    print(f"Number of document chunks: {len(docs)}")
    print("\n --- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("\n--- Finished creating embeddings ---")
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs,embeddings,persist_directory=persistent_directory
    )
    print("\n--- Finished creating and persisting vector store ---")
else:
    print("vector store already exists. No need to initialize ")

