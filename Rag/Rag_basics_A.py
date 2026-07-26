import os
# Import CharacterTextSplitter to break large text into smaller chunks
from langchain.text_splitter import CharacterTextSplitter 
# Import TextLoader to load text file from a .txt file
from langchain_community.document_loaders import TextLoader
# Import chroma - a vector database to store text embeddings
from langchain_community.vectorstores import chroma
# Import HuggingFaceEmbeddings - convert text into numbers
from langchain_community.embeddings import HuggingFaceEmbeddings

# Get the folder where the python script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to text file
file_path = os.path.join(current_dir,"4_rag","books","odyssey.txt")
# Build the path where the vector database will be saved
persistent_directory = os.path.join(current_dir,"db","chroma_db")

# check if the vector database already exits(avoid recreating it)
if not os.path.exists(persistent_directory):
    print("Initializing vector store...")
    # check if the text file actually exits at the given path
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path"
        )

    # Create a TextLoader object to read the text file
    loader = TextLoader(file_path)
    # Load the entire text file into the memory
    documents = loader.load()
    # Create a text splitter that breaks text into chunk of 1000 characters
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )    
    # split the loaded document into smaller chunks
    docs = text_splitter.split_documents(documents)

    print("\n-- Document chunks information --")
    print(f"Number of document chunk: {len(docs)}")

    # print the first chunk as example
    print(f"Sample chunk: \n{docs[0].page_content}\n")

    print(f"\n---Creating embeddings---")
    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-VL"
    )
    print("\n--Finished creating embeddings--")
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs,embeddings,persist_directory=persistent_directory
    )
    print("\n--- Finished Creating Vector Store ---")
else:
    print("Vector store already exist.No need to initialize")





