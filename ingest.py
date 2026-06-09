from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#Configuration at the tops: we do this so that we dont need to change the viriable deep down in the code
KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def ingest():

    print("Step 1: Loading knowledge files...")
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")


    print("Step 2: Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Step 3: Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print(f"  Model loaded: {EMBED_MODEL}")

    print("Step 4: Embedding chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Saved {len(chunks)} chunks to {CHROMA_DIR}/")
    print("Ingestion complete!")

if __name__ == "__main__":
        ingest()

