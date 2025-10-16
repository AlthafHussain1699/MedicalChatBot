import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

def load_pdf(path: str):
    """Load PDF and return documents."""
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_docs(docs, chunk_size=1000, overlap=200):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    """Create a FAISS vector store with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def get_llm():
    """Initialize the Groq LLM with a supported model."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("❌ Missing GROQ_API_KEY. Add it to your .env file.")
    
    # Updated model name (currently supported)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=key
    )

def main(pdf_path):
    print("📄 Loading PDF...")
    docs = load_pdf(pdf_path)

    print("✂️ Splitting text...")
    chunks = chunk_docs(docs)

    print("🧩 Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    print("🤖 Connecting to Groq model...")
    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n✅ Ready! Ask questions about your PDF.\nType 'exit' to quit.\n")

    # Chat loop
    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Updated for LangChain v0.2+
        answer = qa({"query": query})["result"]
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    import sys
    main("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
