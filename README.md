RAG AI Medical Chat App with Groq LLM
Short Description

This is a Retrieval-Augmented Generation (RAG) application that allows you to ask questions about PDF documents. It uses FAISS for vector storage, HuggingFace embeddings for text encoding, and the Groq LLM for natural language answers. The application is implemented in Python with LangChain v0.2+, and is fully compatible with virtual environments (venv).

Features

Load and process PDF documents.

Split PDF text into smaller chunks for better retrieval.

Create a vector store using FAISS with embeddings from HuggingFace.

Ask natural language questions and get accurate answers using Groq LLM.

Fully runs inside a Python virtual environment.

Command-line interface for interactive Q&A.

Easily extensible for web interface integration (e.g., Streamlit).

Installation

Clone the repository

git clone <your-repo-url>
cd rag_groq_app


Create and activate a virtual environment

python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Set your Groq API key

Create a .env file in the project root:

GROQ_API_KEY=your-groq-api-key

Usage

Place the PDF you want to query in the project directory.

Run the application:

python rag_pdf_groq.py your_document.pdf


Ask questions about your PDF interactively:

Your question: What is the main topic of this PDF?
Answer: ...


Type exit or quit to close the program.

Dependencies

Python 3.10+

langchain

langchain-community

langchain-groq

langchain-huggingface

faiss-cpu

sentence-transformers

pypdf

python-dotenv

Notes

Make sure your Groq account has access to a supported LLM model.

Currently uses HuggingFace embeddings for text encoding. Groq embeddings support may be added later.

The script is compatible with LangChain v0.2+, using .invoke() or __call__ instead of the deprecated run() method.
