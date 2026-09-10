import os
import tempfile

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables from .env
load_dotenv()


app = FastAPI()


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home page
@app.get("/")
def home():
    return FileResponse("public/index.html")


# Health check
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "message": "RAG Chatbot backend is working",
        "gemini_key_found": bool(os.getenv("GEMINI_API_KEY"))
    }


# RAG chatbot endpoint
@app.post("/api/chat")
async def chat_with_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    pdf_path = None

    try:
        # Get Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return {
                "error": "GEMINI_API_KEY is not configured."
            }

        # Check that a PDF was uploaded
        if not file.filename.lower().endswith(".pdf"):
            return {
                "error": "Please upload a PDF file."
            }

        # Check that question is not empty
        if not question.strip():
            return {
                "error": "Please enter a question."
            }

        # Read uploaded PDF
        pdf_data = await file.read()

        if not pdf_data:
            return {
                "error": "The uploaded PDF is empty."
            }

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(pdf_data)
            pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            return {
                "error": "Could not extract text from the PDF."
            }

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            return {
                "error": "Could not create document chunks."
            }

        # Create embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create FAISS vector database
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model
        )

        # Retrieve the most relevant chunks
        retrieved_docs = vectorstore.similarity_search(
            question,
            k=3
        )

        # Build context
        context = "\n\n".join(
            document.page_content
            for document in retrieved_docs
        )

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Gemini model
        model = genai.GenerativeModel(
            "gemini-3.6-flash"
        )

        # RAG prompt
        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I couldn't find that information in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        response = model.generate_content(prompt)

        # Return answer
        return {
            "answer": response.text
        }

    except Exception as e:
        return {
            "error": str(e)
        }

    finally:
        # Delete temporary PDF
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass