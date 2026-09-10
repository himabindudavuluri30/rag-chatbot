import os
import re
import tempfile

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pypdf import PdfReader

import google.generativeai as genai


# Load environment variables
load_dotenv()


app = FastAPI()


# CORS
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


# Simple text retrieval
def retrieve_relevant_text(text, question, top_k=5):
    """
    Lightweight retrieval method for Vercel.

    Splits the PDF text into paragraphs/small sections,
    scores them based on question words,
    and returns the most relevant sections.
    """

    sections = re.split(r"\n\s*\n", text)

    question_words = set(
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", question)
        if len(word) > 2
    )

    scored_sections = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        section_words = set(
            word.lower()
            for word in re.findall(r"\b[a-zA-Z0-9]+\b", section)
        )

        score = len(question_words.intersection(section_words))

        scored_sections.append((score, section))

    scored_sections.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = [
        section
        for score, section in scored_sections[:top_k]
        if score > 0
    ]

    # If no exact keyword matches, use the beginning
    # of the document as fallback context.
    if not selected:
        selected = [
            section
            for score, section in scored_sections[:top_k]
        ]

    return "\n\n".join(selected)


# Chat endpoint
@app.post("/api/chat")
async def chat_with_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    pdf_path = None

    try:
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return {
                "error": "GEMINI_API_KEY is not configured."
            }

        # Check PDF
        if not file.filename.lower().endswith(".pdf"):
            return {
                "error": "Please upload a PDF file."
            }

        # Check question
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

        # Extract text using PyPDF
        reader = PdfReader(pdf_path)

        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                pages_text.append(page_text)

        document_text = "\n\n".join(pages_text)

        if not document_text.strip():
            return {
                "error": "Could not extract text from the PDF."
            }

        # Lightweight retrieval
        context = retrieve_relevant_text(
            document_text,
            question,
            top_k=5
        )

        # Configure Gemini
        genai.configure(api_key=api_key)

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

        # Generate response
        response = model.generate_content(prompt)

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