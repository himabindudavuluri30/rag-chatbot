# 🤖 AI PDF RAG Chatbot

An AI-powered PDF question-answering chatbot that allows users to upload a PDF and ask questions about its content. The application extracts information from the document, retrieves relevant text, and generates answers using Gemini AI.

## 🚀 Live Demo

[**Try the AI PDF RAG Chatbot**](https://rag-chatbot-phi-teal.vercel.app/)

## 📌 Features

* 📄 Upload PDF documents
* 🔍 Retrieve relevant information from the document
* 🤖 Generate answers using Gemini AI
* ⚡ Lightweight FastAPI backend
* 🌐 Simple web-based chatbot interface
* ☁️ Deployed on Vercel
* 📓 Includes the original Google Colab RAG implementation

## 🧠 How It Works

```text
        Upload PDF
             ↓
      Extract PDF Text
             ↓
    Retrieve Relevant Text
             ↓
      Build RAG Context
             ↓
        Gemini AI
             ↓
       Generate Answer
```

## 🛠️ Technologies Used

* Python
* FastAPI
* PyPDF
* Gemini AI
* JavaScript
* HTML & CSS
* Vercel

## 📂 Project Structure

```text
rag-chatbot/
│
├── RAG_chatbot.ipynb
├── api/
│   └── index.py
├── public/
│   └── index.html
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

## 📓 Original RAG Notebook

`RAG_chatbot.ipynb` contains the original Google Colab implementation of the project, including:

* PDF loading
* Text chunking
* Sentence Transformers embeddings
* FAISS vector database
* Similarity search
* Gemini-based answer generation

The deployed application uses a lightweight retrieval approach optimized for serverless deployment.

## 🔐 Environment Variable

The application requires a Gemini API key.

Create a `.env` file locally:

```text
GEMINI_API_KEY=your_api_key_here
```

For Vercel deployment, add `GEMINI_API_KEY` through Vercel Environment Variables.

**Never commit the `.env` file to GitHub.**

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/himabindudavuluri30/rag-chatbot.git
cd rag-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn api.index:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## 🎯 Project Goal

The goal of this project is to demonstrate how Retrieval-Augmented Generation can be used to build a document question-answering system that combines document retrieval with generative AI.



