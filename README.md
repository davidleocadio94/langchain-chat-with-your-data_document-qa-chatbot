---
title: Document Q&A Chatbot
emoji: 📄
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

# Document Q&A Chatbot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/davidleocadio94DLAI/langchain-chat-with-your-data_document-qa-chatbot)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://langchain.com)

A conversational document Q&A system using RAG (Retrieval Augmented Generation). Upload a PDF and chat with your documents using natural language.

## Features

- **PDF Document Processing** - Upload any PDF and create searchable embeddings
- **Conversational Memory** - Follow-up questions understand context
- **Source Citations** - See which parts of the document answers come from
- **Vector Search** - ChromaDB for efficient similarity search

## Tech Stack

![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-412991)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/davidleocadio94/langchain-chat-with-your-data_document-qa-chatbot.git
cd langchain-chat-with-your-data_document-qa-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Locally

```bash
# Set your API key
export OPENAI_API_KEY=your-api-key

# Run the app
python app.py
```

Open http://localhost:7860 in your browser.

### Run with Docker

```bash
# Build the image
docker build -t document-qa-chatbot .

# Run the container
docker run -p 7860:7860 -e OPENAI_API_KEY=your-api-key document-qa-chatbot
```

Open http://localhost:7860 in your browser.

## How It Works

1. **Document Loading** - PyPDF extracts text from uploaded PDFs
2. **Text Splitting** - RecursiveCharacterTextSplitter creates overlapping chunks
3. **Embedding** - OpenAI embeddings convert text to vectors
4. **Vector Store** - ChromaDB stores and indexes embeddings
5. **Retrieval** - Similar chunks are retrieved for each question
6. **Generation** - GPT-3.5-Turbo generates answers using retrieved context

### Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                          ↓
Question → Embedding → Similarity Search → Retrieved Chunks
                                                          ↓
                                          LLM → Answer with Sources
```

## Example Questions

- "What are the main topics covered in this document?"
- "Can you summarize the key points?"
- "What examples are given to illustrate the concepts?"

---

Built as part of the [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) course on DeepLearning.AI
