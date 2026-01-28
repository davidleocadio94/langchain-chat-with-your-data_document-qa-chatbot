"""LangChain Document Q&A - RAG Pipeline for PDF Documents"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class DocumentQA:
    """Document Q&A system using RAG (Retrieval Augmented Generation)."""

    def __init__(self):
        self.vectordb = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def load_and_process_pdf(self, pdf_path: str) -> str:
        """Load a PDF and create vector store."""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and vector store
        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
        )

        # Create QA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=self.vectordb.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

        return f"Loaded {len(docs)} pages, created {len(splits)} chunks"

    def ask_question(self, question: str) -> tuple[str, list]:
        """Ask a question about the loaded document."""
        if not self.qa_chain:
            return "Please upload a PDF document first.", []

        result = self.qa_chain({"question": question})
        answer = result["answer"]

        # Extract source info
        sources = []
        for doc in result.get("source_documents", [])[:3]:
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:200] + "..."
            sources.append(f"Page {page}: {preview}")

        return answer, sources

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        return "Conversation memory cleared."


# Global instance for Gradio
qa_system = DocumentQA()


def process_pdf(file) -> str:
    """Process uploaded PDF file."""
    if file is None:
        return "Please upload a PDF file."

    try:
        result = qa_system.load_and_process_pdf(file.name)
        qa_system.clear_memory()
        return f"✅ {result}. Ready for questions!"
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"


def chat(message: str, history: list) -> str:
    """Chat with the document."""
    if not message.strip():
        return ""

    answer, sources = qa_system.ask_question(message)

    response = answer
    if sources:
        response += "\n\n---\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            response += f"\n{i}. {source}\n"

    return response
