"""LangChain Document Q&A - RAG Pipeline for PDF Documents"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class DocumentQA:
    """Document Q&A system using RAG (Retrieval Augmented Generation)."""

    def __init__(self):
        self.vectordb = None
        self.retriever = None
        self.llm = None
        self.chat_history = []

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

        # Set up retriever and LLM
        self.retriever = self.vectordb.as_retriever()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        return f"Loaded {len(docs)} pages, created {len(splits)} chunks"

    def ask_question(self, question: str) -> tuple[str, list]:
        """Ask a question about the loaded document."""
        if not self.retriever:
            return "Please upload a PDF document first.", []

        # Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format chat history
        history_str = ""
        for h in self.chat_history[-5:]:  # Keep last 5 exchanges
            history_str += f"Human: {h['question']}\nAssistant: {h['answer']}\n"

        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context and chat history.
        If you cannot find the answer in the context, say so.

        Context:
        {context}

        Chat History:
        {history}

        Question: {question}

        Answer:
        """)

        # Create chain
        chain = prompt | self.llm | StrOutputParser()

        # Get answer
        answer = chain.invoke({
            "context": context,
            "history": history_str,
            "question": question
        })

        # Store in history
        self.chat_history.append({"question": question, "answer": answer})

        # Extract source info
        sources = []
        for doc in docs[:3]:
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:200] + "..."
            sources.append(f"Page {page}: {preview}")

        return answer, sources

    def clear_memory(self):
        """Clear conversation memory."""
        self.chat_history = []
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
