from app.core.repo_extractor import extract_repo_files
from app.core.code_chunker import split_code_by_language
from app.core.chroma_store import LangchainChromaStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def format_docs(docs):
    """Format retrieved documents with file path context."""
    formatted = []
    for doc in docs:
        file_path = doc.metadata.get('file', 'unknown')
        language = doc.metadata.get('language', 'unknown')
        formatted.append(f"File: {file_path} (Language: {language})\n{doc.page_content}")
    return "\n\n" + "="*80 + "\n\n".join(formatted)


def process_repo_to_chroma(github_url: str, persist_dir="chroma_store"):
    """
    Full pipeline:
    1. Clone & extract repo files
    2. Chunk code using split_code_by_language()
    3. Store chunks in Chroma using LangChain
    4. Return a RAG chain for querying
    
    Args:
        github_url (str): GitHub repository URL
        persist_dir (str): Directory to persist ChromaDB
        
    Returns:
        RAG chain for querying the repository
    """
    
    # Validate input
    if not github_url.strip():
        raise ValueError("GitHub URL cannot be empty")
    
    if not (github_url.startswith("https://github.com/") or 
            github_url.startswith("git@github.com:")):
        raise ValueError("Invalid GitHub URL format. Must start with 'https://github.com/' or 'git@github.com:'")

    try:
        # Step 1: Extract files from repo
        files = extract_repo_files(github_url)

        if len(files) == 0:
            raise ValueError("No source files found in repository")

        # Step 2: Init vector DB
        store = LangchainChromaStore(persist_dir=persist_dir)
        all_chunks = []

        # Step 3: Run chunker file-by-file
        for file_path, content in files.items():
            try:
                file_chunks = split_code_by_language(
                    text=content,
                    path=file_path
                )
                all_chunks.extend(file_chunks)
            except Exception as e:
                continue

        if len(all_chunks) == 0:
            raise ValueError("No chunks created from repository files")

        # Step 4: Store into Chroma
        store.add_chunks(all_chunks)

        # Step 5: Create RAG chain
        retriever = store.get_retriever(k=5)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2
        )

        def format_docs(docs):
            formatted = []
            for doc in docs:
                file_path = doc.metadata.get('file', 'unknown')
                language = doc.metadata.get('language', 'unknown')
                formatted.append(f"File: {file_path} (Language: {language})\n{doc.page_content}")
            return "\n\n" + "="*80 + "\n\n".join(formatted)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code analysis assistant. Use the following code snippets from the repository to answer the question.
        Pay close attention to file paths, function names, class definitions, and code structure.

        Code Context:
        {context}

        If the context doesn't contain enough information to answer the question, say so clearly."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Create base chain
        base_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        qa_chain = RunnableWithMessageHistory(
            base_chain,
            store.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

        return qa_chain

    except Exception as e:
        raise