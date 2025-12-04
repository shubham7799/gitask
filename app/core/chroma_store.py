from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
import os

load_dotenv()

class LangchainChromaStore:

    def __init__(self, persist_dir="chroma_store"):
        os.makedirs(persist_dir, exist_ok=True)

        self.embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.db = Chroma(
            collection_name="code_chunks",
            embedding_function=self.embedder,
            persist_directory=persist_dir
        )

        self.chat_histories = {}

    def add_chunks(self, chunks):
        texts = [c["content"] for c in chunks]
        paths = [c["file"] for c in chunks]

        self.db.add_texts(texts=texts, paths=paths)

    def get_retriever(self, k=5):
        return self.db.as_retriever(search_kwargs={"k": k})
    
    def get_session_history(self, session_id: str):
        """Get or create chat history for a session."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]