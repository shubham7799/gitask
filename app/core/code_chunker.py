from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import os

def detect_language(path):
    ext = os.path.splitext(path)[1].lower()

    mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".php": Language.PHP,
        ".rs": Language.RUST,
        ".c": Language.CPP,
        ".cpp": Language.CPP,
    }

    return mapping.get(ext, None)

def split_code_by_language(path, text, chunk_size=800, chunk_overlap=150):
    lang = detect_language(path)

    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        # fallback generic splitter (Markdown, YAML, etc.)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    docs = splitter.create_documents([text])

    # Convert docs to simple dicts for RAG use
    return [
        {
            "file": path,
            "language": lang.name if lang else "generic",
            "content": d.page_content,
            "metadata": d.metadata
        }
        for d in docs
    ]
