import os
import shutil
import tempfile
from git import Repo

def extract_repo_files(github_url: str, extensions=None):
    """
    Clones a GitHub repo and extracts all files matching the given extensions.

    Args:
        github_url (str): Public GitHub repository URL.
        extensions (list): Allowed file extensions (e.g., ['.py', '.js']).

    Returns:
        dict: { "file_path": "file_content", ... }
    """

    if extensions is None:
        extensions = [
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".dart", ".java", ".kt",
            ".go", ".rs",
            ".md", ".yaml", ".yml"
        ]

    temp_dir = tempfile.mkdtemp()

    try:
        Repo.clone_from(github_url, temp_dir)

        collected_files = {}

        for root, dirs, files in os.walk(temp_dir):
            # Skip unwanted directories
            skip_dirs = ["node_modules", "build", "dist", ".git", "__pycache__"]
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                file_path = os.path.join(root, file)

                # Check extension
                if any(file.endswith(ext) for ext in extensions):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            collected_files[file_path] = f.read()
                    except Exception:
                        pass  # Ignore unreadable files

        return collected_files

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)