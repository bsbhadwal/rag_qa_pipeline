# config.py

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API and Model Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME", "gemini-2.0-flash"
)  # flash models are a good compromise between speed & instruction following. Avoid 2.5 as its still in beta
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Base Output Directory ---
# This will be a subdirectory relative to the script's location.
# All other output paths will be relative to this base directory.
# Example: if script is in /scripts/ and BASE_OUTPUT_DIR_NAME is "pipeline_output",
# then the base path will be /scripts/pipeline_output/
BASE_OUTPUT_DIR_NAME = "pipeline_outputs"

# --- Path Configuration (relative to BASE_OUTPUT_DIR_NAME) ---
# These define the names of subdirectories within the BASE_OUTPUT_DIR_NAME.
# pipeline_logic.py will be responsible for creating the full Path objects.
# For example, the full path to the cloned repo will be:
# Path(BASE_OUTPUT_DIR_NAME) / DEFAULT_REPO_SUBDIR
DEFAULT_REPO_SUBDIR = "cloned_repo"  # Subdirectory for the cloned repository
DEFAULT_CHROMA_SUBDIR = "chroma_db_persistent"  # Subdirectory for ChromaDB storage
DEFAULT_CACHE_SUBDIR = (
    "pipeline_cache_data"  # Subdirectory for caching intermediate files
)

# --- ChromaDB Configuration ---
DEFAULT_COLLECTION_NAME = "code_chunks"

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Code Processing Configuration ---
CODE_SPLITTER_CHUNK_LINES = 100  # Safe, maintains semantic cohesion
CODE_SPLITTER_CHUNK_LINES_OVERLAP = 20  # Ensures function context isn't lost
CODE_SPLITTER_MAX_CHARS = (
    8000  # Compatible with most embedding/token limits (nomic is 8196)
)

FILE_EXTENSION_TO_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "c_sharp",  # LlamaIndex uses 'c_sharp' for C#
    ".go": "go",
    ".rb": "ruby",
    # Add more extensions and their corresponding tree-sitter language names as needed
    # ".jsx": "javascript", # Example for JSX if using JS parser
    # ".tsx": "typescript", # Example for TSX if using TS parser
}
# Supported file suffixes will be derived from the keys of this map.
SUPPORTED_FILE_SUFFIXES = tuple(FILE_EXTENSION_TO_LANGUAGE_MAP.keys())


# --- File Names ---
# These are filenames, not directories, and will typically be placed within the cache directory.
CHUNKS_FILENAME = "chunks.jsonl"
HYBRID_CHUNKS_FILENAME = "hybrid_chunks.jsonl"

# --- Operational Parameters ---
MIN_SLEEP_SEC = (
    3  # Min sleep duration. We use sleep to stay within Gemini free tier restrictions.
)
MAX_SLEEP_SEC = (
    5  # Max sleep duration. Use sleep to stay within Gemini free tier restrictions.
)
FORCE_REFRESH_DEFAULT = False  # Default for force_refresh flags. Refresh flags ignore any locally cached files
SIMILARITY_TOP_K = 3  # Number of source documents to retrieve for context

# --- Git Clone Configuration ---
DEFAULT_REPO_URL = "https://github.com/psf/requests.git"  # default repo url


# --- Default Questions for Querying ---
DEFAULT_QUESTIONS = [
    # Easy
    "What does the `requests.get()` function do?",
    "What is the return type of a successful call to `requests.get()`?",
    # Medium
    "Explain how sending JSON data with `requests.post()` using the `json` parameter works.",
    "Explain the role of the `requests.Session` object.",
    "Explain how the `params` argument in `requests.get()` modifies the request URL.",
    "Explain the role of the `requests.utils.get_netrc_auth()` function.",
    # Hard
    "Explain how `requests` handles connection pooling using `urllib3`.",
    "Review the `requests.adapters.HTTPAdapter` class and its `send` method.",
    "What is the role of `PreparedRequest` objects in the `requests` library's request lifecycle?",
    "Explain how custom authentication mechanisms can be implemented for `requests` by creating a subclass of `requests.auth.AuthBase`.",
    # Multipart - Require deduction but from context - super HARD!
    "Describe the primary purpose of the requests.get() function. What are the key attributes of the Response object you'd typically inspect after a successful call, and what are their Python data types?"
    "Explain the difference between using the data parameter and the json parameter when making a POST request with requests.post(). When would you choose one over the other, and what is the typical Content-Type header set for each?"
    "How should you handle potential errors when making a request using the requests library? Describe at least two types of errors you might encounter (e.g., network issues, HTTP client/server errors) and how you would catch or check for them. Mention response.raise_for_status() if relevant.",
    "How do you send custom HTTP headers with a requests call (e.g., an Authorization token or a custom User-Agent)? How would you inspect the headers returned by the server in the Response object?",
    "If you need to send data as URL query parameters (e.g., ?name=Alice&age=30), how would you construct this using the requests library with a GET request? What attribute of the Response object would show you the exact URL that was requested?"
    "Explain the difference between response.text, response.content, and response.json(). What Python data type does each typically return, and in what scenarios would you use each one?"
    "What is a requests.Session object, and why would you use it instead of making individual calls like requests.get() or requests.post()? Name at least two benefits.",
    "What is the purpose of the timeout parameter in requests functions? What can happen if you don't set a timeout, and what type of value(s) does it accept?"
    "Briefly describe how you might handle basic HTTP authentication (username/password) when making a request. What parameter is typically used?",
    "Imagine you need to upload a file (e.g., an image) to a server using a POST request. Which parameter in a requests function would you use, and what would be the general structure of the data you provide to that parameter?",
    # FAIL QUESTION - LLM SHOULD RESPOND I DO NOT KNOW FROM PROVIDED CONTEXT (since that is a requirement in Requirements Doc) i.e. A: The provided code snippets do not contain information about "Mogambo".
    "Who is Mogambo?",
]
