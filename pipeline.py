# pipeline_logic.py

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
from chromadb import PersistentClient
from llama_index.core import Document, PromptTemplate, VectorStoreIndex
from llama_index.core.base.response.schema import Response  # For type hinting
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Local imports
import config
import constants
from constants import LogMessages as LM  # Alias for convenience

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
# TODO: Investigate whether logging can be disabled during initialization of this third-party library
# This line tuns off the irritating INFO log messages -
# google_genai.models - INFO - 5933 - AFC remote call 1 is done.
logging.getLogger("google_genai").setLevel(logging.WARNING)

# TODO: Investigate whether logging can be disabled during initialization of this third-party library
# This line tuns off the irritating INFO log messages -
# httpx - INFO - 1025 - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent "HTTP/1.1 200 OK"
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Utility Functions ---
def save_to_jsonl(data_list: List[Dict], file_path: Path) -> bool:
    """Saves a list of dictionaries to a JSONL file."""
    # Ensure the parent directory for the file exists, creating it if necessry.
    try:
        # This will create any missing parent directories along the path.
        # exist_ok=True means it won't raise an error if the directory already exists.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding=constants.UTF8) as f:
            for item in data_list:
                f.write(json.dumps(item) + "\n")
        logger.info(
            LM.SAVED_TO_JSONL_SUCCESS.format(count=len(data_list), path=file_path)
        )
        return True
    except Exception as e:
        # Log an error if any exception occurs during the file write operation.
        logger.error(LM.ERROR_SAVING_JSONL.format(path=file_path, error=e))
        return False


def load_from_jsonl(file_path: Path) -> List[Dict]:
    """
    Loads data from a JSONL file.
    Each line in the file is expected to be a valid JSON object.
    Invalid JSON lines are skipped with a warning.
    """
    if not file_path.exists():
        logger.info(LM.FILE_NOT_FOUND.format(path=file_path))
        return []

    data_list = []
    try:
        with open(file_path, "r", encoding=constants.UTF8) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Process non-empty lines only
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log a warning for lines that can't be parsed as JSON.
                        logger.warning(
                            LM.INVALID_JSON_LINE.format(
                                line_num=line_num, path=file_path, error=e
                            )
                        )
                        continue
        logger.info(
            LM.LOADED_FROM_JSONL_SUCCESS.format(count=len(data_list), path=file_path)
        )
        return data_list
    except Exception as e:
        # Log an error if any other exception occurs during file reading.
        logger.error(LM.ERROR_LOADING_JSONL.format(path=file_path, error=e))
        return []


def human_sleep(
    min_sec: int = config.MIN_SLEEP_SEC, max_sec: int = config.MAX_SLEEP_SEC
) -> None:
    """Simulate human-like random sleep."""
    # Generate a random delay within the specified min and max seconds.
    # This can be useful for mimicking human interaction or avoiding rate limits.
    delay = random.uniform(min_sec, max_sec)
    # Log the duration of the sleep.
    logger.info(LM.HUMAN_SLEEP_DELAY.format(delay=delay))
    time.sleep(delay)


# --- LLM and Embedding Model Configuration ---
def setup_llm_and_embed_models(
    embed_model_name: str = config.EMBEDDING_MODEL_NAME,
    llm_model_name: str = config.LLM_MODEL_NAME,
    google_api_key: Optional[str] = config.GOOGLE_API_KEY,
) -> Tuple[Optional[HuggingFaceEmbedding], Optional[GoogleGenAI]]:
    """
    Initializes and returns embedding and LLM instances.

    This function attempts to set up the HuggingFace embedding model and the
    GoogleGenAI LLM. It does NOT set global LlamaIndex Settings; that is
    left to the caller.

    Args:
        embed_model_name: Name of the HuggingFace embedding model to use.
        llm_model_name: Name of the Google GenAI model to use.
        google_api_key: The API key for Google GenAI services. If None, LLM setup will be skipped.

    Returns:
        A tuple containing the initialized HuggingFaceEmbedding model instance (or None on failure)
        and the GoogleGenAI LLM instance (or None if API key is missing or setup fails).
    """
    logger.info(LM.CONFIGURING_LLM_SETTINGS)
    embed_model = None
    llm = None

    try:
        # Set up embedding model
        # This model is used to convert text into numerical vectors.
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        logger.info(LM.USING_EMBEDDING_MODEL.format(model_name=embed_model_name))

        # Set up LLM
        # The LLM is used for tasks like text generation and summarization.
        if not google_api_key:
            logger.error(LM.GOOGLE_API_KEY_NOT_FOUND)
            return embed_model, None  # Return embed_model if it was successful

        logger.info(LM.USING_GEMINI_MODEL.format(model_name=llm_model_name))
        llm = GoogleGenAI(model_name=llm_model_name, api_key=google_api_key)

        # Perform a simple test call to verify API key and model access for LLM
        # Currently, we assume constructor success implies basic connectivity.
        try:
            # For GoogleGenAI, a simple way to check connectivity might be to list models or get model info if available.
            # If a direct "get_model_info" for the specific model isn't available or desired,
            # a lightweight call like listing available models can work.
            # For now, we assume the constructor itself or a subsequent call would raise an error if invalid.
            # Example: llm.list_models() # This might be too broad; a specific model check is better if available.
            # The previous llm.get_model_info(llm_model_name) was a placeholder.
            # A robust check depends on the GoogleGenAI library's API.
            # If the constructor succeeds, we'll assume basic connectivity for now.
            logger.info(f"GoogleGenAI initialized for model: {llm_model_name}")
        except Exception as e:
            logger.error(
                f"Failed to connect or verify Google GenAI model {llm_model_name}: {e}. Check API key and model name."
            )
            # return embed_model, None # LLM setup failed

        logger.info(LM.LLM_SETTINGS_CONFIGURED)  # More accurate: models initialised
        return embed_model, llm
    except Exception as e:
        # Log a general error if any part of the setup fails.
        logger.error(LM.FAILED_CONFIGURE_LLM.format(error=e))
        # Return whatever was successfully initialized up to this point.
        return embed_model, llm  # Return whatever was successfully initialized


def call_llm_for_summary(
    code: str, llm: GoogleGenAI, file_path_for_logging: str = "Unknown"
) -> str:
    """Generates a summary for the given code using the provided LLM."""
    # If the code string is empty or only whitespace, return a predefined message.
    if not code.strip():
        return LM.NO_CODE_FOR_SUMMARY

    # Format the user prompt with the provided code.
    user_prompt = constants.USER_PROMPT_TEMPLATE.format(code=code)
    # Construct the list of messages for the LLM chat.
    # This typically includes a system prompt and a user prompt.
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=constants.SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    try:
        # Record the start time to measure LLM response duration.
        start_time = time.time()
        # Call the LLM's chat method with the prepared messages.
        response = llm.chat(messages)
        # Calculate the duration of the LLM call.
        duration = time.time() - start_time
        logger.info(
            LM.LLM_SUMMARY_GENERATION_TIME.format(
                duration=duration, file_path=file_path_for_logging
            )
        )

        summary_text = ""
        # Check if the response and its content are valid.
        if response.message and response.message.content:
            summary_text = response.message.content.strip()
        else:
            # Log a warning if the LLM returns an empty response.
            logger.warning(
                f"LLM response content is empty for {file_path_for_logging}."
            )
            summary_text = "Summary generation failed: LLM returned empty content."

        # Introduce a small, human-like delay after the LLM call.
        human_sleep()  # Add delay
        logger.debug(LM.GENERATED_SUMMARY_DEBUG.format(file_path=file_path_for_logging))
        return summary_text
    except Exception as e:
        # Log an error if summary generation fails.
        logger.error(
            LM.FAILED_GENERATE_SUMMARY.format(file_path=file_path_for_logging, error=e)
        )
        # Return an error message if an exception occurs.
        return LM.SUMMARY_UNAVAILABLE_ERROR.format(error=str(e))


# --- Code Extraction ---
def extract_code_chunks(
    repo_path: Path,
    output_file: Optional[Path] = None,
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Extracts code chunks from files in a repository using language-specific CodeSplitters.

    It iterates through files, identifies supported file types, and uses
    LlamaIndex's CodeSplitter to break down code into manageable chunks.
    Supports caching of extracted chunks to a JSONL file.

    Args:
        repo_path: The Path object pointing to the root of the code repository.
        output_file: Optional Path to a JSONL file for caching/loading chunks.
        force_refresh: If True, re-extracts chunks even if a cache file exists.

    Returns:
        A list of dictionaries, where each dictionary represents a code chunk
        and contains the file path, code content, and metadata.
    """
    if not repo_path.is_dir():
        logger.error(LM.REPO_PATH_NOT_EXIST.format(path=repo_path))
        return []
    if output_file and output_file.exists() and not force_refresh:
        logger.info(LM.LOADING_CHUNKS_FROM_CACHE.format(path=output_file))
        cached_chunks = load_from_jsonl(
            output_file
        )  # Assumes load_from_jsonl is defined elsewhere
        if cached_chunks:
            logger.info(
                LM.LOADED_CHUNKS_FROM_CACHE_COUNT.format(count=len(cached_chunks))
            )
            return cached_chunks
        logger.info(LM.CACHE_EMPTY_OR_CORRUPT.format(path=output_file))

    logger.info(LM.EXTRACTING_CODE_CHUNKS.format(path=repo_path))
    chunks = []
    processed_files = 0

    # Iterating through all files in the repository
    # repo_path.rglob("*") recursively finds all files and directories.
    for file_path in repo_path.rglob("*"):
        file_suffix = (
            file_path.suffix.lower()
        )  # Use lower for case-insensitivity in map

        # Check if the file extension is in our supported list (derived from the map)
        # and ensure it's a file, not a directory.
        if file_suffix in config.SUPPORTED_FILE_SUFFIXES and file_path.is_file():
            # Get the language string for the CodeSplitter from our map
            language = config.FILE_EXTENSION_TO_LANGUAGE_MAP.get(file_suffix)
            # This language string is crucial for the CodeSplitter to use the correct parser.

            if not language:
                logger.warning(
                    f"No language mapping found for suffix {file_suffix} in file {file_path}, though suffix is supported. Skipping."
                )
                continue

            try:
                logger.debug(f"Processing file {file_path} with language: {language}")
                file_content = file_path.read_text(encoding=constants.UTF8)
                # Skip empty files.
                if not file_content.strip():
                    logger.debug(f"File {file_path} is empty. Skipping.")
                    continue

                # Instantiate CodeSplitter for the specific language
                # This splitter intelligently divides code based on syntax and structure.
                # Uses global chunking parameters from config.py
                code_splitter = CodeSplitter(
                    language=language,
                    chunk_lines=config.CODE_SPLITTER_CHUNK_LINES,
                    chunk_lines_overlap=config.CODE_SPLITTER_CHUNK_LINES_OVERLAP,
                    max_chars=config.CODE_SPLITTER_MAX_CHARS,
                )

                # Create a LlamaIndex Document object from the file content.
                doc = Document(
                    text=file_content,
                    metadata={
                        constants.HYBRID_CHUNK_DATA_KEYS["FILE_PATH_METADATA_KEY"]: str(
                            file_path
                        )
                    },
                )
                # Generate nodes (chunks) from the document.
                nodes = code_splitter.get_nodes_from_documents([doc])

                # Append structured chunk data to the list.
                for node in nodes:
                    chunks.append(
                        {
                            constants.CHUNK_DATA_KEYS["FILE"]: str(file_path),
                            constants.CHUNK_DATA_KEYS["CODE"]: node.text,
                            constants.CHUNK_DATA_KEYS["METADATA"]: node.metadata,
                        }
                    )
                processed_files += 1
            except ImportError as ie:
                # This error often means the tree-sitter grammar for the language isn't installed.
                logger.warning(
                    f"Could not process {file_path} for language '{language}'. Tree-sitter grammar might be missing: {ie}. Skipping file."
                )
            except Exception as e:
                # Catch-all for other errors during file processing.
                logger.warning(
                    f"Error processing file {file_path} with language '{language}': {e}. Skipping file."
                )
        elif file_path.is_file():
            logger.debug(
                # Log files that are skipped due to unsupported extensions.
                f"Skipping file {file_path} due to unsupported extension '{file_suffix}'."
            )

    logger.info(
        LM.PROCESSED_FILES_EXTRACTED_CHUNKS.format(
            files=processed_files, chunks=len(chunks)
        )
    )

    if not chunks:
        logger.warning(LM.NO_CHUNKS_EXTRACTED.format(path=repo_path))

    # Save the extracted chunks to the output file if specified and chunks were found.
    if output_file and chunks:
        if save_to_jsonl(
            chunks, output_file
        ):  # Assumes save_to_jsonl is defined elsewhere
            logger.info(
                LM.CACHED_CHUNKS_TO_FILE.format(count=len(chunks), path=output_file)
            )
    return chunks


# --- Chunk Summarization ---
def summarize_chunks(
    chunks: List[Dict],
    llm: GoogleGenAI,
    output_file: Optional[Path] = None,
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Generates summaries for a list of code chunks using the provided LLM.

    Each code chunk is sent to the LLM for summarization. The summary is then
    prepended to the original code to create a "hybrid chunk".
    Supports caching of these hybrid chunks to a JSONL file.

    Args:
        chunks: A list of dictionaries, where each dict represents a code chunk (from `extract_code_chunks`).
        llm: The initialized GoogleGenAI LLM instance to use for summarization.
        output_file: Optional Path to a JSONL file for caching/loading hybrid chunks.
        force_refresh: If True, re-summarizes chunks even if a cache file exists.

    Returns:
        A list of dictionaries, where each dictionary represents a hybrid chunk,
        containing an ID, the combined text (summary + code), and metadata.
    """
    if not chunks:
        logger.warning(LM.NO_CHUNKS_FOR_SUMMARIZATION)
        return []

    if output_file and output_file.exists() and not force_refresh:
        logger.info(LM.LOADING_HYBRID_CHUNKS_FROM_CACHE.format(path=output_file))
        cached_chunks = load_from_jsonl(output_file)
        if cached_chunks:
            logger.info(LM.LOADED_HYBRID_CHUNKS_COUNT.format(count=len(cached_chunks)))
            return cached_chunks
        logger.info(LM.CACHE_EMPTY_OR_CORRUPT.format(path=output_file))

    logger.info(LM.SUMMARIZING_CHUNKS_COUNT.format(count=len(chunks)))
    hybrid_chunks = []
    # Iterate through each chunk to generate its summary.
    for i, chunk_data in enumerate(chunks, 1):
        file_path_str = chunk_data.get(
            constants.CHUNK_DATA_KEYS["FILE"], "Unknown file"
        )
        logger.info(
            LM.PROCESSING_CHUNK_PROGRESS.format(
                current=i, total=len(chunks), file=file_path_str
            )
        )
        try:
            # Extract the code content from the chunk data.
            code_to_summarize = chunk_data[constants.CHUNK_DATA_KEYS["CODE"]]
            # Call the LLM to generate a summary for the code.
            summary = call_llm_for_summary(
                code_to_summarize, llm, file_path_for_logging=file_path_str
            )

            hybrid_text = f"{summary}\n\n{code_to_summarize}"  # Ensure summary is first
            # The summary is prepended to the code to provide context for embedding.
            # Create a unique ID for the chunk based on its content
            chunk_id = hashlib.md5(hybrid_text.encode(constants.UTF8)).hexdigest()

            # Preserve original metadata and file path
            original_metadata = chunk_data.get(
                constants.CHUNK_DATA_KEYS["METADATA"], {}
            )
            # Ensure 'file_path' is at the top level of the metadata for the Document
            # This is important for later retrieval and referencing.
            final_metadata = {
                constants.HYBRID_CHUNK_DATA_KEYS[
                    "FILE_PATH_METADATA_KEY"
                ]: file_path_str,
                constants.HYBRID_CHUNK_DATA_KEYS[
                    "ORIGINAL_METADATA"
                ]: original_metadata,
            }

            # Append the structured hybrid chunk data.
            hybrid_chunks.append(
                {
                    constants.HYBRID_CHUNK_DATA_KEYS["ID"]: chunk_id,
                    constants.HYBRID_CHUNK_DATA_KEYS["TEXT"]: hybrid_text,
                    constants.HYBRID_CHUNK_DATA_KEYS["METADATA"]: final_metadata,
                }
            )
        except Exception as e:
            # Log an error if summarization for a specific chunk fails.
            logger.error(
                LM.ERROR_SUMMARIZING_CHUNK.format(
                    current=i, file=file_path_str, error=e
                )
            )
            continue

    if not hybrid_chunks:
        logger.warning(LM.NO_CHUNKS_SUMMARIZED)

    logger.info(LM.SUCCESSFULLY_SUMMARIZED_CHUNKS.format(count=len(hybrid_chunks)))
    # Save the hybrid chunks to the output file if specified.
    if output_file and hybrid_chunks:
        if save_to_jsonl(hybrid_chunks, output_file):
            logger.info(
                LM.CACHED_HYBRID_CHUNKS_TO_FILE.format(
                    count=len(hybrid_chunks), path=output_file
                )
            )
    return hybrid_chunks


# --- Vector Index Management ---
def _clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares metadata for ChromaDB, ensuring values are basic types."""
    # ChromaDB has specific requirements for metadata field values.
    # They must be strings, integers, floats, or booleans, or lists of these.
    # This function attempts to convert/flatten complex metadata.
    cleaned = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            # Flatten nested dictionaries with a prefix, or handle as needed
            # This is a simple flattening strategy; more complex structures might need
            # a more sophisticated approach or selective dropping of keys.
            for nk, nv in v.items():
                # Ensure keys are unique if flattening, e.g., by prefixing
                # ChromaDB also expects string keys.
                flat_key = f"{k}_{nk}"
                if isinstance(nv, (str, int, float, bool)):
                    cleaned[flat_key] = nv
                else:
                    # Convert other types within nested dicts to string as a fallback.
                    cleaned[flat_key] = str(nv)  # Convert other types to string
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list):  # Chroma supports lists of strings/numbers/bools
            # Check if all items in the list are of supported basic types.
            if all(isinstance(item, (str, int, float, bool)) for item in v):
                cleaned[k] = v
            else:
                # If not, convert all items in the list to strings.
                cleaned[k] = [str(item) for item in v]  # Convert items to string
        else:
            cleaned[k] = str(v)  # Convert other types to string
    return cleaned


def build_vector_index(
    hybrid_chunks: List[Dict],
    embed_model: HuggingFaceEmbedding,  # Pass the model instance
    chroma_dir: str,
    collection_name: str,
) -> Optional[VectorStoreIndex]:
    """
    Builds or updates a ChromaDB vector index from hybrid chunks.

    This function takes the hybrid chunks (text with prepended summaries),
    generates embeddings for them using the provided `embed_model`,
    and stores them in a ChromaDB collection. It then creates a
    LlamaIndex VectorStoreIndex on top of this ChromaDB collection.

    Args:
        hybrid_chunks: A list of dictionaries, each representing a hybrid chunk.
        embed_model: The initialized HuggingFaceEmbedding model instance.
        chroma_dir: The directory path where ChromaDB should store its data.
        collection_name: The name of the collection within ChromaDB to use.

    Returns:
        A LlamaIndex VectorStoreIndex instance if successful, otherwise None.
    """
    if not hybrid_chunks:
        logger.error(LM.NO_HYBRID_CHUNKS_FOR_INDEXING)
        return None

    logger.info(
        LM.BUILDING_VECTOR_INDEX_COUNT.format(
            count=len(hybrid_chunks),
            collection_name=collection_name,
            chroma_dir=chroma_dir,
        )
    )

    # Create LlamaIndex Documents
    # These Document objects will be embedded and stored.
    documents: List[BaseNode] = []
    for chunk in hybrid_chunks:
        # Metadata must be cleaned for ChromaDB compatibility BEFORE Document creation if it's complex
        doc_metadata = chunk.get(constants.HYBRID_CHUNK_DATA_KEYS["METADATA"], {})
        cleaned_doc_metadata = _clean_metadata_for_chroma(doc_metadata)

        # The ID is important for potential updates and de-duplication.
        doc_id = chunk.get(constants.HYBRID_CHUNK_DATA_KEYS["ID"])
        documents.append(
            Document(
                id_=doc_id,  # LlamaIndex uses id_
                text=chunk[constants.HYBRID_CHUNK_DATA_KEYS["TEXT"]],
                metadata=cleaned_doc_metadata,
            )
        )

    if not documents:
        logger.error(LM.NO_LLAMA_DOCS_PREPARED)
        return None
    logger.info(LM.PREPARED_LLAMA_DOCS_COUNT.format(count=len(documents)))

    # Set up ChromaDB client and collection
    # PersistentClient ensures data is saved to disk.
    try:
        chroma_client = PersistentClient(path=chroma_dir)
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            # metadata={"hnsw:space": "cosine"} # Example: specify embedding distance
        )
        initial_count = chroma_collection.count()
        logger.info(
            LM.COLLECTION_INITIAL_COUNT.format(
                name=collection_name, count=initial_count
            )
        )
    except Exception as e:
        # Failure here means we can't store the vectors.
        logger.error(
            f"Failed to initialize ChromaDB client or collection '{collection_name}' at '{chroma_dir}': {e}"
        )
        return None

    # Explicitly generate embeddings for each document before adding to vector store
    # This is crucial as node.get_embedding() expects node.embedding to be set.
    # We iterate through each document and generate its embedding using the embed_model.
    logger.info(LM.GENERATING_EMBEDDINGS_COUNT.format(count=len(documents)))
    try:
        for doc in documents:
            if doc.embedding is None:  # Only generate if not already present
                # Use the passed embed_model instance directly
                # get_content(MetadataMode.EMBED) ensures only relevant text is embedded.
                doc.embedding = embed_model.get_text_embedding(
                    doc.get_content(metadata_mode=MetadataMode.EMBED)
                )
        logger.info(LM.EMBEDDINGS_GENERATED)
    except Exception as e:
        logger.error(f"Failed during explicit embedding generation: {e}", exc_info=True)
        return None

    # Create VectorStore instance
    # This acts as an adapter between LlamaIndex and ChromaDB.
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # The global Settings.embed_model is set by the caller (run_build_pipeline).
    # While it might be redundant here if all operations use the passed 'embed_model',
    # it's kept for safety for any deeper LlamaIndex calls.
    # TODO: Review if this global setting is strictly necessary here if embed_model is passed around.
    Settings.embed_model = embed_model

    logger.info(
        LM.ADDING_DOCS_TO_VECTOR_STORE.format(
            count=len(documents), collection_name=collection_name
        )
    )
    try:
        # Now that documents have their .embedding attribute set, this should work.
        # This adds the documents (with their pre-generated embeddings) to ChromaDB.
        vector_store.add(documents)
        logger.info(
            LM.ADD_DOCS_TO_VECTOR_STORE_SUCCESS.format(
                count=len(documents), collection_name=collection_name
            )
        )
    except Exception as e:
        logger.error(
            # Log detailed error if adding to vector store fails.
            LM.ERROR_VECTOR_STORE_ADD.format(
                count=len(documents), collection_name=collection_name, error=e
            ),
            exc_info=True,
        )
        return None

    count_after_add = chroma_collection.count()
    # Log the collection count after adding documents.
    logger.info(
        LM.COLLECTION_COUNT_AFTER_ADD.format(
            collection_name=collection_name, count=count_after_add
        )
    )

    expected_count = initial_count + len(documents)
    # Check for discrepancies in document count, which might indicate issues like ID collisions.
    # If IDs are not unique, ChromaDB might update existing entries instead of adding new ones.
    if count_after_add < initial_count + len(documents) and initial_count > 0:
        logger.warning(
            f"Collection count for '{collection_name}' ({count_after_add}) after adding documents "
            f"is less than initial ({initial_count}) + added ({len(documents)}). "
            "This might be due to duplicate IDs leading to updates instead of new entries."
        )
    elif count_after_add == initial_count and len(documents) > 0:
        logger.warning(
            f"Collection count for '{collection_name}' ({count_after_add}) did not change "
            f"after attempting to add {len(documents)} documents. Check for ID conflicts or other issues."
        )

    # Build the index object from the vector store
    logger.info(
        LM.BUILDING_INDEX_FROM_VECTOR_STORE.format(collection_name=collection_name)
    )
    try:
        # Pass embed_model here as well for consistency and explicitness
        # This creates the LlamaIndex object that can be queried.
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )
        logger.info(
            LM.VECTOR_INDEX_BUILT_SUCCESS.format(
                name=collection_name, chroma_dir=chroma_dir
            )
        )

        final_count = chroma_collection.count()
        # Final check on the collection count.
        logger.info(
            LM.FINAL_COLLECTION_COUNT.format(
                collection_name=collection_name, count=final_count
            )
        )
        if final_count == 0 and len(documents) > 0:
            logger.warning(
                LM.COLLECTION_EMPTY_AFTER_ADD_WARNING.format(
                    name=collection_name, attempted_count=len(documents)
                )
            )
        return index
    except Exception as e:
        logger.error(
            # Log failure if the final index object cannot be built.
            LM.FAILED_BUILD_VECTOR_INDEX.format(
                collection_name=collection_name, error=e
            ),
            exc_info=True,
        )
        return None


def load_vector_index(
    embed_model: HuggingFaceEmbedding,  # Pass the model instance
    chroma_dir: str,
    collection_name: str,
) -> Optional[VectorStoreIndex]:
    """
    Loads an existing vector index from ChromaDB.

    This function connects to an existing ChromaDB instance at `chroma_dir`,
    retrieves the specified `collection_name`, and then reconstructs a
    LlamaIndex VectorStoreIndex using that collection and the provided `embed_model`.

    Args:
        embed_model: The initialized HuggingFaceEmbedding model instance (must match the one used for building).
        chroma_dir: The directory path where ChromaDB data is stored.
        collection_name: The name of the ChromaDB collection to load.
    Returns:
        A LlamaIndex VectorStoreIndex instance if successful, otherwise None.
    """
    chroma_path = Path(chroma_dir)
    if not chroma_path.is_dir() or not any(chroma_path.iterdir()):
        logger.info(LM.CHROMA_DIR_EMPTY_OR_NOT_EXIST.format(path=chroma_dir))
        return None

    try:
        chroma_client = PersistentClient(path=chroma_dir)
        try:
            # Check if collection exists
            chroma_collection = chroma_client.get_collection(name=collection_name)
            # If the collection exists but is empty, it's not useful to load.
            if chroma_collection.count() == 0:
                logger.info(
                    LM.COLLECTION_EXISTS_BUT_EMPTY.format(
                        name=collection_name, chroma_dir=chroma_dir
                    )
                )
                return None
        except (
            Exception
        ) as e:  # Catches chromadb.errors.CollectionNotFoundError and others
            # This handles cases where the collection doesn't exist or other ChromaDB errors.
            logger.info(
                LM.COLLECTION_NOT_FOUND_OR_ERROR.format(
                    name=collection_name, chroma_dir=chroma_dir, e=e
                )
            )
            return None

        # Set global embed_model for LlamaIndex components that might need it
        # This is important for VectorStoreIndex.from_vector_store if it relies on global settings
        # for any internal operations, though we also pass it explicitly.
        Settings.embed_model = embed_model
        # Create the LlamaIndex ChromaVectorStore adapter.
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # Pass embed_model explicitly
        # Reconstruct the LlamaIndex VectorStoreIndex.
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

        logger.info(
            LM.LOADED_VECTOR_INDEX_SUCCESS.format(
                name=collection_name, chroma_dir=chroma_dir
            )
        )
        return index
    except Exception as e:
        logger.error(
            # Log any other errors during the index loading process.
            LM.FAILED_LOAD_VECTOR_INDEX.format(
                name=collection_name, chroma_dir=chroma_dir, error=e
            ),
            exc_info=True,
        )
        return None


def query_index(
    index: VectorStoreIndex, question: str, llm: GoogleGenAI
) -> Optional[Response]:
    """
    Queries the given LlamaIndex VectorStoreIndex with a question.

    It sets up a query engine with the provided LLM and a custom QA prompt template,
    then executes the query against the index.

    Args:
        index: The LlamaIndex VectorStoreIndex to query.
        question: The question string to ask the index.
        llm: The initialized GoogleGenAI LLM instance to use for synthesizing the answer.

    Returns:
        A LlamaIndex Response object containing the answer and source nodes, or None on failure.
    """
    if not question.strip():
        logger.error(LM.EMPTY_QUESTION_PROVIDED)
        return None

    try:
        # Ensure global LLM is set if any internal LlamaIndex components might rely on it.
        Settings.llm = llm

        # Create a PromptTemplate object from your string
        custom_qa_prompt = PromptTemplate(constants.CODE_QA_PROMPT_TEMPLATE_STR)

        logger.info(LM.QUERYING_INDEX_WITH_QUESTION.format(question=question))

        # Pass the custom prompt template to the query engine
        # The query engine handles retrieval from the vector store and answer synthesis by the LLM.
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=config.SIMILARITY_TOP_K,
            text_qa_template=custom_qa_prompt,  # <--- ADD THIS
            streaming=False,
        )
        start_time = time.time()
        # Execute the query.
        response_obj = query_engine.query(question)
        duration = time.time() - start_time

        # Log information about retrieved source nodes.
        if response_obj.source_nodes:
            logger.info(f"Retrieved {len(response_obj.source_nodes)} source_nodes.")
        else:
            logger.info("No source_nodes retrieved with the response.")

        # Check if a response was successfully generated.
        if response_obj and response_obj.response:
            logger.debug(f"Synthesized Answer: {str(response_obj.response)[:500]}...")
        else:
            logger.warning(LM.QUERY_RETURNED_NONE)
            return None

        logger.info(LM.QUERY_EXECUTION_TIME.format(duration=duration))
        return response_obj
    except Exception as e:
        # Log any errors that occur during the query process.
        logger.error(LM.ERROR_DURING_QUERY.format(error=e), exc_info=True)
        return None


#
# --- Pipeline Orchestrators ---


# --- Pipeline Orchestrators ---
def run_build_pipeline(
    repo_path: Path,
    embed_model: HuggingFaceEmbedding,
    llm: GoogleGenAI,
    chroma_dir: str,
    collection_name: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> Optional[VectorStoreIndex]:
    """
    Orchestrates the complete build pipeline for creating a vector index.

    This involves:
    1. Extracting code chunks from a repository.
    2. Summarizing these chunks using an LLM.
    3. Building a vector index from the summarized (hybrid) chunks.
    4. Optionally, verifying the persisted index.

    Args:
        repo_path: Path to the code repository.
        embed_model: Initialized HuggingFaceEmbedding model.
        llm: Initialized GoogleGenAI LLM.
        chroma_dir: Directory for ChromaDB storage.
        collection_name: Name for the ChromaDB collection.
        cache_dir: Directory for caching intermediate files (chunks, hybrid_chunks).
        force_refresh: If True, forces re-processing of chunks and summaries, ignoring caches.
    Returns:
        The built LlamaIndex VectorStoreIndex, or None on failure.
    """
    logger.info(
        LM.STARTING_BUILD_PIPELINE.format(
            repo=repo_path, chroma_path=chroma_dir, collection_name=collection_name
        )
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Define paths for cached intermediate files.
    chunks_file = cache_dir / config.CHUNKS_FILENAME
    hybrid_chunks_file = cache_dir / config.HYBRID_CHUNKS_FILENAME

    # Step 1: Extract code chunks
    # This step parses the repository and splits code files into smaller pieces.
    logger.info(LM.STEP_1_EXTRACT_CHUNKS)
    code_chunks = extract_code_chunks(
        repo_path, chunks_file, force_refresh=force_refresh
    )
    if not code_chunks:
        logger.error(LM.FAILED_EXTRACT_CODE_CHUNKS)
        return None

    # Step 2: Summarize chunks
    # Each code chunk is summarized by the LLM to create hybrid chunks.
    logger.info(LM.STEP_2_SUMMARIZE_CHUNKS)
    hybrid_chunks = summarize_chunks(
        code_chunks, llm, hybrid_chunks_file, force_refresh=force_refresh
    )
    if not hybrid_chunks:
        logger.error(LM.FAILED_SUMMARIZE_CHUNKS)
        return None

    # Step 3: Build vector index
    logger.info(LM.STEP_3_BUILD_INDEX)
    # The hybrid chunks are embedded and stored in the vector database.
    # Ensure global Settings are set for LlamaIndex components if they rely on them internally
    # This is done before calling build_vector_index which now also uses the passed embed_model explicitly.
    Settings.embed_model = embed_model
    Settings.llm = llm

    index = build_vector_index(hybrid_chunks, embed_model, chroma_dir, collection_name)
    if not index:
        logger.error(LM.FAILED_BUILD_INDEX_PIPELINE)
        return None

    # Step 4: Verification (optional, but good for sanity)
    # This step checks if the ChromaDB collection was created and contains data.
    logger.info(
        LM.VERIFYING_INDEX_PERSISTENCE.format(path=chroma_dir, name=collection_name)
    )
    try:
        verification_client = PersistentClient(path=chroma_dir)
        verification_collection = verification_client.get_collection(
            name=collection_name
        )
        count = verification_collection.count()
        if count > 0:
            logger.info(
                LM.VERIFICATION_SUCCESSFUL.format(name=collection_name, count=count)
            )
        else:
            logger.error(LM.VERIFICATION_FAILED_EMPTY.format(name=collection_name))
            logger.error(LM.VERIFICATION_FAILED_EMPTY_DETAIL)
            # Decide if this is a critical failure for the pipeline
            # For now, it's a warning, but could be made a hard failure.
            # return None #
    except Exception as e:
        logger.error(LM.VERIFICATION_FAILED_ERROR.format(name=collection_name, error=e))
        # Similar to above, this is currently a warning.
        # return None

    logger.info(
        LM.BUILD_PIPELINE_COMPLETED.format(name=collection_name, path=chroma_dir)
    )
    return index


def run_query_pipeline(
    question: str,
    embed_model: HuggingFaceEmbedding,
    llm: GoogleGenAI,
    chroma_dir: str,
    collection_name: str,
    index: Optional[VectorStoreIndex] = None,
) -> Optional[Response]:  # Updated return type
    """
    Orchestrates the query pipeline.

    If an `index` is not provided, it attempts to load an existing one from ChromaDB.
    Then, it queries the index with the given `question` using the `llm`.

    Args:
        question: The question to ask the RAG system.
        embed_model: Initialized HuggingFaceEmbedding model.
        llm: Initialized GoogleGenAI LLM.
        chroma_dir: Directory for ChromaDB storage.
        collection_name: Name of the ChromaDB collection.
        index: Optional pre-loaded LlamaIndex VectorStoreIndex. If None, it will be loaded.
    Returns:
        The LlamaIndex Response object containing the answer and source nodes, or None on failure.
    """
    logger.debug(
        LM.STARTING_QUERY_PIPELINE.format(
            question=question, path=chroma_dir, collection_name=collection_name
        )
    )

    # Set global settings for LlamaIndex components
    # This ensures that LlamaIndex internals use the correct models.
    Settings.embed_model = embed_model
    Settings.llm = llm

    if index is None:
        # If no index is passed, attempt to load it from the specified ChromaDB location.
        logger.info(LM.LOADING_VECTOR_INDEX)
        index = load_vector_index(embed_model, chroma_dir, collection_name)
        if not index:
            logger.error(LM.FAILED_LOAD_VECTOR_INDEX_PIPELINE)
            return None

    response_obj = query_index(
        index, question, llm
    )  # This now returns a Response object or None

    # Check if the query was successful and a response was generated.
    if response_obj and response_obj.response:
        logger.debug(LM.QUERY_PIPELINE_COMPLETED)
    else:
        logger.error(
            LM.QUERY_PIPELINE_FAILED
        )  # Or more specific if response_obj is None vs response_obj.response is None

    return response_obj


# --- Git Clone Function ---
def clone_github_repo(
    repo_url: str, clone_dir_str: str, force_clone: bool = False
) -> bool:
    """
    Clones a GitHub repository to a specified directory.

    Args:
        repo_url: The URL of the GitHub repository to clone.
        clone_dir_str: The local directory path (as a string) where the repository should be cloned.
        force_clone: If True, removes the `clone_dir` if it exists before cloning.
                     Otherwise, if the directory exists and is not empty, it assumes the repo
                     is already cloned and returns True.
    Returns:
        True if the repository is successfully cloned or already exists (and not force_clone),
        False otherwise.
    """
    clone_dir = Path(clone_dir_str)

    if clone_dir.exists() and any(clone_dir.iterdir()):  # Check if not empty
        if not force_clone:
            logger.info(LM.REPO_DIR_ALREADY_EXISTS.format(dir=clone_dir_str))
            return True  # Assuming existing repo is fine, no need to re-clone.
        else:
            logger.info(f"Force clone: Removing existing directory {clone_dir_str}...")
            import shutil

            try:
                shutil.rmtree(clone_dir_str)
                logger.info(f"Successfully removed {clone_dir_str}.")
            # Catching OSError for potential issues like permission errors during rmtree.
            except OSError as e:
                logger.error(
                    f"Error removing directory {clone_dir_str}: {e}. Please remove manually."
                )
                return False

    # Ensure parent directory exists if clone_dir is nested
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    # This prevents errors if the parent path doesn't exist.

    logger.info(LM.CLONING_REPO_TO_DIR.format(url=repo_url, dir=clone_dir_str))
    command = ["git", "clone", repo_url, clone_dir_str]

    try:
        # Execute the git clone command.
        # check=True will raise CalledProcessError if git clone fails.
        process = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding=constants.UTF8
        )
        logger.info(LM.REPO_CLONED_SUCCESS.format(dir=clone_dir_str))
        # Log stdout from the git command for debugging if needed.
        if process.stdout:
            logger.debug(LM.GIT_OUTPUT_STDOUT.format(output=process.stdout.strip()))
        return True
    except subprocess.CalledProcessError as e:
        # This block handles errors specifically from the git command failing.
        logger.error(LM.ERROR_GIT_CLONE)
        logger.error(LM.GIT_COMMAND.format(command=" ".join(e.cmd)))
        logger.error(LM.GIT_RETURN_CODE.format(code=e.returncode))
        if e.stdout:
            logger.error(LM.GIT_STDOUT.format(output=e.stdout.strip()))
        if e.stderr:
            logger.error(LM.GIT_STDERR.format(output=e.stderr.strip()))
        logger.error(LM.FAILED_CLONE_REPO_DETAIL.format(dir=clone_dir_str))
        return False
    except FileNotFoundError:
        # This error occurs if the 'git' command is not found in the system's PATH.
        logger.error(LM.GIT_CMD_NOT_FOUND)
        return False
    except Exception as e:
        # Catch any other unexpected errors during the cloning process.
        logger.error(LM.UNEXPECTED_CLONE_ERROR.format(error=e))
        return False


# --- Main Entry Point ---
def main():
    """Main entry point for the refactored pipeline."""
    # This function orchestrates the entire process:
    # 1. Sets up paths and configurations.
    # 2. Initializes LLM and embedding models.
    # 3. Clones the target repository.
    # 4. Loads an existing vector index or builds a new one.
    # 5. Queries the index with predefined questions.
    # --- Path Configuration using BASE_OUTPUT_DIR_NAME from config.py ---
    script_base_dir = Path(__file__).resolve().parent
    base_output_path = script_base_dir / config.BASE_OUTPUT_DIR_NAME

    # Construct full, absolute paths from the base directory
    repo_path = base_output_path / config.DEFAULT_REPO_SUBDIR
    chroma_dir = str(
        base_output_path / config.DEFAULT_CHROMA_SUBDIR
    )  # Chroma client needs a string path
    cache_dir = base_output_path / config.DEFAULT_CACHE_SUBDIR

    # Other configurations
    collection_name = config.DEFAULT_COLLECTION_NAME

    # Ensure all output directories exist before we start
    # This prevents errors later if directories are missing.
    repo_path.mkdir(parents=True, exist_ok=True)
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"All outputs will be stored under the base directory: {base_output_path}"
    )

    # --- Questions for Querying (from config.py) ---
    questions = config.DEFAULT_QUESTIONS
    if not questions or len(questions) < 1:
        logger.warning(LM.NO_QUESTIONS_PROVIDED)

    # --- 1. Setup LLM and Embedding Models ---
    # These models are fundamental to the RAG pipeline.
    embed_model, llm = setup_llm_and_embed_models()
    if not embed_model or not llm:
        logger.error(LM.FAILED_SETUP_LLM_EXITING)
        return

    # Explicitly set global LlamaIndex settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    # This makes the models available to LlamaIndex components that might use global settings.
    logger.info("Global LlamaIndex Settings for embed_model and llm have been set.")

    # --- 2. Clone Repository into the designated subdirectory ---
    clone_target_dir = str(repo_path)
    # The repository content will be the source for our RAG system.
    if not clone_github_repo(
        config.DEFAULT_REPO_URL, clone_target_dir, force_clone=False
    ):
        logger.error(f"Failed to clone repository to {clone_target_dir}. Exiting.")
        return

    if not any(repo_path.iterdir()):
        # If cloning failed or the repo is empty, we can't proceed.
        logger.error(
            f"Repository directory {repo_path} is empty after clone attempt. Cannot proceed."
        )
        return

    # --- 3. Try to Load Existing Index or Build New One ---
    logger.info(
        # Attempt to load an index if one was previously built to save time.
        LM.ATTEMPTING_LOAD_EXISTING_INDEX.format(
            path=chroma_dir, collection_name=collection_name
        )
    )
    index = load_vector_index(embed_model, chroma_dir, collection_name)

    if index is None:
        logger.info(
            # If no existing index is found, or loading fails, build a new one.
            LM.NO_INDEX_FOUND_BUILDING_NEW.format(
                path=chroma_dir, collection_name=collection_name
            )
        )
        index = run_build_pipeline(
            repo_path=repo_path,
            embed_model=embed_model,
            llm=llm,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            cache_dir=cache_dir,
            force_refresh=config.FORCE_REFRESH_DEFAULT,
        )
        if index is None:
            logger.error(LM.FAILED_BUILD_INDEX_EXITING)
            return
    else:
        logger.info(
            LM.LOADED_EXISTING_INDEX_SUCCESS.format(collection_name=collection_name)
        )

    # --- 4. Query the Index ---
    # If the index is available (either loaded or newly built), proceed to query.
    if index:
        for question in questions:
            print(f"\nQ: {question}")  # Changed to "Q: "

            # Run the query pipeline for each question.
            response_obj = (
                run_query_pipeline(  # This now returns a Response object or None
                    question=question,
                    embed_model=embed_model,
                    llm=llm,
                    chroma_dir=chroma_dir,
                    collection_name=collection_name,
                    index=index,
                )
            )

            # Process and print the response.
            if response_obj and response_obj.response:
                print(f"A: {response_obj.response}")  # Changed to "A: "
                # If source nodes are available, print them as references.
                if response_obj.source_nodes:
                    print("--- References ---")
                    # Iterate through the source nodes, which provide context for the answer.
                    for i, source_node_with_score in enumerate(
                        response_obj.source_nodes, 1
                    ):
                        node = source_node_with_score.node
                        file_path = node.metadata.get(
                            constants.HYBRID_CHUNK_DATA_KEYS["FILE_PATH_METADATA_KEY"],
                            "Unknown file",
                        )
                        score = (
                            source_node_with_score.score
                            if source_node_with_score.score is not None
                            else "N/A"
                        )

                        # Displaying a snippet of the node text (e.g., first 300 characters)
                        text_snippet = (
                            node.text[:300]
                            .strip()
                            .replace("\n", " ")
                            # Make snippet more compact for printing.
                        )  # Make it more compact for printing

                        print(
                            f"REF:{i} - File: {file_path} (Score: {score if isinstance(score, str) else f'{score:.4f}'})"
                        )
                        print(f"   Snippet: {text_snippet}...")
                    print("------------------")
                else:
                    print("   (No specific source references found for this answer)")
            # Handle cases where no answer could be retrieved.
            else:
                logger.error(LM.FAILED_GET_QUERY_RESPONSE.format(question=question))
                print(
                    "A: Could not retrieve an answer for this question."
                )  # Changed to "A: "
    else:
        # If the index is not available after build/load attempts, log an error.
        logger.error("Index is not available. Cannot run queries.")

    logger.info(LM.MAIN_PROCESSING_COMPLETE)


if __name__ == "__main__":
    main()
