# constants.py

# --- LLM Prompts ---
SYSTEM_PROMPT = """You are a code analysis assistant. Your task is to read Python functions and explain them in clear, structured English. Focus on precision and completeness. When given a code snippet, analyze it and provide a summary using the following format:
* **Purpose:** A brief explanation of what the function is intended to do.
* **Inputs:** A list of all input parameters and their expected types or roles.
* **Outputs:** A description of the return value(s), including type and meaning.
* **Exceptions:** Any exceptions the function may raise and under what conditions.
* **Notes:** Any additional observations, such as edge cases, dependencies, or assumptions.
"""

USER_PROMPT_TEMPLATE = """Explain what this function does in structured English:
###
{code}
###
Return a summary with:
- Purpose:
- Inputs:
- Outputs:
- Exceptions:
- Notes:
"""

CODE_QA_PROMPT_TEMPLATE_STR = (
    "You are an expert AI programming assistant. Your task is to answer questions about a given codebase. "
    "Use the provided code snippets (context) to answer the user's question. "
    "If the context does not contain enough information to answer the question, clearly state that. "
    "Do not make up information not present in the provided code snippets.\n"
    "When explaining code, be clear and concise.\n"
    "---------------------\n"
    "Context from the codebase:\n"
    "{context_str}\n"
    "---------------------\n"
    "User's Question: {query_str}\n"
    "Answer: "
)


# --- Log Message Templates ---
# Using a class or dictionary for organization
class LogMessages:
    # File Operations
    SAVED_TO_JSONL_SUCCESS = "Successfully saved {count} items to {path}"
    ERROR_SAVING_JSONL = "Error saving data to {path}: {error}"
    FILE_NOT_FOUND = "File not found: {path}"
    LOADED_FROM_JSONL_SUCCESS = "Successfully loaded {count} items from {path}"
    ERROR_LOADING_JSONL = "Error loading data from {path}: {error}"
    INVALID_JSON_LINE = "Invalid JSON on line {line_num} in {path}: {error}"
    CREATED_DIRECTORY = "Created directory: {path}"

    # Delays
    HUMAN_SLEEP_DELAY = "[DELAY] Sleeping for {delay:.2f} seconds..."

    # LLM and Embedding Model Setup
    CONFIGURING_LLM_SETTINGS = "Configuring LlamaIndex settings..."
    GOOGLE_API_KEY_NOT_FOUND = (
        "GOOGLE_API_KEY not found in environment variables. Please set it."
    )
    USING_GEMINI_MODEL = "Using Gemini model: {model_name}"
    USING_EMBEDDING_MODEL = "Using Embedding model: {model_name}"
    LLM_SETTINGS_CONFIGURED = (
        "LlamaIndex settings configured successfully with LLM and Embedding model."
    )
    FAILED_CONFIGURE_LLM = "Failed to configure LlamaIndex settings: {error}"

    # Summarization
    NO_CODE_FOR_SUMMARY = "No code provided for summary."
    LLM_NOT_CONFIGURED_FOR_SUMMARY = "Summary unavailable: LLM not configured. Ensure setup_llm_and_embed_models() was called and successful."
    LLM_SUMMARY_GENERATION_TIME = (
        "LLM took {duration:.2f}s to generate summary for chunk from {file_path}"
    )
    GENERATED_SUMMARY_DEBUG = "Generated summary for code chunk from {file_path}"
    FAILED_GENERATE_SUMMARY = (
        "Failed to generate summary for chunk from {file_path}: {error}"
    )
    SUMMARY_UNAVAILABLE_ERROR = "Summary unavailable due to error: {error}"

    # Code Extraction
    REPO_PATH_NOT_EXIST = "Repository path does not exist: {path}"
    LOADING_CHUNKS_FROM_CACHE = "Loading chunks from cache: {path}"
    LOADED_CHUNKS_FROM_CACHE_COUNT = "Loaded {count} chunks from cache."
    CACHE_EMPTY_OR_CORRUPT = "Cache file {path} empty or corrupted, re-extracting..."
    EXTRACTING_CODE_CHUNKS = "Extracting code chunks from: {path}"
    ERROR_PROCESSING_FILE = "Error processing file {path}: {error}"
    PROCESSED_FILES_EXTRACTED_CHUNKS = (
        "Processed {files} files, extracted {chunks} chunks."
    )
    CACHED_CHUNKS_TO_FILE = "Cached {count} chunks to: {path}"
    NO_CHUNKS_EXTRACTED = "No code chunks were extracted from {path}."

    # Chunk Summarization
    NO_CHUNKS_FOR_SUMMARIZATION = "No chunks provided for summarization."
    LLM_NOT_CONFIGURED_WARNING = (
        "LLM not configured. Call setup_llm_and_embed_models() first."
    )
    LOADING_HYBRID_CHUNKS_FROM_CACHE = "Loading hybrid chunks from cache: {path}"
    LOADED_HYBRID_CHUNKS_COUNT = "Loaded {count} hybrid chunks from cache."
    SUMMARIZING_CHUNKS_COUNT = "Summarizing {count} code chunks..."
    PROCESSING_CHUNK_PROGRESS = "Processing chunk {current}/{total} from {file}"
    ERROR_SUMMARIZING_CHUNK = "Error summarizing chunk {current} from {file}: {error}"
    SUCCESSFULLY_SUMMARIZED_CHUNKS = "Successfully summarized {count} chunks."
    CACHED_HYBRID_CHUNKS_TO_FILE = "Cached {count} hybrid chunks to: {path}"
    NO_CHUNKS_SUMMARIZED = "No chunks were summarized."

    # Vector Index Management
    NO_HYBRID_CHUNKS_FOR_INDEXING = "No hybrid chunks provided for index building."
    EMBED_MODEL_NOT_CONFIGURED = "Embedding model not configured. Ensure setup_llm_and_embed_models() was called and successful."
    BUILDING_VECTOR_INDEX_COUNT = "Building vector index with {count} hybrid chunks for collection '{collection_name}' in '{chroma_dir}'."
    PREPARED_LLAMA_DOCS_COUNT = "Prepared {count} LlamaIndex Documents for indexing."
    NO_LLAMA_DOCS_PREPARED = (
        "No LlamaIndex Documents were prepared. Cannot build index."
    )
    GENERATING_EMBEDDINGS_COUNT = "Generating embeddings for {count} documents..."
    EMBEDDINGS_GENERATED = "Embeddings generated for all documents."
    COLLECTION_INITIAL_COUNT = (
        "Collection '{name}' initial count (before LlamaIndex build): {count}"
    )
    ADDING_DOCS_TO_VECTOR_STORE = "Adding {count} documents to the vector store for collection '{collection_name}'..."
    ADD_DOCS_TO_VECTOR_STORE_SUCCESS = "Successfully added {count} documents to the vector store for collection '{collection_name}'."
    ERROR_VECTOR_STORE_ADD = "Error during vector_store.add() for {count} documents into '{collection_name}': {error}"
    COLLECTION_COUNT_AFTER_ADD = (
        "Collection '{collection_name}' count after adding documents: {count}"
    )
    COLLECTION_COUNT_MISMATCH = "Collection '{collection_name}' count ({actual}) after adding documents does not match expected count ({expected}). Initial: {initial}, Added: {added}."
    BUILDING_INDEX_FROM_VECTOR_STORE = "Attempting to build VectorStoreIndex from the vector_store for collection '{collection_name}'..."
    VECTOR_INDEX_BUILT_SUCCESS = (
        "Successfully built vector index for collection '{name}' in '{chroma_dir}'."
    )
    FINAL_COLLECTION_COUNT = "[INFO] Final collection count for '{collection_name}' from original object: {count}"
    COLLECTION_EMPTY_AFTER_ADD_WARNING = "Collection '{name}' count is 0 after attempting to add documents, despite attempting to add {attempted_count} documents. This suggests documents were not successfully added to the ChromaDB collection."
    FAILED_BUILD_VECTOR_INDEX = (
        "Failed to build vector index for collection '{collection_name}': {error}"
    )
    CHROMA_DIR_EMPTY_OR_NOT_EXIST = (
        "ChromaDB directory '{path}' does not exist or is empty. Cannot load index."
    )
    COLLECTION_EXISTS_BUT_EMPTY = (
        "Collection '{name}' exists in '{chroma_dir}' but is empty. Cannot load index."
    )
    COLLECTION_NOT_FOUND_OR_ERROR = "Collection '{name}' not found in '{chroma_dir}' or error accessing: {e}. Cannot load index."
    LOADED_VECTOR_INDEX_SUCCESS = (
        "Successfully loaded vector index for collection '{name}' from '{chroma_dir}'."
    )
    FAILED_LOAD_VECTOR_INDEX = "Failed to load vector index for collection '{name}' from '{chroma_dir}': {error}"

    # Query Engine
    NO_INDEX_FOR_QUERYING = "No index provided for querying."
    EMPTY_QUESTION_PROVIDED = "Empty question provided. Cannot query."
    QUERYING_INDEX_WITH_QUESTION = "Querying index with: '{question}'"
    QUERY_EXECUTION_TIME = "Query executed successfully in {duration:.2f}s."
    QUERY_RESPONSE = "Response: {response}"  # Note: response itself might be long
    ERROR_DURING_QUERY = "Error during query execution: {error}"
    QUERY_RETURNED_NONE = "Query returned no response."

    # Pipeline Orchestration
    STARTING_BUILD_PIPELINE = "Starting build pipeline for repository: {repo}. ChromaDB path: {chroma_path}, Collection: {collection_name}."
    STEP_1_EXTRACT_CHUNKS = "Step 1: Extracting code chunks..."
    FAILED_EXTRACT_CODE_CHUNKS = "Build pipeline failed: Failed to extract code chunks."
    STEP_2_SUMMARIZE_CHUNKS = "Step 2: Summarizing code chunks..."
    FAILED_SUMMARIZE_CHUNKS = "Build pipeline failed: Failed to summarize chunks."
    STEP_3_BUILD_INDEX = "Step 3: Building vector index..."
    FAILED_BUILD_INDEX_PIPELINE = "Build pipeline failed: Failed to build vector index."
    VERIFYING_INDEX_PERSISTENCE = "Verifying index persistence in '{path}' for collection '{name}' immediately after build..."
    VERIFICATION_SUCCESSFUL = (
        "Verification successful: Collection '{name}' has {count} items on disk."
    )
    VERIFICATION_FAILED_EMPTY = "VERIFICATION FAILED: Collection '{name}' is EMPTY on disk immediately after build (read by new client)."
    VERIFICATION_FAILED_EMPTY_DETAIL = "This indicates an issue with ChromaDB not persisting data correctly or data not being added properly."
    VERIFICATION_FAILED_ERROR = "VERIFICATION FAILED: Error accessing collection '{name}' for verification: {error}"
    BUILD_PIPELINE_COMPLETED = "Build pipeline completed successfully. Index for '{name}' should be persisted in '{path}'."
    BUILD_PIPELINE_FAILED = "Build pipeline failed."

    STARTING_QUERY_PIPELINE = "Starting query pipeline with question: '{question}'. ChromaDB path: {path}, Collection: {collection_name}."
    LOADING_VECTOR_INDEX = "Loading vector index for query pipeline..."
    FAILED_LOAD_VECTOR_INDEX_PIPELINE = (
        "Query pipeline failed: Failed to load vector index."
    )
    QUERY_PIPELINE_COMPLETED = "Query pipeline completed successfully."
    QUERY_PIPELINE_FAILED = "Query pipeline failed."

    # Main/Setup
    FAILED_SETUP_LLM_EXITING = "Failed to setup LLM and/or embedding model. Exiting."
    ATTEMPTING_LOAD_EXISTING_INDEX = (
        "Attempting to load existing index from: {path}, collection: {collection_name}."
    )
    NO_INDEX_FOUND_BUILDING_NEW = "No existing index found or index is empty in {path} for collection {collection_name}. Building new index..."
    FAILED_BUILD_INDEX_EXITING = "Failed to build index. Exiting."
    LOADED_EXISTING_INDEX_SUCCESS = (
        "Successfully loaded existing index for collection: {collection_name}."
    )
    FAILED_GET_QUERY_RESPONSE = "Failed to get response from query for: '{question}'"
    MAIN_PROCESSING_COMPLETE = "Main processing complete."

    # Git Clone
    CLONING_REPO_TO_DIR = "Attempting to clone repository: {url} into {dir}"
    REPO_CLONED_SUCCESS = "Repository cloned successfully to {dir}"
    GIT_OUTPUT_STDOUT = "Git output (stdout):\n{output}"
    ERROR_GIT_CLONE = "Error during git clone."
    GIT_COMMAND = "Command: {command}"
    GIT_RETURN_CODE = "Return code: {code}"
    GIT_STDOUT = "Stdout:\n{output}"
    GIT_STDERR = "Stderr:\n{output}"
    FAILED_CLONE_REPO_DETAIL = "Failed to clone repository. The directory '{dir}' might already exist and not be an empty directory, or the URL might be invalid, or you might lack permissions."
    GIT_CMD_NOT_FOUND = "Error: Git command not found. Please ensure Git is installed and in your system's PATH."
    UNEXPECTED_CLONE_ERROR = "An unexpected error occurred during cloning: {error}"
    REPO_DIR_ALREADY_EXISTS = "Clone directory '{dir}' already exists. Skipping clone. Use --force-clone to re-clone."
    NO_QUESTIONS_PROVIDED = "No questions provided in config.py. Assuming you only need to clone the repository and index."


# --- Dictionary Keys ---
# For consistent access to dictionary fields
CHUNK_DATA_KEYS = {
    "FILE": "file",
    "CODE": "code",
    "METADATA": "metadata",
}

HYBRID_CHUNK_DATA_KEYS = {
    "ID": "id",
    "TEXT": "text",
    "METADATA": "metadata",
    "ORIGINAL_METADATA": "original_metadata",  # Key within the nested metadata
    "FILE_PATH_METADATA_KEY": "file_path",  # Key for file_path in Document metadata
}

# --- Encoding ---
UTF8 = "utf-8"

# --- ChromaDB Client ---
PERSISTENT_CLIENT = "PersistentClient"  # For logging or reference

# --- LlamaIndex ---
LLAMA_INDEX_NODE_ID_KEY = "id_"  # Field name for LlamaIndex Document id
