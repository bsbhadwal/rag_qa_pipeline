import os
import sys
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

# Add the parent directory to sys.path to allow imports from pipeline, config, etc.
# This assumes 'web' is a subdirectory of your main project where pipeline.py resides.
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Now import from your project files - assuming main logic file is pipeline.py
try:
    from llama_index.core.settings import Settings  # For setting models globally

    import config
    import constants
    import pipeline  # Changed from pipeline_logic
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(
        f"Ensure pipeline.py, config.py, and constants.py are in the directory: {parent_dir}"
    )
    print(f"Current sys.path: {sys.path}")
    pass


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for flash messages, good practice

# --- Global variables to store models and index ---
embed_model_global = None
llm_global = None
index_global = None
models_loaded = False


def initialize_models_and_index():
    """
    Initializes the embedding model, LLM, and loads/builds the vector index.
    """
    global embed_model_global, llm_global, index_global, models_loaded

    if models_loaded:
        print("Models and index already loaded.")
        return

    print("Initializing models and index for the web app...")

    project_root_dir = parent_dir
    base_output_path = project_root_dir / config.BASE_OUTPUT_DIR_NAME

    repo_path = base_output_path / config.DEFAULT_REPO_SUBDIR
    chroma_dir_str = str(base_output_path / config.DEFAULT_CHROMA_SUBDIR)
    cache_dir = base_output_path / config.DEFAULT_CACHE_SUBDIR
    collection_name = config.DEFAULT_COLLECTION_NAME

    repo_path.mkdir(parents=True, exist_ok=True)
    Path(chroma_dir_str).mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output path for web app context: {base_output_path}")

    # 1. Setup LLM and Embedding Models
    embed_model, llm = (
        pipeline.setup_llm_and_embed_models()
    )  # Changed from pipeline_logic
    if not embed_model or not llm:
        print("ERROR: Failed to setup LLM and/or embedding model for the web app.")
        return

    embed_model_global = embed_model
    llm_global = llm

    Settings.embed_model = embed_model_global
    Settings.llm = llm_global
    print(
        "Global LlamaIndex Settings for embed_model and llm have been set for web app."
    )

    # 2. Clone Repository
    clone_target_dir = str(repo_path)
    if not pipeline.clone_github_repo(
        config.DEFAULT_REPO_URL, clone_target_dir, force_clone=False
    ):  # Changed
        print(
            f"ERROR: Failed to clone repository to {clone_target_dir} for the web app."
        )
        return

    if not any(repo_path.iterdir()):
        print(
            f"ERROR: Repository directory {repo_path} is empty after clone attempt. Cannot proceed."
        )
        return

    # 3. Load or Build Index
    print(
        f"Attempting to load existing index from: {chroma_dir_str}, collection: {collection_name}"
    )
    index = pipeline.load_vector_index(
        embed_model_global, chroma_dir_str, collection_name
    )  # Changed

    if index is None:
        print(
            f"No existing index found. Building new index for web app... (This might take a while)"
        )
        index = pipeline.run_build_pipeline(  # Changed
            repo_path=repo_path,
            embed_model=embed_model_global,
            llm=llm_global,
            chroma_dir=chroma_dir_str,
            collection_name=collection_name,
            cache_dir=cache_dir,
            force_refresh=False,
        )
        if index is None:
            print("ERROR: Failed to build index for the web app.")
            return
    else:
        print("Successfully loaded existing index for web app.")

    index_global = index
    models_loaded = True
    print("Models and index initialization complete.")


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page for asking questions.
    """
    if not models_loaded:
        try:
            initialize_models_and_index()
            if not models_loaded:
                return (
                    "Error: Could not initialize AI models or index. Please check server logs.",
                    500,
                )
        except Exception as e:
            print(f"Critical error during initialization: {e}")
            return f"Server initialization error: {e}", 500

    if request.method == "POST":
        question_text = request.form.get(
            "question"
        )  # Renamed to avoid conflict with 'question' variable in template context
        if not question_text:
            return render_template("index.html", error="Please enter a question.")

        if not embed_model_global or not llm_global or not index_global:
            return (
                "Error: AI Models or Index not loaded. Please check server logs.",
                500,
            )

        print(f"Received question: {question_text}")

        project_root_dir = parent_dir
        base_output_path = project_root_dir / config.BASE_OUTPUT_DIR_NAME
        chroma_dir_str = str(base_output_path / config.DEFAULT_CHROMA_SUBDIR)
        collection_name = config.DEFAULT_COLLECTION_NAME

        response_obj = pipeline.run_query_pipeline(  # Changed
            question=question_text,
            embed_model=embed_model_global,
            llm=llm_global,
            chroma_dir=chroma_dir_str,
            collection_name=collection_name,
            index=index_global,
        )

        answer = None
        references = []
        if response_obj and response_obj.response:
            answer = response_obj.response
            if response_obj.source_nodes:
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
                    try:
                        relative_file_path = Path(file_path).relative_to(
                            project_root_dir
                            / config.BASE_OUTPUT_DIR_NAME
                            / config.DEFAULT_REPO_SUBDIR
                        )
                    except ValueError:
                        relative_file_path = file_path

                    text_snippet = node.text[:300].strip().replace("\n", " ") + "..."
                    references.append(
                        {
                            "id": i,
                            "file": str(relative_file_path),
                            "score": (
                                f"{score:.4f}" if isinstance(score, float) else score
                            ),
                            "snippet": text_snippet,
                        }
                    )
        else:
            answer = "Sorry, I couldn't find an answer to your question or an error occurred."

        return render_template(
            "index.html",
            question_text=question_text,
            answer=answer,
            references=references,
        )  # Pass question_text

    return render_template("index.html")


if __name__ == "__main__":
    print(
        "Flask app starting... Models will be initialized on the first request to '/'."
    )
    app.run(debug=True, host="0.0.0.0", port=5001)
