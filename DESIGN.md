## Design Decisions & Library Choices: RAG-based Code Q\&A System

### 1. Problem Statement

Developers often need to understand unfamiliar codebases quickly. Whether for onboarding, debugging, or enhancing functionality, navigating complex repositories is time-consuming and mentally taxing. A system that can ingest a codebase and answer questions about it using natural language would significantly accelerate developer productivity and comprehension.

### 2. Existing Solutions

A range of tutorials and repos attempt to solve this using RAG (Retrieval-Augmented Generation). A typical example:
*[Build a RAG System for Your Codebase in 5 Easy Steps](https://medium.com/google-cloud/build-a-rag-system-for-your-codebase-in-5-easy-steps-a3506c10599b)*

### 3. Shortcomings of Existing Approaches

Despite the hype, most of these solutions fall short due to two core issues:

#### Naive Chunking

Code is often chunked as if it were prose—i.e., split by lines, tokens, or sentences—without understanding semantic units like functions or classes.

#### Language Mismatch

Code is written in programming languages. Questions are asked in natural language (usually English). Bridging the semantic gap between the two is non-trivial, yet most implementations ignore it entirely.

### 4. Problems We Solve

* True semantic understanding of code
* Bridging code and natural language for Q\&A

### 5. Our Solution

We solve the two core problems by introducing **Hybrid Semantic Chunking**:

#### Semantic Chunking with CodeSplitter

We use LlamaIndex's CodeSplitter to break the code into semantically meaningful units—functions and classes—instead of arbitrary text blocks.

#### Hybrid Chunk Enrichment with LLMs

Each code chunk is passed through an LLM that generates a structured, natural-language summary, including:

* What the function/class does
* Inputs and outputs
* Exceptions raised
* Return types
* Side effects or usage hints

These enriched summaries are combined with the original code to form hybrid chunks, which are embedded and stored in a vector database for retrieval and question answering.

### 6. Tech Stack

| Purpose                 | Library / Tool            |
| ----------------------- | ------------------------- |
| RAG Framework           | LlamaIndex                |
| Code Chunking           | CodeSplitter (LlamaIndex) |
| Embedding Model         | all-MiniLM-L6-v2          |
| Vector Store            | ChromaDB (local)          |
| LLM (for summarization) | Gemini Flash (API)        |
| Chunk Cache Format      | JSONL                     |

### 7. Non-Functional Requirements Met

* Local embeddings to reduce external API costs
* Local vector store using ChromaDB for efficient, persistent storage
* JSONL-based chunk caching to avoid reprocessing the same repo
* Runs on a consumer-grade desktop (Intel N100, 16GB RAM, no GPU, 1MBPS internet); and yet answers most queries in under 5 seconds
* Robust logging covering errors, informational messages, and debug output
* Internationalization-ready with constants in a dedicated `constants.py` file
* Config-driven architecture with `config.py` for easy adaptation
* Our only remote dependency is Google-Gemini LLM which provides a generous free tier. However, the code can be run with a local LLM like DeepSeek 1.5. The 8GB models can be run on any consumer hardware with a single graphics card for real-time answers. This is why we are using a RAG-centric platform like LlamaIndex.

### 8. Architectural and Design Choices Explained

#### 8.1. Procedural (Functional) Architecture

**Decision:** The core pipeline (`pipeline.py`) is implemented using a procedural, function-driven approach.

**Justifications:**

* **Rapid Prototyping (PRD: 2-day timeline)**
* **Simplicity for Defined Scope**
* **Modularity through Functions**
* **Dependency Injection for Testability**
* **Clear Data Flow**
* **Minimal Shared State**
* **Focus on Core RAG Logic**
* **Evolution Path to OOP if needed (see enhancements)**

#### 8.2. Core RAG Framework: LlamaIndex

**Components Used:**

* `llama_index.core` (Documents, Nodes, VectorStoreIndex, QueryEngine)
* `llama_index.core.node_parser.CodeSplitter`
* `llama_index.embeddings.huggingface.HuggingFaceEmbedding`
* `llama_index.llms.google_genai.GoogleGenAI`
* `llama_index.vector_stores.chroma.ChromaVectorStore`

**Justifications:**

* Rapid development
* RAG-centric components
* Modular design
* CodeSplitter for syntactic chunking
* Provenance tracking via `source_nodes`

#### 8.3. Embedding Model: nomic-embed-text-v1.5 (default config uses for efficiency & baseline - all-MiniLM-L6-v2 )

**Interface:** `llama_index.embeddings.huggingface.HuggingFaceEmbedding`

**Justifications:**

* Large context window (8192 tokens)
* Optimized for retrieval
* Local execution and open-source
* add `trust_remote_code=True` for model compatibility

#### 8.4. Language Model (LLM): Google Gemini Flash

**Interface:** `llama_index.llms.google_genai.GoogleGenAI`

**Justifications:**

* Powerful generative capabilities
* Accessible API
* Effective for summarization and Q\&A
* Free tier for prototyping

#### 8.5. Vector Store: ChromaDB

**Interfaces:**

* `llama_index.vector_stores.chroma.ChromaVectorStore`
* `chromadb.PersistentClient`

**Justifications:**

* Simple local setup
* Persistence across runs
* Seamless LlamaIndex integration

#### 8.6. Configuration & Environment Management

**Libraries:** `python-dotenv`
**Files:** `config.py`, `constants.py`, `.env`

**Justifications:**

* Secure API key management
* Centralized configuration
* Readability and maintainability

#### 8.7. Python Standard Libraries

**Libraries:** `pathlib`, `logging`, `json`, `hashlib`, `os`, `subprocess`, `time`, `random`

**Justifications:**

* File handling (`pathlib`)
* Logging and diagnostics
* JSONL cache format
* Chunk hashing (`hashlib`)
* Environment and subprocess control
* Utilities (`time`, `random`)

## Enhancements for RAG-Based Code Q\&A System

### 1. Embedding Strategy

* **Embedding quality** is critical for RAG performance. Recommendation:

  * Use **code-specialized embedding models** such as:

    * `Qodo-Embed-1`
    * `Codestral Embed` (Mistral AI)
    * `Nomic Embed Code`
  * Even open-source models like **E5** variants show better results on code tasks.

* **Current Limitation:**

  * Code and LLM-generated summaries are embedded with the same model.

* **Proposed Improvement:**

  * Use **two separate embedders** for:

    * Raw code
    * LLM-generated summaries
  * Store in **separate vector collections**.
  * Merge results using **metadata joins** (e.g., file/function names).

* **Parallelization (Out of Scope):**

  * A connection pool using **multiple API keys** and threads can boost indexing speed linearly.
  * Not suitable for production due to security concerns.

* **Future Evaluation:**

  * Test performance of newer models like `voyage-code-2`.

### 2. Vector Store Improvements

* Combine **vector search** with **structured metadata filters**:

  * E.g., Use `.py` extension to narrow queries to Python files.

* Add **keyword/BM25 full-text search** over metadata and summaries:

  * ChromaDB lacks hybrid search natively.
  * Workarounds exist ([Chroma Issue #1330](https://github.com/chroma-core/chroma/issues/1330)).
  * For production, consider migrating to **Weaviate**.

### 3. LLM Choice

* Evaluate alternatives based on latency, cost, and accuracy:

  * **OpenAI (GPT-4.5)**
  * **Anthropic (Claude)**
  * **DeepSeek**
  * **Mistral** APIs or local deployment

### 4. Error Handling

* Improve robustness in the following areas:

  * API failures and timeouts
  * Rate limit handling
  * File I/O and edge cases

### 5. Scalability

* Current pipeline processes a **single repo sequentially**.
* Add support for **`asyncio`** or parallel processing for:

  * Batch indexing
  * Large repositories

### 6. Web UI

* Prototype UI is intentionally minimal.
* Future improvements could include:

  * Multi-repo navigation
  * Search and filter interface
  * Highlighted code previews

---

**Note:**
This solution focuses on solving the core challenges using existing libraries and minimal overhead. While further enhancements are possible, they are beyond the current PRD scope.

The design meets all core requirements under prototyping constraints and provides a strong foundation for a scalable, production-ready RAG-based Code Q\&A system.

## Time Sheet

Todat time: 1 day / 8 hrs (focussed time)

1. Understanding requirements - 1 hour
2. Research problems and possible solutions / prototype - 3.5 hours
3. Test cases - 1 hour
4. Coding - 2.5 hours
5. AI validation code - 1 hour
