**Product Requirements Document (PRD)**

**Title**: RAG-based Code QA System for Open Source Projects\
**Duration**: 2 Days\
**Goal**: Build a functional prototype of a system that can answer developer questions about any open-source GitHub code repository using a RAG (Retrieval-Augmented Generation) approach.

**Problem Statement**

Developers often need to understand unfamiliar codebases quickly. A system that can ingest a codebase and answer questions about it would greatly help with onboarding, debugging, and comprehension tasks.

**Objective**

Build a simple AI-powered Q&A system over the contents of a GitHub repository. This system should:

- Ingest code from an open-source GitHub project.
- Process the code and metadata into a searchable format.
- Use RAG to answer natural language questions based on the ingested content.
- Expose a basic web-based chat interface for interaction.
- Use any programming language of your own choice.

**Functional Requirements**

1.  **Code Ingestion**

  - Clone/download a given public GitHub repository.
  - Parse and preprocess code files (.py, .js, .java, .ts, etc.).
  - Chunk and index code files using a vector store of your choice.
  - Metadata storage and retrieval as necessary.

2.  **Retrieval-Augmented Generation (RAG)**

  - Retrieve relevant code chunks based on a question.
  - Use an LLM (local or OpenAI API) to generate an answer using the context.

3.  **Sample Questions to Support**

  - What does <function_name> do?
  - Explain how a <feature> works?
  - What is the return type of <function>?
  - Review a certain <function>.
  - Explain the role of <function_or_module>.

4.  **Web UI**

  - A minimal chat window interface.
  - User inputs a question and sees the answer with relevant file references.

5.  **Bonus**

  - Clean code structure.
  - Modular design.
  - Unit tests for major components.
  - Comments and typing annotations.

**Deliverables**

- Source code in a GitHub repository.
- README with setup instructions and examples.
- A short **design document** that includes:

  - Overall architecture.
  - Design decisions and trade-offs.
  - Pros and cons of the chosen approach.
  - Technologies and libraries used.

**Evaluation Criteria**

- Correctness and completeness of RAG implementation.
- Relevance and accuracy of answers.
- Code readability and modularity.
- UI simplicity and usability.
- Depth of design, document and justification of decisions.
- Bonus: tests and documentation quality.

**Timeline**

- Duration: 2 days from receiving the challenge.
- Expected time commitment: ~6-12 focused hours.

PS

- Host your code in your own git repo and share a link.
- Avoid plagiarism.
