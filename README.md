# RAG-Based Document Q&A Application

## Overview

This application allows users to upload multiple texts files and pdf files and md files and ask questions about their content using a Retrieval-Augmented Generation (RAG) pipeline. The system breaks documents into chunks, stores them in a vector database, and uses semantic search to find relevant information before generating answers by finding the top 20 most relevant chunks.

## Features

The user uploads a .txt, .md, or .pdf file or a combo or all 3 and the app breaks the docs into small pieces, finds the most relevant parts, and answers using only those in a chat that remembers your conversation.
It also caches files so they aren’t reprocessed each time.

## How to Run

### Prerequisites
- API key

### Steps to Run

1. **Open your Codespace** from your forked repository on the `assignment1` branch

2. **Open the terminal** in Codespace

3. **Set your API key**:
```bash
   export API_KEY="your_api_key_here"
```

4. **Run the Streamlit application**:
```bash
   streamlit run chat_with_pdf.py
```

5. **Open the application**: Click "Open in Browser" when the popup appears in the bottom-right corner

6. **Use the application**:
   - Upload one or more documents (.txt, .md, or .pdf)
   - Wait for processing to complete
   - Ask questions about your documents in the chat interface

## Implementation Details

### Code Implementation

The original starter code (`chat_with_pdf.py`) was a simple file Q&A application that 
- Only supported single .txt file uploads
- Read the entire file content and passed an directly to the LLM 
- No document chunking or vector search (NO RAG IMPLEMENTATION)

The new and improved enhanced version implements a full RAG pipeline with the following improvement that uses the chunking method that is shown in the notebook:

**PDF Support**: Added support for PDF files using LangChain's `PyPDFLoader`. The original code only handled text files that could be decoded directly as UTF-8 strings.

**Multiple File Uploads**: Modified the file uploader to accept multiple files simultaneously with the `accept_multiple_files=True` parameter. All uploaded documents are processed together and can be queried as a single knowledge base.

**RAG Pipeline Implementation**: The core enhancement is the implementation of a complete RAG system with three main components:

**Document Chunking**: I split documents into smaller pieces using `RecursiveCharacterTextSplitter` with a chunk size of 200 characters and no overlap. This is the same approach from the class notebook. Breaking documents into chunks lets the system work with large files and find just the relevant sections for each question.

**Vector Storage**: Each chunk gets converted into an embedding using OpenAI's text-embedding-3-large model and stored in a ChromaDB vector database. This makes it possible to search for chunks that are semantically similar to the user's question.

**Retrieval and Generation**: When someone asks a question, the system finds the 20 most similar chunks (I used k=20 like in the notebook example). These chunks get formatted as context and sent to GPT-4o along with the question. The model then generates an answer based on what it finds in those chunks.

**Session State Management**: I added session state tracking for the vector database and the list of processed files. This gets used to save processing time because if we didn;t store the vectors in the database, o the app would have to recalculate all the embeddings and rebuild the entire vector database from scratch on every interaction, which would be really slow. By storing the vector database in session state, it only gets created once when files are first uploaded, then gets reused for all the questions.

**Structured Prompting**: Uses LangChain's `PromptTemplate` following the same structure from the notebook.. This way the model only answers based only on the retrieved context and to say when it doesn't know something, instead of hallucianting an response.

**File Handling**: Since LangChain's document loaders require file paths rather than byte streams, the application temporarily saves uploaded files using Python's `tempfile` module, loads them with the appropriate loader, then cleans up the temporary files after processing.

### Technical Stack

- **Streamlit**: Web interface and file upload handling
- **LangChain**: RAG pipeline framework (document loaders, text splitters, prompt templates)
- **ChromaDB**: Vector database for similarity search
- **OpenAI**: Language model (gpt-4o) and embeddings (text-embedding-3-large)

### Architecture

The application follows this flow:
1. User uploads documents through Streamlit interface
2. Documents are saved temporarily and loaded with appropriate LangChain loaders
3. Documents are split into 200-character chunks
4. Chunks are embedded and stored in ChromaDB vector store
5. When user asks a question, the system retrieves the 20 most similar chunks
6. Retrieved chunks are formatted as context and passed to the LLM with the question
7. LLM generates an answer based only on the retrieved context

## Configuration

No changes were made to `requirements.txt` or `.devcontainer` configurations. All necessary dependencies were already included in the provided Codespace template.

## File Structure
```
chat_with_pdf.py    # Main application file
README.md           # This file
ref-log.md          # Reference log documenting sources and AI usage
```