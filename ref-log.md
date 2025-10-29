# Reference Log

## External Sources and Documentation

### LangChain Documentation
- **URL**: https://python.langchain.com/docs/
- **Used for**: Understanding how to implement the RAG pipeline, particularly document loaders (TextLoader and PyPDFLoader), text splitters (RecursiveCharacterTextSplitter), and vector store integration with ChromaDB. Also referenced for prompt template formatting.


### ChromaDB Documentation
- **URL**: https://docs.trychroma.com/
- **Used for**: Understanding how to create and query the vector store, specifically the `from_documents` method and similarity search configuration.

## Class Materials

### langgraph_chroma_retreiver.ipynb (Class Notebook)
- **Source**: Provided by instructor in class
- **Used for**: 
  - RAG pipeline structure and workflow
  - Chunking strategy (chunk_size=200, chunk_overlap=0)
  - Retrieval configuration (k=20 for similarity search)
  - Prompt template format and structure
  - Example of how to use `RecursiveCharacterTextSplitter`, `Chroma.from_documents`, and `PromptTemplate`

### Original chat_with_pdf.py (Starter Code)
- **Source**: Provided in assignment template
- **Used for**: Base Streamlit application structure, session state pattern for chat messages, and chat interface design

## GenAI Usage

### ChatGPT (OpenAI)


I used ChatGPT to help with technical challenges when adapting the class notebook to a Streamlit application.

**PDF File Support**: I asked how to load PDF files using LangChain. ChatGPT suggested using `PyPDFLoader`.

**File Handling**: I needed help converting Streamlit's uploaded file objects into file paths for LangChain's loaders. ChatGPT suggested using `tempfile.NamedTemporaryFile()` to temporarily save files to disk.

The core RAG implementation came from the class notebook. ChatGPT helped with Streamlit-specific integration.
I also used chatgpt to help me with markdown syntax.