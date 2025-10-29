import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import tempfile

#use llm setup code from notebook setup
llm = ChatOpenAI(
    model="openai.gpt-4o",  # Fixed: was gpt-5
    temperature=0.2,
)

st.title("📝 File Q&A with OpenAI")

# File uploader
# added pdf support
uploaded_files = st.file_uploader("Upload articles", type=("txt", "md", "pdf"), accept_multiple_files=True)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the articles"}]

#use vector store 
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = []

# Process uploaded files immediately
if uploaded_files:
    # Check if new files were uploaded
    current_files = []
    for file in uploaded_files:
        current_files.append(file.name)
    if current_files != st.session_state["processed_files"]:
        with st.spinner("Processing documents..."):
            all_chunks = []

            # Loop through each uploaded file
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                file_name = uploaded_file.name
                if '.' in file_name:
                    file_extension = file_name.split('.')[-1]
                else:
                    file_extension = 'txt'  # default

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()
                tmp_file_path = temp_file.name
                
                #Choose loader based on file type
                if file_extension == "pdf":
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path)
                
                documents = loader.load()
                
                # Chunk the document (from notebook)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=0
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                
                #DELETE the file
                os.remove(tmp_file_path)
            
            # Create vector store from ALL chunks
            st.session_state["vectorstore"] = Chroma.from_documents(
                documents=all_chunks,
                embedding=OpenAIEmbeddings(model="openai.text-embedding-3-large")
            )
            st.session_state["processed_files"] = current_files
            
            st.success(f"✅ Processed {len(uploaded_files)} document(s)! Created {len(all_chunks)} chunks total.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input(
    "Ask something about the articles",
    disabled=not uploaded_files,
)

if question and st.session_state["vectorstore"]:
    # Setup retrieval (from notebook)
    retriever = st.session_state["vectorstore"].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    # Format docs function (from notebook)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Get the relevant docs use ipynb example
    relevant_docs = retriever.invoke(question)
    context = format_docs(relevant_docs)
    
    # Prompt template (from notebook)
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    
    Context: {context} 
    
    Answer:
"""
    prompt = PromptTemplate.from_template(template)
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    # Get response
    with st.chat_message("assistant"):
        messages = prompt.invoke({"question": question, "context": context})
        response = llm.invoke(messages)
        st.write(response.content)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
