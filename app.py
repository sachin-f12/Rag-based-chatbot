import os
import sys
import shutil
import uuid
import streamlit as st
from unittest.mock import MagicMock

# 🛠️ Fix the 'pwd' module issue on Windows
if sys.platform == "win32":
    sys.modules["pwd"] = MagicMock()
    sys.modules["grp"] = MagicMock()

# ✅ Correct import for Pydantic v2
from pydantic import BaseModel  

# Continue with LangChain imports after fixing pwd issue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone as VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "rag-chat"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=768, metric="cosine")

index = pc.Index(index_name)

# Initialize Embeddings & LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 PDF Chatbot with Pinecone & Gemini")

# Sidebar for file upload & deletion
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and split document
            loader = PyMuPDFLoader(file_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(data)

            # Generate unique IDs
            ids = [f"{uploaded_file.name}_{uuid.uuid4()}" for _ in text_chunks]

            # Store in Pinecone
            VectorStore.from_documents(text_chunks, embeddings, index_name=index_name, ids=ids)

            st.success("File uploaded and processed successfully! 🎉")
            st.session_state["file"] = uploaded_file.name
            st.session_state["ids"] = ids

    # Delete file button
    if "file" in st.session_state and st.button("🗑️ Delete File"):
        try:
            os.remove(f"temp_{st.session_state['file']}")  # Delete local file
            index.delete(ids=st.session_state["ids"])  # Remove from Pinecone
            st.success("File deleted successfully!")
            del st.session_state["file"]
            del st.session_state["ids"]
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# Chat Interface
st.subheader("💬 Chat with your document")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me anything about the uploaded document!"}]

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query
    retriever = VectorStore.from_existing_index(index_name, embeddings).as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    response = qa_chain({"query": user_input})["result"]

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save conversation history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": response})



