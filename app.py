import sys
from unittest.mock import MagicMock

# Mock 'pwd' and 'grp' modules to avoid import errors on Windows
sys.modules["pwd"] = MagicMock()
sys.modules["grp"] = MagicMock()

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter




# Access API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Import required modules
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient  # Correct import
from langchain.vectorstores import Pinecone as VectorStore  # Avoid name conflict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "rag-chat"

# Ensure index exists before using
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=768, metric="cosine")  # Adjust dimension as needed

index = pc.Index(index_name)

# Initialize Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the chat model (Fixed model name)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", convert_system_message_to_human=True)

# Streamlit UI
st.title("PDF Chatbot with Pinecone and Gemini")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_path = f"temp_{uploaded_file.name}.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Load the PDF document
    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    # Generate unique IDs for deletion later
    ids = [f"{uploaded_file.name}_{i}" for i in range(len(text_chunks))]

    # Embed the chunks and store in Pinecone
    vector_store = VectorStore.from_documents(
        text_chunks, embeddings, index_name=index_name, ids=ids
    )

    st.success("File uploaded and processed successfully!")

    # Delete file button
    if st.button("Delete File"):
        try:
            os.remove(file_path)  # Delete the file
            index.delete(ids=ids)  # Delete only relevant embeddings
            st.success("File and its data deleted successfully!")
        except Exception as e:
            st.error(f"Error deleting file: {e}")

# Chat interface
user_input = st.text_input("You: ")
if user_input:
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    response = qa_chain({"query": user_input})
    st.text_area("Bot:", value=response['result'], height=200)
