import streamlit as st
import docx
from docx import Document
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import pandas as pd
from langchain_groq import ChatGroq
import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set Groq API Key and Model
api_key = "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"

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to read a DOCX file
def read_docx(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text.strip())
    return "\n".join(text)

# Chunking function
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Generate embeddings
def generate_embeddings(chunks):
    if not chunks:
        print("No chunks to process.")
        return []
    return embedding_model.embed_documents(chunks)

# Create FAISS vector database
def create_faiss_db(chunks, embeddings):
    if not chunks or not embeddings:
        print("Cannot create FAISS DB without valid chunks and embeddings.")
        return None
    return FAISS.from_texts(chunks, embedding_model)

# Compare two documents
def compare_documents(doc1_path, doc2_path, parameters):
    # Read documents
    doc1_text = read_docx(doc1_path)
    doc2_text = read_docx(doc2_path)

    # Chunk documents
    chunks1 = chunk_text(doc1_text)
    chunks2 = chunk_text(doc2_text)

    # Generate embeddings
    embeddings1 = generate_embeddings(chunks1)
    embeddings2 = generate_embeddings(chunks2)

    # Create FAISS vector databases
    db1 = create_faiss_db(chunks1, embeddings1)
    db2 = create_faiss_db(chunks2, embeddings2)

    # Initialize LLM (ChatGroq)
    llm = ChatGroq(groq_api_key="api_key", model_name="llama3-8b-8192")

    # Retrieval QA Chains
    retriever1 = db1.as_retriever()
    retriever2 = db2.as_retriever()

    qa_chain1 = RetrievalQA.from_chain_type(llm, retriever=retriever1)
    qa_chain2 = RetrievalQA.from_chain_type(llm, retriever=retriever2)

    # Compare documents based on parameters
    results = []
    for param in parameters:
        response1 = qa_chain1.run(param)
        response2 = qa_chain2.run(param)
        similarity = "Similar" if response1.strip() == response2.strip() else "Different"
        results.append({"Parameter": param, "Doc1": response1, "Doc2": response2, "Similarity": similarity})

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

# Streamlit UI
st.title("Intelligent Document Comparer")

uploaded_file1 = st.file_uploader("Upload Document 1 (DOCX)", type=["docx"])
uploaded_file2 = st.file_uploader("Upload Document 2 (DOCX)", type=["docx"])

if uploaded_file1 and uploaded_file2:
    # Save the uploaded files temporarily
    with open("temp_doc1.docx", "wb") as f:
        f.write(uploaded_file1.getbuffer())
    with open("temp_doc2.docx", "wb") as f:
        f.write(uploaded_file2.getbuffer())
    
    doc1_path = "temp_doc1.docx"
    doc2_path = "temp_doc2.docx"

    # Allow user to input comparison parameters
    st.subheader("Enter Parameters for Comparison")
    default_parameters = ["Purpose of the document", "Key terms and definitions", "Primary clauses"]
    parameter_input = st.text_area(
        "Enter parameters separated by commas:",
        value=", ".join(default_parameters)
    )
    parameters = [param.strip() for param in parameter_input.split(",") if param.strip()]

    if st.button("Compare Documents"):
        # Perform document comparison
        with st.spinner("Comparing documents..."):
            results_df = compare_documents(doc1_path, doc2_path, parameters)
        
        # Display the results
        st.success("Comparison completed!")
        st.dataframe(results_df)

        # Download results as Excel
        output_file = "comparison_results.xlsx"
        results_df.to_excel(output_file, index=False)

        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Comparison Results",
                data=f,
                file_name="comparison_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Please upload both documents to start the comparison.")
