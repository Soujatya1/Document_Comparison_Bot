import streamlit as st
import docx
from docx import Document
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import pandas as pd
from langchain_groq import ChatGroq

# Set Groq API Key and Model
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Helper Functions
def read_docx(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text.strip())
    return "\n".join(text)

def chunk_text(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings)

def store_embeddings_in_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index, query, top_k=5, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return indices.flatten()

def analyze_similarity(index1, index2, chunks1, chunks2, parameters):
    results = []
    for param in parameters:
        similar_chunks1 = [chunks1[idx] for idx in search_faiss(index1, param)]
        similar_chunks2 = [chunks2[idx] for idx in search_faiss(index2, param)]
        comparison_prompt = f"Compare these texts based on {param}:\n\nText1: {similar_chunks1}\n\nText2: {similar_chunks2}"
        response = llm(comparison_prompt)
        results.append((param, response["generation"].strip()))
    return results

# Streamlit App
st.title("Intelligent Document Comparer")
st.sidebar.header("Upload Documents")

# File Upload Section
doc1_file = st.sidebar.file_uploader("Upload Document 1 (docx)", type=["docx"])
doc2_file = st.sidebar.file_uploader("Upload Document 2 (docx)", type=["docx"])

# Parameter Input
parameters = st.sidebar.text_area(
    "Enter Parameters for Comparison (one per line)", 
    placeholder="E.g., Summary\nKey Points\nDifferences in Terminology"
).splitlines()

# Process Button
if st.sidebar.button("Compare Documents"):
    if doc1_file and doc2_file and parameters:
        with st.spinner("Processing..."):
            # Read and Process Documents
            text1 = read_docx(doc1_file)
            text2 = read_docx(doc2_file)

            # Chunking
            chunks1 = list(chunk_text(text1))
            chunks2 = list(chunk_text(text2))

            # Embedding Generation
            embeddings1 = generate_embeddings(chunks1)
            embeddings2 = generate_embeddings(chunks2)

            # FAISS Index Creation
            index1 = store_embeddings_in_faiss(embeddings1)
            index2 = store_embeddings_in_faiss(embeddings2)

            # Analyze Similarities
            results = analyze_similarity(index1, index2, chunks1, chunks2, parameters)

            # Display Results
            st.success("Comparison Complete!")
            st.write("### Comparison Results")
            results_df = pd.DataFrame(results, columns=["Parameter", "Comparison"])
            st.table(results_df)

            # Option to Download Results
            output_path = "comparison_results.xlsx"
            results_df.to_excel(output_path, index=False)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Results as Excel",
                    data=file,
                    file_name="comparison_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.error("Please upload both documents and specify parameters!")

# About Section
st.sidebar.write("---")
st.sidebar.write("### About")
st.sidebar.info(
    "This app compares two documents based on specified parameters using embeddings, FAISS, and Groq LLM."
)
