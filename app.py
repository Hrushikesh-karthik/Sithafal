import fitz  # PyMuPDF
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import re

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract text from page
    return text

# Function to split the extracted text into chunks (for better granularity)
def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    chunk = []
    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) > chunk_size:
            chunks.append(' '.join(chunk))
            chunk = []
    if chunk:
        chunks.append(' '.join(chunk))  # Add the last chunk
    return chunks

# Function to generate embeddings using a pre-trained model (BERT-based for text embeddings)
def generate_embeddings(text_chunks, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())  # Mean of last layer hidden states
    return np.array(embeddings)

# Function to store embeddings in a FAISS vector database
def store_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    index.add(embeddings)
    return index

# Function to retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, index, text_chunks, model_name="bert-base-uncased", k=5):
    query_embedding = generate_embeddings([query], model_name=model_name)
    D, I = index.search(query_embedding, k)  # Search for top-k relevant chunks
    return [text_chunks[i] for i in I[0]]

# Function to generate a response from the retrieved chunks using a basic LLM (for example, GPT-like model)
def generate_response(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Using the following information, answer the query: {query}\n\nContext:\n{context}\nAnswer:"
    # Here, we are using a simplified generation approach
    # In practice, you would use GPT, BERT, or any other LLM with the appropriate model API.
    response = f"Generated response based on the context: {context[:300]}..."  # Limit output for simplicity
    return response

# Main workflow
def main():
    # Step 1: Extract text from PDF
    pdf_path = "data.pdf"  # Replace with your PDF path
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split text into chunks
    text_chunks = chunk_text(pdf_text)
    
    # Step 3: Generate embeddings and store them in FAISS
    embeddings = generate_embeddings(text_chunks)
    index = store_embeddings(embeddings)
    
    # Step 4: Query handling (user input)
    query = "Unemployment based on type of degree"  # Example query
    relevant_chunks = retrieve_relevant_chunks(query, index, text_chunks)
    
    # Step 5: Generate response
    response = generate_response(query, relevant_chunks)
    print("Response:", response)

# Example to process and compare tables or data
def extract_unemployment_info(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(1)  # Page 2 in the document (0-indexed)
    text = page.get_text("text")
    # Use regex to extract unemployment data based on degree type (customize as per the document structure)
    pattern = r"(\w+)\s+(\d+\.\d+)"  # Example pattern for degree and unemployment rates (you need to adjust this)
    matches = re.findall(pattern, text)
    return matches

def extract_tabular_data(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(5)  # Page 6 in the document (0-indexed)
    text = page.get_text("text")
    # Process tabular data (you can use regex or text processing)
    return text

if __name__ == "__main__":
    # Run the main pipeline
    main()

    # Extract specific data from pages
    pdf_path = "data.pdf"  # Replace with your PDF path
    unemployment_info = extract_unemployment_info(pdf_path)
    print("Unemployment Information:", unemployment_info)

    tabular_data = extract_tabular_data(pdf_path)
    print("Tabular Data:", tabular_data)
