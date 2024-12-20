This Python code performs several tasks related to extracting, processing, and querying information from PDF files using different libraries such as PyMuPDF (for PDF extraction), FAISS (for similarity search), and Hugging Face transformers (for generating text embeddings). Here's a breakdown of what each section of the code does:
To setup the project:
commands:
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
To run: python app.py

**You might face this error**:
OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/

**Run this command to clear that**: $env:KMP_DUPLICATE_LIB_OK = "TRUE"
**Explanation**
### 1. Imports
- **fitz (PyMuPDF)**: Used to extract text from PDF files.
- **numpy**: Used for numerical operations, specifically to handle arrays and embeddings.
- **faiss**: A library for similarity search, storing and querying vectorized embeddings.
- **transformers**: Used to load pre-trained models (like BERT) for text embedding generation.
- **torch**: For handling tensors and operations related to neural networks (used with transformers).
- **re**: For regular expression operations to extract specific patterns from text.

### 2. Functions Overview
The script defines multiple functions, each handling a specific part of the workflow.

#### 2.1. `extract_text_from_pdf(pdf_path)`
- **Purpose**: Extracts text from a PDF file at the specified `pdf_path`.
- **How**: It opens the PDF, iterates through each page, and extracts text using `get_text("text")`.

#### 2.2. `chunk_text(text, chunk_size=500)`
- **Purpose**: Splits the extracted text into smaller chunks (default size of 500 characters) to improve the granularity of processing and make the text more manageable.
- **How**: The text is split into words, and words are grouped into chunks. Each chunk is added to the list when it exceeds the specified chunk size.

#### 2.3. `generate_embeddings(text_chunks, model_name="bert-base-uncased")`
- **Purpose**: Converts text chunks into numerical embeddings (vectors) using a pre-trained BERT model.
- **How**: It uses the Hugging Face transformers library to tokenize each chunk, processes it through a BERT-based model, and averages the embeddings from the last hidden layer to create a representation for each chunk.

#### 2.4. `store_embeddings(embeddings)`
- **Purpose**: Stores the embeddings in a FAISS index for efficient similarity search.
- **How**: A FAISS index is created using the L2 distance metric, and the embeddings are added to the index.

#### 2.5. `retrieve_relevant_chunks(query, index, text_chunks, model_name="bert-base-uncased", k=5)`
- **Purpose**: Given a query, retrieves the top `k` most relevant text chunks from the PDF based on similarity to the query.
- **How**: The query is converted into an embedding, and FAISS is used to search for the closest matches in the stored embeddings. The relevant chunks are then returned.

#### 2.6. `generate_response(query, relevant_chunks)`
- **Purpose**: Generates a response based on the context provided by the relevant chunks.
- **How**: It combines the retrieved chunks into a context string and prepares a prompt for generating a response. In practice, this function would integrate with a large language model (like GPT) to produce a meaningful answer, but here it's simplified for demonstration.

### 3. Main Workflow (`main`)
- **Purpose**: Orchestrates the entire process: extracting text from a PDF, splitting it into chunks, generating embeddings, storing them in FAISS, querying for relevant chunks, and generating a response.
- **Steps**:
  1. **Extract text from PDF**: The function `extract_text_from_pdf` is called to extract all the text from the PDF.
  2. **Split text into chunks**: The text is split into smaller chunks using `chunk_text`.
  3. **Generate embeddings**: The `generate_embeddings` function is used to create embeddings for each chunk.
  4. **Store embeddings in FAISS**: The embeddings are stored in a FAISS index.
  5. **Handle query**: A sample query is used to retrieve relevant chunks using `retrieve_relevant_chunks`.
  6. **Generate a response**: The `generate_response` function produces a response based on the retrieved chunks.

### 4. Extract Specific Data Functions
- **`extract_unemployment_info(pdf_path)`**: Extracts unemployment data (degree type and corresponding rates) from a specific page in the PDF using a regular expression. This function is customized to look for specific patterns like a degree name followed by a number (e.g., unemployment rate).
- **`extract_tabular_data(pdf_path)`**: Extracts tabular data from a specific page in the PDF. This can be processed further to handle tables, though in this code, it simply returns raw text.

### 5. Execution Flow
- The script executes the **main pipeline** by calling the `main` function, which extracts the text, processes it, and answers a query.
- It also extracts specific information related to unemployment data and tabular data, using the functions `extract_unemployment_info` and `extract_tabular_data`.

### 6. Expected Outputs
- The output of the **main function** is a response generated based on the context of the relevant chunks retrieved from the PDF.
- The **extracted unemployment information** is printed, showing the degree type and corresponding unemployment rates.
- The **tabular data** is extracted and printed.

### Summary
This code processes PDF files by extracting text, breaking it into smaller chunks, generating embeddings for those chunks using BERT, storing them in a FAISS index for fast similarity search, and then allowing the user to query the document to retrieve relevant information. Additionally, it includes specific functions for extracting tabular and pattern-based data from the document. This approach enables advanced document search and query capabilities, often used in information retrieval and natural language processing (NLP) applications.
