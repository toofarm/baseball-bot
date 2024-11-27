import requests
import chromadb
from google.api_core import retry
import google.generativeai as genai

# GitHub repository details
REPO_OWNER = "jldbc"
REPO_NAME = "pybaseball"
DOCS_PATH = "docs"
BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{DOCS_PATH}"

# Step 1: Fetch all markdown files
def fetch_markdown_files():
    response = requests.get(BASE_URL)
    response.raise_for_status()
    files = response.json()
    return [file for file in files if file["name"].endswith(".md")]

# Function to download file content
def download_file_content(file_url):
    response = requests.get(file_url)
    response.raise_for_status()
    return response.text

# Step 2: Download and save each file
def download_all_files(files):
    md_files = []

    for file in files:
        file_name = file["name"]
        file_content_url = file["download_url"]

        # Fetch content
        print(f"Downloading {file_name}...")
        file_content = download_file_content(file_content_url)

        # Insert into list
        md_files.append(file_content)
        
    print("All files have been processed and stored.")
    return md_files

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]

# Main function to preprocess data
def preprocess_data():
    # Create the embeddings
    DB_NAME = "pybaseballdb"
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    chroma_client = chromadb.Client()
    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

    if db.count() == 0:
        # Fetch markdown files
        files = fetch_markdown_files()
        md_files = download_all_files(files)
        db.add(documents=md_files, ids=[str(i) for i in range(len(md_files))])

    return db

if __name__ == "__main__":
    preprocess_data()
        