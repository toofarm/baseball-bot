import os
import google.generativeai as genai
from data_preprocessing import preprocess_data, GeminiEmbeddingFunction as embed_fn

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

db = preprocess_data()

# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "How can I find out about which teams led pitching and batting metrics in a particular year?"

result = db.query(query_texts=[query], n_results=4)

passage = result["documents"][0]

flat_passage = ' '.join(passage)

# print(passage)
print(flat_passage)

