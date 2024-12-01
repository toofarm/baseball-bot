import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timezone
from src.data_preprocessing import GeminiEmbeddingFunction as embed_fn

load_dotenv()

def query_model(db, query: str, timestamp: int):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if not GOOGLE_API_KEY:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=GOOGLE_API_KEY)

    # Switch to query mode when generating embeddings.
    embed_fn.document_mode = False

    # Search the Chroma DB using the specified query.
    result = db.query(query_texts=[query], n_results=4)

    # Form the prompt
    passage = result["documents"][0]

    flat_passage = ' '.join(passage)

    passage_oneline = flat_passage.replace('\n', ' ')
    query_oneline = query.replace('\n', ' ')

    prompt = f"""You are a helpful and informative bot that assists in researching Major League Baseball statistics by explaining how to retrieve 
    statistical data using the Pybaseball Python library.

    When provided with a prompt, please use the included documents to explain how best to use Pybaseball in satisfying said prompt. If possible, provide an example of how to use Pybaseball's avaiable methods to achieve the requested result. Sometimes, Pybaseball may not have the exact method needed to satisfy the prompt. In this case, please provide an alternative method or suggest a different library that may be more suitable.

    Please strike a friendly and informative tone. If the passage is irrelevant to the answer, you may ignore it.

    PROMPT: {query_oneline}
    ANSWER: {passage_oneline}
    """

    # Generate content
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    answer = model.generate_content(prompt)

    # Output timestamp
    output_timestamp = int(datetime.now(timezone.utc).timestamp())
    input_timestamp = timestamp

    return {
        "response": answer.text,
        "performance_metrics": {
            "prompt_token_count": answer.usage_metadata.prompt_token_count,
            "candidates_token_count": answer.usage_metadata.candidates_token_count,
            "total_token_count": answer.usage_metadata.total_token_count,
            "input_timestamp": input_timestamp,
            "output_timestamp": output_timestamp,
            "timestamp_delta": output_timestamp - input_timestamp
        }
    }

