from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from loguru import logger

# App functions
from src.data_preprocessing import preprocess_data
from src.model_interactions import query_model

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Logger
logger.add("logs.log", rotation="5 MB")

app = FastAPI()

class Query(BaseModel):
    message: str
    timestamp: int

@app.get("/")
async def root():
    return {"message": "To use this API, send a query to /query with a message describing the information you're trying to retrieve."}

@app.get("/query")
async def query(query: Query):
    # Check input payload
    if not query.message:
        logger.info("Received query without message.")
        return {"error": "Please provide a message in the query payload."}
    elif not query.timestamp:
        logger.info("Received query without timestamp.")
        return {"error": "Please provide a timestamp in the query payload."}

    logger.info(f"Received query: {query.message}")

    # Populate and return db
    db = preprocess_data()

    # Query the model
    response = query_model(db, query.message, query.timestamp)

    jsoned = jsonable_encoder(response)

    logger.info(f"Response metrics: {jsoned.get('performance_metrics')}")

    return jsoned