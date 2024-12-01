from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# App functions
from src.data_preprocessing import preprocess_data
from src.model_interactions import query_model

app = FastAPI()

class Query(BaseModel):
    message: str
    timestamp: int

@app.get("/")
async def root():
    return {"message": "To use this API, send a query to /query with a message describing the information you're trying to retrieve."}

@app.get("/query")
async def query(query: Query):
    # Populate and return db
    db = preprocess_data()

    # Query the model
    response = query_model(db, query.message, query.timestamp)

    jsoned = jsonable_encoder(response)

    return jsoned