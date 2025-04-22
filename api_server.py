# api_server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app import build_app_graph

app = FastAPI()
graph = build_app_graph()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        result = graph.invoke({"input": req.query})
        return {
            "status": "ok",
            "query": req.query,
            "response": result.get("responses"),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
def root():
    return {"message": "TMDBGPT API is live. Use POST /query with {query: '...'}"}
