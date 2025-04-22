from fastapi import FastAPI
from pydantic import BaseModel
from app import build_app_graph
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = build_app_graph()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        result = graph.invoke({"input": req.query})
        trace = {
            "steps": [step.get("endpoint") or step.get("step_id") for step in result.get("plan_steps", [])],
            "question_type": result.get("question_type"),
            "resolved_entities": result.get("resolved_entities")
        }
        return {
            "status": "ok",
            "query": req.query,
            "response": result.get("formatted_response"),
            "trace": trace
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
def root():
    return {"message": "TMDBGPT API is live. Use POST /query with {query: '...'}"}
