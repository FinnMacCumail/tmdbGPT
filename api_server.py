from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from app import build_app_graph
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tmdbgpt-api")

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
async def run_query(req: QueryRequest, request: Request):
    # Log raw request body for debugging
    try:
        raw_body = await request.body()
        logger.debug(f"🧾 Raw body: {raw_body.decode('utf-8')}")
    except Exception as e:
        logger.warning(f"⚠️ Could not log raw body: {e}")

    logger.info(f"🧠 Parsed query: {req.query!r}")

    try:
        result = graph.invoke({"input": req.query})

        logger.info("✅ Graph invoke completed.")

        return {
            "entries": result.get("formatted_response", []),
            "question_type": result.get("question_type", "list"),
            "response_format": result.get("response_format", "ranked_list"),
            "execution_trace": result.get("execution_trace", []),
            "explanation": result.get("explanation", "")
        }
    except Exception as e:
        logger.error(f"❌ Error during graph.invoke: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/")
def root():
    return {"message": "TMDBGPT API is live. Use POST /query with {query: '...'}"}
