"""FastAPI app — analyst query endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from financial_advisory_agent.agents import create_analyst_agent  # noqa: E402

_analyst = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _analyst
    _analyst = create_analyst_agent()
    yield


app = FastAPI(title="Financial Advisory API", lifespan=lifespan)


class AnalystRequest(BaseModel):
    query: str


class AnalystResponse(BaseModel):
    response: str


@app.post("/analyst", response_model=AnalystResponse)
def query_analyst(body: AnalystRequest):
    if _analyst is None:
        raise HTTPException(status_code=503, detail="Analyst not ready")
    result = _analyst.invoke({"messages": [{"role": "user", "content": body.query}]})
    return AnalystResponse(response=result["messages"][-1].content)
