"""TriageAI — FastAPI serving layer."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import TriageAgent

app = FastAPI(title="TriageAI", version="1.0.0")
agent = TriageAgent()

BASE = Path(__file__).parent.parent


class Ticket(BaseModel):
    text: str


class Batch(BaseModel):
    tickets: list[str]


@app.get("/")
def index():
    return FileResponse(BASE / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "triageai"}


@app.post("/triage")
def triage(ticket: Ticket):
    return agent.triage(ticket.text).to_dict()


@app.post("/triage/batch")
def triage_batch(batch: Batch):
    return {
        "results": [
            agent.triage(t, ticket_ref=f"TRIAGE-{i+1:03d}").to_dict()
            for i, t in enumerate(batch.tickets)
        ]
    }


@app.get("/info")
def info():
    return {
        "pipeline": ["perceive (category classifier)",
                     "assess (urgency scorer)",
                     "sense (sentiment)",
                     "decide (routing policy)",
                     "act (reply draft)"],
        "models": ["TF-IDF + LogisticRegression (category)",
                   "TF-IDF + LogisticRegression (urgency)"],
        "stack": ["Python", "FastAPI", "scikit-learn", "Docker"],
    }
