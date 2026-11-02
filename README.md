# TriageAI — Agentic Support-Ticket Triage (NIVUNA demo project)

A brand-new, working full-stack AI demo: an agent pipeline that reads a customer
support ticket and decides what to do with it — classify, score urgency, sense
sentiment, route to a team, draft a first response.

## Architecture

```
ticket text
    │ PERCEIVE  → category classifier (TF-IDF + LogisticRegression)
    │ ASSESS    → urgency scorer (ML model + critical-keyword rule booster)
    │ SENSE     → sentiment (lexicon)
    │ DECIDE    → routing policy (category → team; critical → on-call)
    ▼ ACT       → first-response draft (template; LLM seam marked in code)
triage decision (JSON) → FastAPI → web UI
```

## Run it

```bash
cd triageai
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python data/generate_data.py   # build the 800-ticket dataset
./venv/bin/python train.py                # train + save both models
./venv/bin/python -m uvicorn app.main:app --port 8765
# open http://localhost:8765
```

Or with Docker:

```bash
docker build -t triageai . && docker run -p 8765:8765 triageai
```

## API

- `POST /triage` — `{"text": "..."}` → full agent decision
- `POST /triage/batch` — `{"tickets": [...]}` → batch decisions
- `GET /info` — pipeline + model info
- `GET /` — demo UI

## Demo script (Wednesday)

1. **Hook (5 min):** "Support teams drown in tickets. I'll show you an agent that reads
   each ticket and decides — category, urgency, sentiment, which team, and a draft
   reply — in under a second."
2. **Live demo (15 min):** paste the 5 prepared tickets (below) one by one. Start with
   the production-outage one — watch it go critical + on-call escalation. Then the
   dark-mode feature request — low urgency, product team. The contrast tells the story.
3. **Architecture (15 min):** draw the 5-step pipeline live. Key talking points:
   - Why an agent workflow, not one big model: each step is inspectable, testable,
     and swappable — you can see the reasoning chain in the UI.
   - TF-IDF + LogisticRegression: fast, explainable, trains in seconds on CPU —
     the right tool for short-text classification; no GPU needed.
   - The urgency rule booster: ML gets you 80%, deterministic rules cover the
     critical tail the model might miss.
   - The ACT seam: templates today, LLM drop-in tomorrow — the pipeline doesn't care.
4. **Deep dives (15 min):** evaluation (accuracy per step, confusion on urgency),
   class imbalance in urgency labels, why not fine-tune an LLM for this, latency
   budget, monitoring drift in ticket language.
5. **Close (10 min):** "If I were training your learners: session 1 — text
   classification intuition; session 2 — building the agent pipeline;
   session 3 — from template replies to LLM generation."

## Prepared demo tickets (all verified against the trained models)

1. "Our production API has been down for two hours, all customers affected, fix this immediately!"
   → Technical Issue · CRITICAL · engineering (on-call escalation) · 1-hour SLA
2. "I was charged twice on my last invoice, please refund the duplicate payment immediately."
   → Billing · CRITICAL · billing-team (on-call escalation) · 1-hour SLA
3. "I'm locked out of my account and the password reset email never arrived."
   → Account Access · HIGH · support · 4-hour SLA
4. "It would be great to have dark mode in the dashboard for night shifts."
   → Feature Request · LOW · product · 48-hour SLA
5. "What are your support hours during public holidays? No rush."
   → General Inquiry · LOW · support · 48-hour SLA
