"""
TriageAI agent: a deterministic agent workflow for support-ticket triage.

Pipeline (each step feeds the next, like an agent's reasoning chain):
    1. PERCEIVE  - classify the ticket category (ML model)
    2. ASSESS    - score urgency (ML model + rule boosters)
    3. SENSE     - detect customer sentiment (lexicon)
    4. DECIDE    - route to the right team (policy over steps 1-3)
    5. ACT       - draft a first-response reply (grounded template)

The ACT step is template-based by design for the demo; the code marks the
exact seam where an LLM call would slot in for generative replies.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import re

BASE = Path(__file__).parent.parent
MODELS = BASE / "models"

TEAM_FOR_CATEGORY = {
    "billing": "billing-team",
    "technical": "engineering",
    "account_access": "support",
    "feature_request": "product",
    "general": "support",
}

CATEGORY_LABEL = {
    "billing": "Billing",
    "technical": "Technical Issue",
    "account_access": "Account Access",
    "feature_request": "Feature Request",
    "general": "General Inquiry",
}

POSITIVE_WORDS = {"great", "love", "thanks", "thank", "awesome", "excellent",
                  "helpful", "glad", "appreciate", "wonderful", "perfect"}
NEGATIVE_WORDS = {"angry", "frustrated", "terrible", "awful", "worst", "hate",
                  "broken", "useless", "disappointed", "furious", "ridiculous",
                  "unacceptable", "outraged", "pathetic"}

CRITICAL_HINTS = [r"\burgent\b", r"\bimmediately\b", r"\basap\b", r"\bright now\b",
                  r"\bdown\b", r"\blocking\b", r"\blosing (money|customers)\b",
                  r"\ball users\b", r"\bproduction\b"]

REPLY_TEMPLATES = {
    "billing": ("Thanks for reaching out — I've flagged this for our billing team "
                "as {urgency} priority. We'll review the charge details and get back "
                "to you within {sla}. Your reference: {ticket_ref}."),
    "technical": ("Sorry you're hitting this — our engineering team has this as a "
                  "{urgency} priority item. We've logged the error details and will "
                  "update you within {sla}. Your reference: {ticket_ref}."),
    "account_access": ("Account access issues are never fun — this is with our support "
                       "team now at {urgency} priority. Expect an update within {sla}. "
                       "Your reference: {ticket_ref}."),
    "feature_request": ("Thanks for the suggestion! I've logged this with our product "
                        "team. We review requests like this every sprint — reference: "
                        "{ticket_ref}."),
    "general": ("Thanks for getting in touch! Our support team will answer within "
                "{sla}. Your reference: {ticket_ref}."),
}

SLA = {"critical": "1 hour", "high": "4 hours", "medium": "24 hours", "low": "48 hours"}


@dataclass
class TriageDecision:
    category: str
    category_label: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    sentiment: str
    routed_team: str
    sla: str
    suggested_reply: str
    reasoning: list

    def to_dict(self):
        return asdict(self)


class TriageAgent:
    def __init__(self):
        self.category_model = joblib.load(MODELS / "category_model.joblib")
        self.urgency_model = joblib.load(MODELS / "urgency_model.joblib")

    # ---- Step 1: PERCEIVE ----
    def perceive(self, text: str):
        probs = self.category_model.predict_proba([text])[0]
        idx = probs.argmax()
        return self.category_model.classes_[idx], float(probs[idx])

    # ---- Step 2: ASSESS ----
    def assess(self, text: str):
        probs = self.urgency_model.predict_proba([text])[0]
        idx = probs.argmax()
        urgency, conf = self.urgency_model.classes_[idx], float(probs[idx])
        lowered = text.lower()
        if any(re.search(p, lowered) for p in CRITICAL_HINTS) and urgency != "critical":
            return "critical", max(conf, 0.85), "rule booster: critical keyword"
        return urgency, conf, None

    # ---- Step 3: SENSE ----
    @staticmethod
    def sense(text: str):
        words = set(re.findall(r"[a-z]+", text.lower()))
        neg = len(words & NEGATIVE_WORDS)
        pos = len(words & POSITIVE_WORDS)
        if neg > pos:
            return "negative"
        if pos > neg:
            return "positive"
        return "neutral"

    # ---- Step 4: DECIDE ----
    @staticmethod
    def decide(category: str, urgency: str, sentiment: str):
        team = TEAM_FOR_CATEGORY[category]
        reason = f"category '{category}' maps to {team}"
        if urgency == "critical":
            team = f"{team} (on-call escalation)"
            reason += "; critical urgency escalates to on-call"
        if sentiment == "negative" and urgency in ("high", "critical"):
            reason += "; negative sentiment keeps priority high"
        return team, reason

    # ---- Step 5: ACT ----
    @staticmethod
    def act(category: str, urgency: str, ticket_ref: str):
        # SEAM: replace this template call with an LLM draft for generative replies.
        return REPLY_TEMPLATES[category].format(
            urgency=urgency, sla=SLA[urgency], ticket_ref=ticket_ref)

    def triage(self, text: str, ticket_ref: str = "TRIAGE-DEMO") -> TriageDecision:
        reasoning = []
        category, cat_conf = self.perceive(text)
        reasoning.append(f"PERCEIVE: '{CATEGORY_LABEL[category]}' ({cat_conf:.0%} confidence)")
        urgency, urg_conf, booster = self.assess(text)
        reasoning.append(f"ASSESS: urgency '{urgency}' ({urg_conf:.0%} confidence)"
                         + (f" [{booster}]" if booster else ""))
        sentiment = self.sense(text)
        reasoning.append(f"SENSE: customer sentiment '{sentiment}'")
        team, why = self.decide(category, urgency, sentiment)
        reasoning.append(f"DECIDE: routed to {team} — {why}")
        reply = self.act(category, urgency, ticket_ref)
        reasoning.append("ACT: first-response draft prepared for human review")
        return TriageDecision(
            category=category,
            category_label=CATEGORY_LABEL[category],
            category_confidence=round(cat_conf, 3),
            urgency=urgency,
            urgency_confidence=round(urg_conf, 3),
            sentiment=sentiment,
            routed_team=team,
            sla=SLA[urgency],
            suggested_reply=reply,
            reasoning=reasoning,
        )
