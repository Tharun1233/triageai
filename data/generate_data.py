"""
Synthetic support-ticket dataset generator for TriageAI.
Seeded RNG -> identical dataset on every run (reproducible demo).
"""
import csv
import random

RNG = random.Random(42)

CATEGORIES = ["billing", "technical", "account_access", "feature_request", "general"]

TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month, please refund the duplicate charge.",
        "My invoice for {month} shows an amount I don't recognize. Can you explain the breakdown?",
        "Please cancel my subscription effective immediately and confirm I won't be billed again.",
        "I upgraded my plan yesterday but the new features are not unlocked yet I was charged already.",
        "The payment failed three times even though my card is valid. I need this resolved today.",
        "Why did my renewal price increase? I was not notified about any price change.",
        "I need a GST invoice for my last three payments for reimbursement.",
        "My trial ended and I got charged without any warning email. I want a refund.",
        "The discount coupon I applied at checkout did not reflect in the final amount.",
        "Please update the billing email on my account to {email}.",
    ],
    "technical": [
        "The dashboard keeps crashing whenever I export a report larger than 50MB.",
        "API requests are timing out since this morning, our integration is completely down.",
        "I'm getting a 500 error when I try to save settings on the integrations page.",
        "The mobile app freezes on the login screen after the latest update.",
        "Webhooks have stopped firing for the past two hours, nothing reaches our endpoint.",
        "Search returns no results even for items I know exist in the workspace.",
        "The export CSV contains garbled characters instead of proper text.",
        "Single sign-on is broken for our whole team after we rotated the certificate.",
        "Page load times have tripled today, the app is barely usable.",
        "I can't upload files larger than 10MB even though the limit says 100MB.",
        "Our production API is down, all customer requests are failing with timeouts.",
        "The service has been unreachable for an hour, our customers are affected.",
        "We are seeing a spike of 500 errors across all endpoints since the deploy.",
        "The database is extremely slow today, queries that took seconds now take minutes.",
        "Our checkout page is broken, customers cannot complete purchases.",
        "The staging environment went down right before our release window.",
    ],
    "account_access": [
        "I'm locked out of my account after too many login attempts, please reset it.",
        "I never received the password reset email, I've checked spam twice.",
        "Please enable two-factor authentication for all users in our organization.",
        "A former employee still has access to our workspace, revoke it immediately.",
        "I can't log in with Google SSO, it says my account is not provisioned.",
        "How do I transfer account ownership to my colleague? I'm leaving the company.",
        "My account was suspended for no reason, I need it reactivated urgently.",
        "The invite email for my teammate expired, please resend it.",
        "I want to delete my account and all associated data permanently.",
        "My role was changed from admin to viewer and I can't manage the team anymore.",
    ],
    "feature_request": [
        "It would be great to have dark mode in the dashboard, our team works nights.",
        "Please add bulk import for users via CSV, adding them one by one takes forever.",
        "We need Slack notifications for failed jobs, email alerts are too slow.",
        "Can you add support for exporting reports to PDF, not just CSV?",
        "It would help to have an audit log showing who changed what in settings.",
        "Please consider adding SSO support for Okta, we can't roll out without it.",
        "We'd love a public API endpoint for usage analytics.",
        "Add the ability to schedule reports weekly instead of only daily.",
        "It would be useful to pin important dashboards to the top of the home page.",
        "Please add multi-currency support for invoicing our international clients.",
    ],
    "general": [
        "What are your support hours during public holidays?",
        "Where can I find the documentation for the new API version?",
        "Do you offer onboarding sessions for new teams?",
        "What is the difference between the Pro and Enterprise plans?",
        "How do I contact sales for a custom quote?",
        "Is there a status page where I can check for ongoing incidents?",
        "Do you have a data processing agreement available for review?",
        "Where is my data hosted? We need EU residency for compliance.",
        "How do I change the language of the dashboard to Spanish?",
        "Can I get a trial extension for another two weeks to evaluate?",
    ],
}

URGENCY_PHRASES = {
    "critical": [
        "this is urgent", "immediately", "asap", "right now", "completely down",
        "losing money", "losing customers", "all users affected", "production is down",
        "need this fixed today", "blocking our launch",
    ],
    "high": [
        "urgent", "quickly", "as soon as possible", "important", "deadline",
        "can't work", "stuck", "escalate",
    ],
    "low": [
        "whenever you get a chance", "no rush", "just curious", "it would be nice",
        "sometime", "eventually",
    ],
}

BASE_URGENCY = {
    "billing": "medium",
    "technical": "high",
    "account_access": "high",
    "feature_request": "low",
    "general": "low",
}

MONTHS = ["January", "February", "March", "April", "May", "June"]


def make_ticket(category: str) -> tuple[str, str]:
    text = RNG.choice(TEMPLATES[category])
    text = text.replace("{month}", RNG.choice(MONTHS))
    text = text.replace("{email}", f"billing{RNG.randint(1,99)}@example.com")
    urgency = BASE_URGENCY[category]
    roll = RNG.random()
    if roll < 0.35:
        level = RNG.choice(["critical", "high", "low"])
        phrase = RNG.choice(URGENCY_PHRASES[level])
        text = f"{text} {phrase.capitalize()}." if RNG.random() < 0.5 else f"{phrase.capitalize()}, {text[0].lower() + text[1:]}"
        urgency = level
    return text, urgency


def main(n_per_category: int = 160, out: str = "data/tickets.csv"):
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_category):
            text, urgency = make_ticket(cat)
            rows.append((text, cat, urgency))
    RNG.shuffle(rows)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "category", "urgency"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} tickets to {out}")


if __name__ == "__main__":
    main()
