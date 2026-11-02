"""
Train the TriageAI models: ticket category classifier + urgency classifier.
TF-IDF + LogisticRegression. Saves artifacts to models/.
"""
import csv
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent


def load_data(path: Path):
    texts, cats, urg = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            cats.append(row["category"])
            urg.append(row["urgency"])
    return texts, cats, urg


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                  sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
    ])


def main():
    texts, cats, urg = load_data(BASE / "data" / "tickets.csv")
    print(f"Loaded {len(texts)} tickets")

    for name, labels in [("category", cats), ("urgency", urg)]:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"\n[{name}] test accuracy: {acc:.3f}")
        print(classification_report(y_test, pipe.predict(X_test)))
        joblib.dump(pipe, BASE / "models" / f"{name}_model.joblib")
        print(f"Saved models/{name}_model.joblib")


if __name__ == "__main__":
    main()
