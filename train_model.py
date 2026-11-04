from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "news_dataset.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "news_classifier.joblib"

def main():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["text", "label"])
    data["label"] = data["label"].astype(str).str.upper().str.strip()
    data = data[data["label"].isin(["REAL", "FAKE"])]
    if data["label"].nunique() != 2 or len(data) < 10:
        raise ValueError("Dataset must contain at least 10 rows and both REAL and FAKE labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].astype(str), data["label"],
        test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000,
                                  ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test accuracy on this dataset: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix [FAKE, REAL] order:")
    print(confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"]))

if __name__ == "__main__":
    main()
