import json
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from ml_engine import predict_sentiment

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_metrics.json")

# Curated labeled dataset (Positive / Negative) for fast offline benchmarking
DATASET = [
    ("Absolutely love this product, build quality is top notch.", "Positive"),
    ("Fast delivery and works smoothly right out of the box.", "Positive"),
    ("Great value for the price, highly recommended.", "Positive"),
    ("Customer support was super helpful and resolved my issue fast.", "Positive"),
    ("The interface is clean, intuitive, and responsive.", "Positive"),
    ("Battery lasts all day and performance is blazing fast.", "Positive"),
    ("Exceeded my expectations in every way. Worth every penny.", "Positive"),
    ("Packaging was neat and shipping arrived ahead of schedule.", "Positive"),
    ("Very durable material and sleek design.", "Positive"),
    ("Easy to set up and runs without any lag.", "Positive"),
    ("Best purchase I have made this year, fantastic quality.", "Positive"),
    ("Affordable, reliable, and performs better than pricier brands.", "Positive"),
    ("Smooth experience from ordering to delivery.", "Positive"),
    ("The dashboard is well designed and easy to navigate.", "Positive"),
    ("Prompt refund and polite support staff, great service.", "Positive"),
    ("Solid performance and zero crashes so far.", "Positive"),
    ("Terrible quality, broke within two days of use.", "Negative"),
    ("Delivery was delayed by a week with no tracking updates.", "Negative"),
    ("Way too overpriced for such cheap plastic material.", "Negative"),
    ("Customer support completely ignored my emails.", "Negative"),
    ("The app crashes every time I try to open settings.", "Negative"),
    ("Extremely slow performance and constant freezing.", "Negative"),
    ("Arrived damaged and the return process is frustrating.", "Negative"),
    ("Confusing UI, impossible to find basic features.", "Negative"),
    ("Complete waste of money, do not buy this.", "Negative"),
    ("Battery drains in two hours, totally unusable.", "Negative"),
    ("Worst customer service experience I have ever had.", "Negative"),
    ("Package was torn and missing accessories.", "Negative"),
    ("Full of bugs and glitches after the latest update.", "Negative"),
    ("Does not work as advertised, very disappointed.", "Negative"),
    ("Poor build quality and terrible audio output.", "Negative"),
    ("Checkout failed twice and charged my card.", "Negative"),
]


def run_benchmark():
    texts = [x[0] for x in DATASET]
    labels = [x[1] for x in DATASET]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.35, random_state=42, stratify=labels
    )

    # 1. Baseline: TF-IDF + Linear SVM
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    svm = LinearSVC(random_state=42)
    svm.fit(X_train_vec, y_train)
    svm_preds = svm.predict(X_test_vec)
    svm_time_ms = round(((time.perf_counter() - t0) / len(X_test)) * 1000, 2)

    # 2. Transformer: Cached HuggingFace model
    t1 = time.perf_counter()
    hf_preds = []
    for text in X_test:
        pred = predict_sentiment(text)["sentiment"]
        # Map Neutral to Negative if using a 3-class model on binary benchmark
        hf_preds.append(pred if pred in ("Positive", "Negative") else "Negative")
    hf_time_ms = round(((time.perf_counter() - t1) / len(X_test)) * 1000, 2)

    def calc_metrics(y_true, y_pred, latency):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, pos_label="Positive", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, pos_label="Positive", zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, pos_label="Positive", zero_division=0), 4),
            "avg_latency_ms": latency,
        }

    metrics = {
        "dataset_size": len(DATASET),
        "test_samples": len(X_test),
        "models": {
            "Transformer (HuggingFace)": calc_metrics(y_test, hf_preds, hf_time_ms),
            "TF-IDF + Linear SVM": calc_metrics(y_test, svm_preds, svm_time_ms),
        },
    }

    with open(EVAL_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to {EVAL_PATH}:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_benchmark()