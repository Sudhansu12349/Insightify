import re
from transformers import pipeline

# Pre-trained 3-class sentiment model (Positive, Neutral, Negative)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print(f"Loading sentiment model ({MODEL_NAME})...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    truncation=True,
    max_length=512,
)
print("Model loaded successfully.")

# Predefined aspects and their trigger keywords
ASPECT_KEYWORDS = {
    "Quality": ["quality", "build", "material", "durable", "broken", "defective", "design", "camera", "screen", "battery"],
    "Price": ["price", "cost", "expensive", "cheap", "worth", "value", "money", "pricing", "overpriced", "affordable"],
    "Delivery": ["delivery", "shipping", "arrived", "package", "packaging", "late", "fast", "courier", "delayed"],
    "Support": ["support", "service", "customer", "refund", "return", "help", "response", "staff", "warranty"],
    "UI/UX": ["ui", "ux", "interface", "app", "website", "dashboard", "easy", "confusing", "navigation", "bug"],
    "Performance": ["fast", "slow", "lag", "crash", "performance", "speed", "smooth", "glitch", "hang"],
}

LABEL_MAP = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
}


def predict_sentiment(text: str) -> dict:
    """Run HuggingFace sentiment inference on a single string."""
    result = sentiment_pipeline(text)[0]
    raw_label = result["label"].lower()
    return {
        "sentiment": LABEL_MAP.get(raw_label, "Neutral"),
        "confidence": round(float(result["score"]), 4),
    }


def extract_aspects(text: str) -> list[dict]:
    """Split review into clauses and score any clause matching a known aspect."""
    # Split on sentence boundaries and contrast conjunctions
    clauses = re.split(r"[.!?;,]+|\b(?:but|however|although|though|yet)\b", text, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 3]

    found_aspects = {}

    for clause in clauses:
        clause_lower = clause.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", clause_lower) for kw in keywords):
                pred = predict_sentiment(clause)
                # Keep the highest-confidence prediction if an aspect appears twice
                if aspect not in found_aspects or pred["confidence"] > found_aspects[aspect]["confidence"]:
                    found_aspects[aspect] = {
                        "aspect": aspect,
                        "sentiment": pred["sentiment"],
                        "confidence": pred["confidence"],
                        "snippet": clause,
                    }

    return list(found_aspects.values())


def analyze_review(text: str) -> dict:
    """Full analysis: overall sentiment + aspect-level sentiment."""
    cleaned = text.strip()
    if not cleaned:
        return {"sentiment": "Neutral", "confidence": 0.0, "aspects": []}

    overall = predict_sentiment(cleaned)
    aspects = extract_aspects(cleaned)

    return {
        "sentiment": overall["sentiment"],
        "confidence": overall["confidence"],
        "aspects": aspects,
    }


if __name__ == "__main__":
    sample_reviews = [
        "The build quality is amazing and delivery was super fast, but the price is way too expensive.",
        "Customer support never replied to my refund request and the app keeps crashing.",
        "Decent product for the money, works as expected.",
    ]

    for rev in sample_reviews:
        print("\nReview:", rev)
        print("Result:", analyze_review(rev))