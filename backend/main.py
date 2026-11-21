import json
import os
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml_engine import analyze_review

DB_PATH = os.path.join(os.path.dirname(__file__), "insightify.db")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_metrics.json")

app = FastAPI(title="Insightify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            category TEXT NOT NULL DEFAULT 'General',
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # Safe migration for existing insightify.db without losing saved reviews
    cols = [row[1] for row in cursor.execute("PRAGMA table_info(reviews)").fetchall()]
    if "category" not in cols:
        cursor.execute("ALTER TABLE reviews ADD COLUMN category TEXT NOT NULL DEFAULT 'General'")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS review_aspects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            aspect TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            snippet TEXT,
            FOREIGN KEY (review_id) REFERENCES reviews (id)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


class ReviewRequest(BaseModel):
    customer_name: Optional[str] = "Anonymous"
    category: Optional[str] = "General"
    review_text: str = Field(..., min_length=3, max_length=2000)


@app.post("/api/analyze")
def analyze_and_save(req: ReviewRequest):
    clean_text = req.review_text.strip()
    if len(clean_text) < 3 or not any(ch.isalpha() for ch in clean_text):
        raise HTTPException(
            status_code=422,
            detail="Review text must contain at least 3 characters with readable words.",
        )

    customer_name = (req.customer_name or "Anonymous").strip() or "Anonymous"
    category = (req.category or "General").strip() or "General"
    result = analyze_review(clean_text)
    created_at = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reviews (customer_name, category, review_text, sentiment, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (customer_name, category, clean_text, result["sentiment"], result["confidence"], created_at),
    )
    review_id = cursor.lastrowid

    for asp in result["aspects"]:
        cursor.execute(
            "INSERT INTO review_aspects (review_id, aspect, sentiment, confidence, snippet) "
            "VALUES (?, ?, ?, ?, ?)",
            (review_id, asp["aspect"], asp["sentiment"], asp["confidence"], asp["snippet"]),
        )

    conn.commit()
    conn.close()

    return {
        "id": review_id,
        "customer_name": customer_name,
        "category": category,
        "review_text": clean_text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "aspects": result["aspects"],
        "created_at": created_at,
    }


@app.get("/api/dashboard")
def get_dashboard(category: Optional[str] = Query(default=None)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    categories = [
        row["category"]
        for row in cursor.execute(
            "SELECT DISTINCT category FROM reviews WHERE category IS NOT NULL AND TRIM(category) != '' ORDER BY category ASC"
        ).fetchall()
    ]

    filter_cat = category.strip() if category and category.strip() not in ("", "All Products") else None

    if filter_cat:
        total_reviews = cursor.execute(
            "SELECT COUNT(*) FROM reviews WHERE category = ?", (filter_cat,)
        ).fetchone()[0]
        sent_rows = cursor.execute(
            "SELECT sentiment, COUNT(*) AS cnt FROM reviews WHERE category = ? GROUP BY sentiment",
            (filter_cat,),
        ).fetchall()
        asp_rows = cursor.execute(
            """
            SELECT ra.aspect, ra.sentiment, COUNT(*) AS cnt
            FROM review_aspects ra
            JOIN reviews r ON ra.review_id = r.id
            WHERE r.category = ?
            GROUP BY ra.aspect, ra.sentiment
            """,
            (filter_cat,),
        ).fetchall()
        recent_rows = cursor.execute(
            "SELECT id, customer_name, category, review_text, sentiment, confidence, created_at "
            "FROM reviews WHERE category = ? ORDER BY id DESC LIMIT 20",
            (filter_cat,),
        ).fetchall()
    else:
        total_reviews = cursor.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        sent_rows = cursor.execute(
            "SELECT sentiment, COUNT(*) AS cnt FROM reviews GROUP BY sentiment"
        ).fetchall()
        asp_rows = cursor.execute(
            "SELECT aspect, sentiment, COUNT(*) AS cnt FROM review_aspects GROUP BY aspect, sentiment"
        ).fetchall()
        recent_rows = cursor.execute(
            "SELECT id, customer_name, category, review_text, sentiment, confidence, created_at "
            "FROM reviews ORDER BY id DESC LIMIT 20"
        ).fetchall()

    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for row in sent_rows:
        if row["sentiment"] in sentiment_counts:
            sentiment_counts[row["sentiment"]] = row["cnt"]

    aspects_summary = {}
    for row in asp_rows:
        asp = row["aspect"]
        if asp not in aspects_summary:
            aspects_summary[asp] = {"Positive": 0, "Neutral": 0, "Negative": 0}
        if row["sentiment"] in aspects_summary[asp]:
            aspects_summary[asp][row["sentiment"]] = row["cnt"]

    conn.close()

    return {
        "active_category": filter_cat or "All Products",
        "categories": categories,
        "total_reviews": total_reviews,
        "sentiment_counts": sentiment_counts,
        "aspects_summary": aspects_summary,
        "recent_reviews": [dict(r) for r in recent_rows],
    }


@app.get("/api/evaluation")
def get_evaluation():
    if not os.path.exists(EVAL_PATH):
        return {"message": "eval_metrics.json not found. Run python train_eval.py inside backend/."}
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)