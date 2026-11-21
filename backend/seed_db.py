import os
import sqlite3
from datetime import datetime, timedelta
from ml_engine import analyze_review

DB_PATH = os.path.join(os.path.dirname(__file__), "insightify.db")

SAMPLE_REVIEWS = [
    ("Aarav Sharma", "The build quality is amazing and delivery was super fast, but the price is way too expensive."),
    ("Priya Nair", "Customer support never replied to my refund request and the app keeps crashing."),
    ("Rohan Verma", "Great value for money! The interface is super easy to use and performance is smooth."),
    ("Sneha Patel", "Package arrived two days late and the screen had a small scratch. Very disappointed."),
    ("Vikram Singh", "Battery life is solid and the design feels premium. Totally worth the price."),
    ("Ananya Iyer", "The website navigation is confusing and checkout was painfully slow."),
    ("Kabir Mehta", "Fast shipping and neat packaging, though the material feels a bit cheap."),
    ("Meera Joshi", "Support staff resolved my warranty issue in 10 minutes. Best customer service!"),
    ("Arjun Das", "App performance lags badly after the latest update. Constant glitches."),
    ("Divya Rao", "Affordable pricing and decent quality for everyday use. Works as expected."),
    ("Siddharth Roy", "Overpriced for what you get. The camera quality is average at best."),
    ("Neha Gupta", "Super smooth UI and fast delivery. Definitely recommending this to friends."),
    ("Karan Malhotra", "Delayed courier and zero response from customer support. Never buying again."),
    ("Pooja Reddy", "Durable build and great speed, though the setup dashboard was slightly confusing."),
    ("Rahul Deshmukh", "Terrible experience. Product arrived broken and the return process is a nightmare."),
    ("Ishita Kapoor", "Love the sleek design and quick performance. Worth every rupee."),
    ("Varun Khanna", "Shipping was fast, but the app UI has too many bugs."),
    ("Tanvi Kulkarni", "Good quality material and very affordable price. Happy with the purchase."),
]


def seed_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"Seeding {len(SAMPLE_REVIEWS)} reviews into {DB_PATH}...")
    base_time = datetime.utcnow() - timedelta(days=len(SAMPLE_REVIEWS))

    for i, (name, text) in enumerate(SAMPLE_REVIEWS):
        result = analyze_review(text)
        created_at = (base_time + timedelta(days=i)).isoformat()

        cursor.execute(
            "INSERT INTO reviews (customer_name, review_text, sentiment, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, text, result["sentiment"], result["confidence"], created_at),
        )
        review_id = cursor.lastrowid

        for asp in result["aspects"]:
            cursor.execute(
                "INSERT INTO review_aspects (review_id, aspect, sentiment, confidence, snippet) VALUES (?, ?, ?, ?, ?)",
                (review_id, asp["aspect"], asp["sentiment"], asp["confidence"], asp["snippet"]),
            )

    conn.commit()
    conn.close()
    print("Database seeded successfully.")


if __name__ == "__main__":
    seed_database()