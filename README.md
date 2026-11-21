# Insightify — AI-Driven Product Review Sentiment & Aspect Analysis

Insightify is a full-stack NLP web application that analyzes customer product reviews in real time, extracts aspect-level sentiment (e.g., Quality, Price, Delivery, Performance, Support, UI/UX), persists results in SQLite, and compares Transformer performance against a classical ML baseline.

---

## Core Features

1. **Overview Dashboard (`GET /api/dashboard`)**
   - Displays live KPIs (total reviews and Positive / Neutral / Negative percentages) pulled directly from SQLite.
   - Renders an interactive sentiment distribution donut chart, aspect-level stacked sentiment bars, and a recent reviews feed.

2. **Live Sentiment & Aspect Analyzer (`POST /api/analyze`)**
   - Runs local **HuggingFace DistilBERT** inference on customer review text.
   - Performs clause-level aspect extraction to identify mixed sentiment across different product features, along with confidence scores and key explanation snippets.
   - Saves every analyzed review and its extracted aspects permanently to `backend/insightify.db`.

3. **Review Inbox & Feedback Queue**
   - Syncs all analyzed reviews from SQLite into a searchable, filterable queue.
   - Allows queuing new pending reviews and triggering live AI inference with automatic persistence across restarts.

4. **Model Evaluation (`GET /api/evaluation`)**
   - Loads offline benchmark metrics (`eval_metrics.json`) comparing the **DistilBERT Transformer** against a **TF-IDF + Linear SVM** baseline across Accuracy, Precision, Recall, F1-Score, and average CPU inference latency.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **NLP / Machine Learning:** HuggingFace Transformers (DistilBERT), PyTorch, Scikit-learn (TF-IDF + Linear SVM)
- **Database:** SQLite3 (`reviews` and `review_aspects` relational schema)
- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript, Custom SVG Charts

---

## Project Structure

```text
insightify/
├── backend/
│   ├── main.py                # FastAPI application & SQLite persistence routes
│   ├── ml_engine.py           # DistilBERT inference & clause-level aspect extraction
│   ├── seed_db.py             # Seeds initial sample reviews into insightify.db
│   ├── train_eval.py          # Offline evaluation (DistilBERT vs. TF-IDF + SVM)
│   └── eval_metrics.json      # Generated benchmark metrics
├── frontend/
│   └── insightify_product_review_intelligence.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local Setup & Running the App

### 1. Clone the Repository & Create a Virtual Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/<your-username>/<your-repo-name>.git
cd insightify
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd insightify
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Seed the Database & Run Evaluation Benchmarks
```powershell
cd backend
python seed_db.py
python train_eval.py
```

### 4. Start the FastAPI Backend (Port `8001`)
```powershell
python -m uvicorn main:app --reload --port 8001
```
Once running, interactive Swagger API docs are available at `http://127.0.0.1:8001/docs`.

### 5. Launch the Frontend Dashboard
Open `frontend/insightify_product_review_intelligence.html` directly in your browser (or serve it locally from the `frontend/` folder via `python -m http.server 5500`).

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/analyze` | Analyzes a review using DistilBERT, extracts aspects, and saves to SQLite |
| `GET` | `/api/dashboard` | Returns live review totals, sentiment breakdown, aspect summaries, and recent reviews |
| `GET` | `/api/evaluation` | Returns offline DistilBERT vs. TF-IDF + SVM benchmark metrics |

---

## Future Improvements

- Fine-tuning DistilBERT on domain-specific product review datasets to better handle sarcasm and oxymorons.
- Supporting multi-row CSV batch inference via a dedicated bulk API endpoint.
- Deploying with a managed cloud database (PostgreSQL) and containerized ML hosting.