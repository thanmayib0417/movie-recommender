# 🎬 CineMatch — Hybrid Movie Recommender

A portfolio-grade hybrid recommender system combining **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)** on the MovieLens dataset, with a Streamlit UI and TMDB movie poster integration.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

> Poster grid mode (requires free TMDB API key):

Recommendations appear as a visual poster grid with match scores, TMDB ratings, and synopsis popups.

---

## Architecture

```
movie-recommender/
├── app.py                        # Streamlit UI
├── requirements.txt
├── .env                          # Your API key (gitignored)
├── .env.example                  # Safe template to commit
├── models/
│   ├── content_based.py          # TF-IDF on genres + tags + title
│   ├── collaborative.py          # SVD via scikit-surprise
│   └── hybrid.py                 # Weighted blend of both
├── utils/
│   ├── data_loader.py            # Auto-downloads MovieLens
│   ├── tmdb.py                   # TMDB poster + metadata fetching
│   └── evaluation.py             # RMSE, Precision@K, Recall@K, NDCG@K
└── notebooks/
    └── eda_and_train.py          # EDA + training walkthrough
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your TMDB API key
cp .env.example .env
# Edit .env and paste your key: TMDB_API_KEY=your_key_here

# 5. Launch the app
streamlit run app.py
```

The MovieLens dataset (~3MB) downloads automatically on first run.

---

## How It Works

### Content-Based Filtering
- Builds a **TF-IDF matrix** from genres (upweighted 3×), user tags, and title keywords
- Recommends movies with highest **cosine similarity** to a seed movie
- Solves the **cold start problem** — works even for movies with zero ratings

### Collaborative Filtering (SVD)
- Matrix factorization via **Singular Value Decomposition**
- Learns latent user and item factors from 100k explicit ratings
- Predicts unseen (user, movie) ratings and ranks by score

### Hybrid Blending
```
hybrid_score = cb_weight × cb_score + (1 − cb_weight) × cf_score
```
Both scores are normalized to [0,1] before blending. Weight is tunable via the UI slider.

### TMDB Integration
- Free TMDB API fetches poster images, TMDB ratings, and plot overviews
- Results cached for 24h to avoid redundant API calls
- Graceful fallback to list view if no key is provided

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE / MAE | Rating prediction accuracy |
| Precision@K | % of top-K recs that are relevant |
| Recall@K | % of relevant items in top-K |
| NDCG@K | Ranking quality with position weighting |
| Cold-start coverage | % of zero-rating movies CB handles |

Run `python notebooks/eda_and_train.py` for the full evaluation report.

---

## Getting a TMDB API Key

1. Sign up free at [themoviedb.org](https://www.themoviedb.org/signup)
2. Go to **Settings → API → Request an API Key**
3. Application name: `Movie Recommender`, URL: `http://localhost:8501`
4. Copy the key into your `.env` file

---

## Dataset

[MovieLens Small](https://grouplens.org/datasets/movielens/latest/) — 100k ratings, 9k movies, 600 users. Auto-downloaded on first run.

For production scale, swap in [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/).

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data processing |
| scikit-learn | TF-IDF, cosine similarity, normalization |
| scikit-surprise | SVD collaborative filtering |
| streamlit | Web UI |
| requests + python-dotenv | TMDB API + env management |
| matplotlib, seaborn | EDA visualizations |
