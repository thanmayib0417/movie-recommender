"""
Exploratory Data Analysis + Model Training Script
Run this first to understand the dataset, then use app.py for the UI.

Usage:
    python notebooks/eda_and_train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data, get_movie_stats
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.evaluation import full_evaluation_report

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 5)

# 1. LOAD DATA

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

movies, ratings, tags = load_data()
movie_stats = get_movie_stats(movies, ratings)

print(f"\nMovies shape:  {movies.shape}")
print(f"Ratings shape: {ratings.shape}")
print(f"Tags shape:    {tags.shape}")
print(f"\nMovies sample:\n{movies.head(3)}")
print(f"\nRatings sample:\n{ratings.head(3)}")

# 2. EDA

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# Rating distribution
print(f"\nRating distribution:\n{ratings['rating'].value_counts().sort_index()}")
print(f"\nMean rating: {ratings['rating'].mean():.2f}")
print(f"Ratings per user (mean): {ratings.groupby('userId').size().mean():.1f}")
print(f"Ratings per movie (mean): {ratings.groupby('movieId').size().mean():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Rating distribution
ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#E50914")
axes[0].set_title("Rating Distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

# Plot 2: Ratings per user (log scale)
user_counts = ratings.groupby("userId").size()
axes[1].hist(user_counts, bins=50, color="#0074D9", edgecolor="white")
axes[1].set_yscale("log")
axes[1].set_title("Ratings per User (log scale)")
axes[1].set_xlabel("# Ratings")

# Plot 3: Top genres
genre_counts = (
    movies["genres"].dropna()
    .str.split(", ")
    .explode()
    .value_counts()
    .head(12)
)
genre_counts.plot(kind="barh", ax=axes[2], color="#2ECC40")
axes[2].set_title("Top 12 Genres")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
plt.show()
print("\nEDA plot saved to eda_overview.png")

# 3. TRAIN CONTENT-BASED MODEL

print("\n" + "=" * 60)
print("STEP 3: Training Content-Based Model")
print("=" * 60)

cb = ContentBasedRecommender(use_embeddings=False)
cb.fit(movies, tags)

# Quick sanity check
test_movie = "Toy Story (1995)"
print(f"\nContent-based recommendations for: '{test_movie}'")
cb_recs = cb.recommend(test_movie, n=5)
print(cb_recs[["title", "genres", "score"]].to_string(index=False))

# 4. TRAIN COLLABORATIVE FILTERING MODEL

print("\n" + "=" * 60)
print("STEP 4: Training Collaborative Filtering Model (SVD)")
print("=" * 60)

cf = CollaborativeRecommender(n_factors=100, n_epochs=20)
cf.fit(ratings)

# Quick sanity check
test_user = ratings["userId"].value_counts().index[0]  # most active user
print(f"\nCF recommendations for user {test_user}:")
cf_recs = cf.recommend(test_user, movies, n=5)
print(cf_recs[["title", "genres", "score"]].to_string(index=False))

# 5. HYBRID RECOMMENDATIONS

print("\n" + "=" * 60)
print("STEP 5: Hybrid Recommendations")
print("=" * 60)

hybrid = HybridRecommender(cb, cf)

print(f"\nHybrid recommendations for user {test_user}, seeded by '{test_movie}':")
h_recs = hybrid.recommend(test_user, test_movie, movies, n=5, cb_weight=0.4)
print(h_recs[["title", "genres", "cb_score", "cf_score", "hybrid_score"]].to_string(index=False))

# 6. EVALUATION

print("\n" + "=" * 60)
print("STEP 6: Evaluation Metrics")
print("=" * 60)

# SVD cross-validation (if surprise available)
try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    algo = SVD(n_factors=100, n_epochs=20, verbose=False)

    print("\nRunning 3-fold cross-validation (this takes ~1 min)...")
    cv_results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=True)
    print(f"\nCross-validation results:")
    print(f"  RMSE: {cv_results['test_rmse'].mean():.4f} ± {cv_results['test_rmse'].std():.4f}")
    print(f"  MAE:  {cv_results['test_mae'].mean():.4f} ± {cv_results['test_mae'].std():.4f}")

except ImportError:
    print("Install `scikit-surprise` for cross-validation: pip install scikit-surprise")

# Ranking metrics
print("\nComputing ranking metrics (Precision@10, Recall@10, NDCG@10)...")
eval_df = full_evaluation_report(cf, ratings, movies, k=10, n_users=50)
print(eval_df.to_string(index=False))

# 7. COLD START DEMO

print("\n" + "=" * 60)
print("STEP 7: Cold Start Comparison")
print("=" * 60)

cold_start = hybrid.evaluate_cold_start(movies, n_test=30)
print(f"\nCold-start coverage: {cold_start['success']}/{cold_start['tested']} "
      f"({cold_start['coverage']:.0%}) — content-based handles all of these.")
print("\nSummary: CF would fail for new movies (no ratings). Content-based recovers gracefully.")
print("\nAll done! Run `streamlit run app.py` to launch the UI.")
