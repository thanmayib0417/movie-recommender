"""
Evaluation Metrics for Recommender Systems

Implements:
    - RMSE / MAE  (rating prediction accuracy)
    - Precision@K  (ranking quality)
    - Recall@K     (ranking quality)
    - Coverage     (catalog coverage)
    - Novelty      (popularity-adjusted diversity)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple


# ── Rating Accuracy ──────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate_cf_surprise(algo, testset) -> Dict[str, float]:
    """
    Evaluate a fitted Surprise algorithm on a testset.

    Returns dict with rmse and mae.
    """
    from surprise import accuracy
    predictions = algo.test(testset)
    return {
        "rmse": accuracy.rmse(predictions, verbose=False),
        "mae": accuracy.mae(predictions, verbose=False),
        "n_predictions": len(predictions)
    }


# ── Ranking Metrics ──────────────────────────────────────────────────────────

def precision_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int
) -> float:
    """
    Precision@K: fraction of top-k recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended movie IDs.
        relevant: Set of relevant (ground-truth liked) movie IDs.
        k: Cutoff.
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k


def recall_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int
) -> float:
    """
    Recall@K: fraction of relevant items found in top-k.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(
    recommended: List[int],
    relevant: List[int],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    """
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant_set
    )
    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_ranking(
    cf_model,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    n_users: int = 100,
    threshold: float = 4.0
) -> Dict[str, float]:
    """
    Evaluate ranking quality using leave-one-out on a sample of users.

    For each user:
        - Hold out their highest-rated movies (>= threshold) as ground truth.
        - Generate top-k recommendations.
        - Compute Precision@K, Recall@K, NDCG@K.

    Args:
        cf_model: Fitted CollaborativeRecommender.
        ratings: Full ratings DataFrame.
        movies: Movies DataFrame.
        k: Cutoff for ranking metrics.
        n_users: Number of users to evaluate (subset for speed).
        threshold: Minimum rating to count as "relevant".
    """
    sampled_users = (
        ratings["userId"]
        .value_counts()
        .loc[lambda x: x >= 20]  # users with enough history
        .index[:n_users]
        .tolist()
    )

    precisions, recalls, ndcgs = [], [], []

    for user_id in sampled_users:
        user_ratings = ratings[ratings["userId"] == user_id]
        relevant = user_ratings[user_ratings["rating"] >= threshold]["movieId"].tolist()

        if not relevant:
            continue

        try:
            recs = cf_model.recommend(user_id, movies, n=k)
            rec_ids = recs["movieId"].tolist()

            precisions.append(precision_at_k(rec_ids, relevant, k))
            recalls.append(recall_at_k(rec_ids, relevant, k))
            ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
        except Exception:
            continue

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions)
    }


# ── Catalog Metrics ──────────────────────────────────────────────────────────

def catalog_coverage(
    recommended_sets: List[List[int]],
    all_items: List[int]
) -> float:
    """
    Fraction of the catalog that appears in at least one recommendation list.
    """
    recommended_all = set(item for recs in recommended_sets for item in recs)
    return len(recommended_all) / len(all_items)


def novelty(
    recommended_sets: List[List[int]],
    item_popularity: pd.Series
) -> float:
    """
    Average novelty = mean self-information of recommended items.
    Less popular items are more novel.

    Args:
        recommended_sets: List of recommendation lists.
        item_popularity: Series mapping movieId -> popularity (e.g. rating count).
    """
    total = item_popularity.sum()
    scores = []
    for recs in recommended_sets:
        for item in recs:
            pop = item_popularity.get(item, 1)
            scores.append(-np.log2(pop / total + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


# ── Full Evaluation Report ───────────────────────────────────────────────────

def full_evaluation_report(
    cf_model,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    n_users: int = 50
) -> pd.DataFrame:
    """Run all metrics and return a summary DataFrame."""
    ranking = evaluate_ranking(cf_model, ratings, movies, k=k, n_users=n_users)

    results = {**ranking}
    return pd.DataFrame([results])
