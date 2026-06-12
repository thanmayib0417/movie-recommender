"""
Hybrid Recommender
Blends Content-Based and Collaborative Filtering scores.

Strategies:
  - Weighted average: final = cb_weight * cb_score + (1 - cb_weight) * cf_score
  - Fallback: if CF has no data for user, fall back to pure content-based
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        cf_recommender: CollaborativeRecommender,
    ):
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender

    def recommend(
        self,
        user_id: int,
        movie_title: str,
        movies: pd.DataFrame,
        n: int = 10,
        cb_weight: float = 0.4,
        candidate_pool: int = 100,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations.

        Strategy:
            1. Get top `candidate_pool` movies by content similarity (to seed the pool).
            2. For each candidate, get CF predicted rating.
            3. Normalize both scores to [0,1] and blend with cb_weight.
            4. Return top-n.

        Args:
            user_id: Target user.
            movie_title: Seed movie for content similarity.
            movies: Full movies DataFrame.
            n: Number of final recommendations.
            cb_weight: Weight for content-based score (0 = pure CF, 1 = pure CB).
            candidate_pool: Size of content-based candidate set.

        Returns:
            DataFrame with [movieId, title, genres, cb_score, cf_score, hybrid_score]
        """
        # Step 1: Get large content-based candidate pool
        try:
            cb_recs = self.content_recommender.recommend(movie_title, n=candidate_pool)
        except ValueError:
            # Fallback to pure CF if movie not found
            return self.cf_recommender.recommend(user_id, movies, n=n)

        if len(cb_recs) == 0:
            return self.cf_recommender.recommend(user_id, movies, n=n)

        # Step 2: Get CF scores for the same candidates
        cf_preds = {}
        for movie_id in cb_recs["movieId"].tolist():
            cf_preds[movie_id] = self.cf_recommender.predict(user_id, movie_id)

        cb_recs = cb_recs.copy()
        cb_recs["cf_score_raw"] = cb_recs["movieId"].map(cf_preds)

        # Step 3: Normalize both to [0, 1]
        scaler = MinMaxScaler()

        cb_scores = cb_recs["score"].values.reshape(-1, 1)
        cf_scores = cb_recs["cf_score_raw"].values.reshape(-1, 1)

        cb_recs["cb_score"] = scaler.fit_transform(cb_scores).flatten()
        cb_recs["cf_score"] = scaler.fit_transform(cf_scores).flatten()

        # Step 4: Blend
        cb_recs["hybrid_score"] = (
            cb_weight * cb_recs["cb_score"]
            + (1 - cb_weight) * cb_recs["cf_score"]
        )

        result = (
            cb_recs[["movieId", "title", "genres", "cb_score", "cf_score", "hybrid_score"]]
            .sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        # Rename hybrid_score → score for UI compatibility
        result["score"] = result["hybrid_score"]
        return result

    def evaluate_cold_start(self, movies: pd.DataFrame, n_test: int = 20) -> dict:
        """
        Simple cold start test: pick random movies and show CB can still recommend.
        Returns coverage metrics.
        """
        sample = movies.sample(n_test)
        successes = 0
        for _, row in sample.iterrows():
            try:
                recs = self.content_recommender.recommend(row["title"], n=5)
                if len(recs) > 0:
                    successes += 1
            except Exception:
                pass
        return {
            "tested": n_test,
            "success": successes,
            "coverage": successes / n_test
        }
