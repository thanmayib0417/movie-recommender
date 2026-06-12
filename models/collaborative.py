"""
Collaborative Filtering Recommender
Uses Singular Value Decomposition (SVD) via the `surprise` library.
Falls back to a simple user-item matrix if surprise is not installed.
"""

import pandas as pd
import numpy as np
import os
import pickle


class CollaborativeRecommender:
    """
    Collaborative filtering via matrix factorization (SVD).

    Trains on explicit user-item ratings.
    Predicts ratings for unseen (user, movie) pairs.
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.algo = None
        self.trainset = None
        self.all_movie_ids = None
        self._use_surprise = True

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, ratings: pd.DataFrame):
        """
        Train SVD on the ratings DataFrame.

        Args:
            ratings: DataFrame with columns [userId, movieId, rating]
        """
        self.all_movie_ids = ratings["movieId"].unique()

        try:
            from surprise import Dataset, Reader, SVD
            from surprise.model_selection import cross_validate

            reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
            data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
            self.trainset = data.build_full_trainset()

            self.algo = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                verbose=False
            )
            self.algo.fit(self.trainset)
            self._use_surprise = True
            print("CF model (SVD) fitted via surprise.")

        except ImportError:
            print("surprise not installed. Falling back to simple bias model.")
            self._fit_fallback(ratings)

    def _fit_fallback(self, ratings: pd.DataFrame):
        """Simple global mean + user bias + item bias fallback."""
        self._use_surprise = False
        self._global_mean = ratings["rating"].mean()
        self._user_bias = ratings.groupby("userId")["rating"].mean() - self._global_mean
        self._item_bias = ratings.groupby("movieId")["rating"].mean() - self._global_mean
        self._rated = ratings.groupby("userId")["movieId"].apply(set).to_dict()

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a (user, movie) pair."""
        if self._use_surprise:
            return self.algo.predict(user_id, movie_id).est
        else:
            u_bias = self._user_bias.get(user_id, 0)
            i_bias = self._item_bias.get(movie_id, 0)
            return float(np.clip(self._global_mean + u_bias + i_bias, 0.5, 5.0))

    # ── Recommendation ──────────────────────────────────────────────────────

    def recommend(self, user_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Return top-n predicted movies for a user, excluding already-rated ones.

        Args:
            user_id: Target user ID.
            movies: Full movies DataFrame [movieId, title, genres].
            n: Number of recommendations.

        Returns:
            DataFrame with [movieId, title, genres, score]
        """
        # Movies this user hasn't rated
        if self._use_surprise:
            try:
                rated = set(
                    iid for (uid, iid, _) in self.trainset.all_ratings()
                    if self.trainset.to_raw_uid(uid) == user_id
                )
            except Exception:
                rated = set()
        else:
            rated = self._rated.get(user_id, set())

        candidate_ids = [m for m in self.all_movie_ids if m not in rated]

        scores = [(mid, self.predict(user_id, mid)) for mid in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n]

        top_ids = [s[0] for s in top]
        top_scores = {s[0]: s[1] for s in top}

        result = movies[movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
        result["score"] = result["movieId"].map(top_scores)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        return result

    def get_all_predictions(self, user_id: int) -> pd.Series:
        """Return predicted ratings for all movies as a Series keyed by movieId."""
        scores = {mid: self.predict(user_id, mid) for mid in self.all_movie_ids}
        return pd.Series(scores)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"CF model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CollaborativeRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
