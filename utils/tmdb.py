"""
TMDB API utility for fetching movie posters and metadata.
Results are cached so we don't hammer the API on repeated lookups.
"""

import re
import requests
import streamlit as st

TMDB_BASE = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w300"


def _clean_title(title: str) -> tuple[str, str | None]:
    """
    Strip year suffix from MovieLens titles like 'Toy Story (1995)'.
    Returns (clean_title, year_string_or_None).
    """
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if match:
        year = match.group(1)
        clean = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()
        return clean, year
    return str(title).strip(), None


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_details(title: str, year, api_key: str) -> dict:
    """
    Fetch poster URL, overview, TMDB rating, and release date.
    Returns empty dict on any failure — caller handles the missing case.
    """
    if not api_key:
        return {}

    # Always strip year from title before searching
    clean_title, extracted_year = _clean_title(title)
    search_year = extracted_year or (str(int(year)) if year and str(year) not in ("nan", "None", "") else None)

    # Try with year first, then without if no results
    for attempt_year in ([search_year, None] if search_year else [None]):
        try:
            params = {"api_key": api_key, "query": clean_title}
            if attempt_year:
                params["year"] = attempt_year

            resp = requests.get(
                f"{TMDB_BASE}/search/movie",
                params=params,
                timeout=6
            )

            if resp.status_code != 200:
                continue

            results = resp.json().get("results", [])
            if not results:
                continue

            result = results[0]
            path = result.get("poster_path")

            return {
                "poster_url": f"{POSTER_BASE}{path}" if path else None,
                "overview": result.get("overview", ""),
                "tmdb_rating": result.get("vote_average", 0),
                "release_date": result.get("release_date", ""),
            }

        except Exception:
            continue

    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def search_movie_poster(title: str, year, api_key: str) -> str | None:
    """Convenience wrapper — returns just the poster URL or None."""
    details = get_movie_details(title, year, api_key)
    return details.get("poster_url")