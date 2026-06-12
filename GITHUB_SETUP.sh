# GitHub Setup Instructions
# Run these commands in your terminal from inside the project folder

# ── Step 1: Initialize git ────────────────────────────────────────────────────
git init
git add .
git status   # double-check .env is NOT listed (it should be greyed out / ignored)

# ── Step 2: First commit ──────────────────────────────────────────────────────
git commit -m "Initial commit: hybrid movie recommender with TMDB posters"

# ── Step 3: Create repo on GitHub ────────────────────────────────────────────
# Go to github.com → New Repository
# Name it: movie-recommender
# Keep it Public (good for portfolio)
# Do NOT initialize with README (we already have one)
# Click "Create repository"

# ── Step 4: Connect and push ──────────────────────────────────────────────────
git remote add origin https://github.com/YOUR_USERNAME/movie-recommender.git
git branch -M main
git push -u origin main

# ── Step 5: Verify .env is gitignored ─────────────────────────────────────────
# Run this — .env should NOT appear in the output
git ls-files | grep .env
# Only .env.example should show up, never .env itself

# ── Future pushes ─────────────────────────────────────────────────────────────
# git add .
# git commit -m "your message"
# git push
