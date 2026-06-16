# Project 2 — Personalized Recommendation System

This project downloads the MovieLens 100k dataset, cleans and preprocesses it, and performs basic exploratory data analysis (EDA).

Steps:

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Download the dataset:

```bash
python scripts/download_movielens.py
```

3. Preprocess the raw files:

```bash
python scripts/preprocess.py
```

4. Run EDA:

```bash
python scripts/eda.py
```

Outputs:

- `data/raw/ml-100k/` — raw downloaded dataset
- `data/processed/ratings.csv` — cleaned ratings (normalized)
- `data/processed/movies.csv` — movie metadata
- `outputs/` — EDA plots and summaries
 

## 📌 Week 2 Progress (SVD Movie Recommendation System)

Day 1

-Loaded processed ratings and movies datasets
-Checked dataset shape and missing values
-Performed basic exploratory data analysis (EDA)

Day 2

-Analyzed movie ratings
-Calculated average ratings per movie
-Identified top-rated movies
-Computed rating counts per movie

Day 3

-Built Item-Based Collaborative Filtering model
-Created movie-user interaction matrix
-Calculated cosine similarity between movies
-Generated top 5 similar movie recommendations

Day 4

-Implemented Singular Value Decomposition (SVD)
-Created user-movie rating matrix
-Applied matrix factorization using TruncatedSVD
-Reduced dimensionality of recommendation system

Day 5

-Built SVD recommendation model using Surprise library
-Trained model on user-item ratings
-Learned collaborative filtering using latent factors

Day 6

-Generated predictions for unseen user-item pairs
-Created Top-N movie recommendations per user
-Ranked movies based on predicted ratings

Day 7

-Evaluated model performance
-Calculated RMSE (Root Mean Squared Error)
-Calculated MAE (Mean Absolute Error)
-Analyzed prediction accuracy

🎯 Final Outcome

-Built a complete Movie Recommendation System
-Implemented both similarity-based and SVD-based models
-Evaluated model performance successfully