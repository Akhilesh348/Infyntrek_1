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

## Week 2 Progress

### Day 1
- Loaded processed ratings and movies datasets
- Checked dataset shape and missing values
- Performed basic exploratory data analysis (EDA)

### Day 2
- Analyzed movie ratings
- Identified top-rated movies using average ratings
- Explored user-item interactions