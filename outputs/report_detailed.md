# Detailed Report: MovieLens 100K Dataset Analysis

## 1. Project Objective
This analysis focused on exploring the MovieLens 100K dataset—a benchmark recommender systems dataset containing 100,000 ratings from 943 users on 1,682 movies. The goal was to:

1. Download and extract the raw MovieLens data
2. Preprocess and clean the data into standardized formats
3. Perform comprehensive exploratory data analysis (EDA)
4. Generate insights about user rating patterns, movie popularity, and data characteristics
5. Prepare the data foundation for future recommender system modeling

This report documents what was accomplished, explains the value of each analysis step, interprets the findings, and outlines future work directions.

---

## 2. Step-by-Step Work Completed

### Step 1: Data Acquisition and Raw Data Understanding
**What was done:**
- Downloaded the MovieLens 100K dataset from GroupLens (via `download_movielens.py`)
- Extracted raw data files:
  - `u.data`: User-item-rating triplets with timestamps (100,000 records)
  - `u.item`: Movie catalog with titles and metadata (1,682 movies)
  - Additional reference files (genres, user demographics, etc.)

**Why this step matters:**
- Establishes ground truth for all downstream analysis
- Enables reproducibility by using a stable, publicly available benchmark dataset
- Provides structured data in standard formats (tab-separated and pipe-separated files)

**Key outcome:**
- 100,000 ratings spanning 943 users and 1,682 unique movies
- Raw data successfully extracted to `data/raw/ml-100k/`

---

### Step 2: Data Preprocessing and Normalization
**What was done (from `preprocess.py`):**
- Loaded ratings from `u.data` with proper parsing:
  - Column mapping: userId | movieId | rating | timestamp
  - Data type conversion to appropriate numeric types
- Loaded movie metadata from `u.item`:
  - Extracted: movieId, title, release_date
- Applied transformations:
  - **Dropped timestamp column** (not needed for initial EDA)
  - **Normalized ratings** from 1-5 scale to 0-1 range using formula: `(rating - 1) / 4`
    - This standardization enables later modeling that expects bounded scales
- Saved clean outputs:
  - `ratings.csv`: Contains userId, movieId, rating, and rating_norm
  - `movies.csv`: Contains movieId, title, and release_date

**Why normalization is important:**
- Machine learning models often perform better with normalized inputs
- 0-1 range is standard for probability-based models and neural networks
- Preserves original rating information while enabling scale-invariant analysis

**Key outcome:**
- 100,000 clean rating records ready for analysis
- Data saved to `data/processed/` in CSV format for portability

---

### Step 3: Exploratory Data Analysis (EDA)
**What was done (from `eda.py`):**

#### **3a. Summary Statistics**
Generated and displayed:
- Total ratings: **100,000**
- Unique users: **943**
- Unique movies: **1,682**
- Average rating: **3.530** (on 1-5 scale)
- Data sparsity: **0.9370** (93.7% of user-item matrix is empty)

#### **3b. Visualizations Created**
1. **Rating Counts Distribution** (`rating_counts.png`)
   - Bar chart of how many ratings each score (1, 2, 3, 4, 5) received
   
2. **Normalized Rating Distribution** (`rating_norm_dist.png`)
   - Histogram of normalized ratings (0-1 scale) with 20 bins
   
3. **User Activity Distribution** (`ratings_per_user.png`)
   - Histogram showing how many ratings each user has given
   - Reveals user engagement heterogeneity

4. **Top 20 Movies by Rating Count** (`top20_by_count.png`)
   - Horizontal bar chart of most-rated movies
   - Top movie: Star Wars (1977) with 583 ratings

5. **Average Rating vs Number of Ratings** (`avg_rating_vs_count.png`)
   - Scatter plot with log-scale x-axis
   - Shows relationship between popularity (rating count) and quality (average rating)

#### **3c. Key Statistics Report** (`report.md`)
Generated markdown summary with top 20 movies by rating count

**Recorded outcome:**
- 5 visualization files generated
- Summary statistics computed and saved
- Sample user-item matrix (50×50) exported to `user_item_sample.csv`

---

## 3. Why This EDA Analysis is Useful

### **Understanding User-Item Interaction Patterns**
- **Purpose:** Recommender systems must understand how users engage with items
- **Value:** High sparsity (93.7%) indicates most users rate very few movies, requiring collaborative filtering or content-based approaches rather than simple matrix completion
- **Actionable insight:** Dense user neighborhoods (similar users) or item neighborhoods (similar movies) are rare

### **Identifying Data Quality and Coverage**
- **Purpose:** Assess whether the dataset is balanced and complete
- **Value:** Average rating of 3.530 suggests slight positive bias (ratings > 3 tend to outnumber low ratings), common in real rating datasets
- **Actionable insight:** Future models should account for rating skew; simple averaging may be biased

### **Understanding Content Popularity**
- **Purpose:** Discover which items dominate the dataset and which are niche
- **Value:** Top movie (Star Wars) has 583 ratings while many have <10, indicating long-tail distribution
- **Actionable insight:** Recommender systems must handle "cold-start" problem (new/niche items with few ratings)

### **User Engagement Heterogeneity**
- **Purpose:** Understand variation in user participation levels
- **Value:** Rating counts distribution reveals some users are highly active (many ratings) while others rate rarely
- **Actionable insight:** User-specific factors (experience level, domain knowledge) should be considered in modeling

### **Rating Quality vs Popularity**
- **Purpose:** Test whether popular items receive inflated ratings
- **Value:** Scatter plot of average_rating vs. rating_count helps detect systematic biases
- **Example pattern:** Frequently rated movies may have different average scores than rarely rated ones
- **Actionable insight:** Separate concerns: item popularity vs. item quality in recommendations

---

## 4. Key Outcomes and What They Suggest

### **Outcome 1: Severe Sparsity (93.7%)**
**Finding:** The user-item matrix is 93.7% empty—most (user, movie) pairs are unobserved.

**What this suggests:**
- Standard matrix factorization will work well (sparsity is expected for collaborative filtering)
- Hybrid approaches (combining user/item content) are necessary for cold-start scenarios
- Implicit feedback models (using rating presence itself) may not add value beyond explicit ratings
- Block-structure analysis is warranted: are there user groups with distinct preferences?

**Implication for modeling:** Recommendation engines must use dimensionality reduction (SVD, NMF, neural embeddings) rather than direct similarity metrics.

---

### **Outcome 2: Moderate Average Rating (3.530)**
**Finding:** Mean rating is above scale midpoint (2.5), indicating positive sentiment bias.

**What this suggests:**
- Users tend to rate movies they like (selection bias): no one rates a movie they didn't watch
- Few very low ratings—the dataset lacks comprehensive coverage of bad movies
- Rating distributions may be right-skewed rather than normal
- Simple averaging for recommendations is biased upward

**Implication for modeling:** Consider Bayesian priors that shrink poorly-rated items toward the population mean; account for the positive bias when comparing predicted vs. actual ratings.

---

### **Outcome 3: Long-Tail Distribution (Top Movie 583 ratings, many <10)**
**Finding:** Top 20 movies account for a disproportionate fraction of all ratings; many movies have very few ratings.

**What this suggests:**
- Blockbuster effect: major releases dominate user attention and dataset volume
- Cold-start challenge: new or niche movies have insufficient data for pattern learning
- Content-based features (genre, director, year) are critical for low-data items
- Ensemble approaches (hybrid recommender) can leverage genre + user similarity

**Implication for modeling:** Two-stage systems work well: use collaborative filtering for popular items, content-based for niche items.

---

### **Outcome 4: Heterogeneous User Engagement**
**Finding:** Rating counts per user vary widely (histogram shows long tail).

**What this suggests:**
- Super-users exist: a small fraction of users provide most ratings
- Casual users exist: majority provide few ratings
- Super-user ratings may have disproportionate influence on item vectors
- User biases (some users consistently rate high/low) are likely

**Implication for modeling:** Normalization by user mean rating and variance is essential; confidence weighting by user experience level may improve results.

---

## 5. Future Work and Recommendations

### **Phase 1: Advanced Exploratory Analysis**
- [ ] **Temporal dynamics:** Analyze how ratings evolve over time; detect rating drift
- [ ] **User segmentation:** Cluster users by rating patterns (demographics are available in `u.user`)
- [ ] **Genre analysis:** Explore genre preferences across user segments
- [ ] **Rating bias analysis:** Quantify per-user and per-item rating bias distributions
- [ ] **Temporal cold-start:** Track when movies entered the dataset; analyze early adoption patterns

---

### Advanced EDA Results (new)
The advanced EDA script produced time-series, clustering, genre, bias, and cold-start analyses. Key results summary (full outputs in the `outputs/` folder):

- **Temporal dynamics:** monthly average rating range 3.398 - 3.591; see [outputs/temporal_avg_rating_monthly.png](outputs/temporal_avg_rating_monthly.png) and [outputs/temporal_rating_counts_monthly.png](outputs/temporal_rating_counts_monthly.png).
- **User segmentation:** 4 clusters discovered; cluster sizes saved in [outputs/user_cluster_summary.csv](outputs/user_cluster_summary.csv) and visualization at [outputs/user_cluster_sizes.png](outputs/user_cluster_sizes.png).
- **Genre analysis:** Top genres by rating count — Drama (39895, mean=3.687), Comedy (29832, mean=3.394), Action (25589, mean=3.480), Thriller (21872, mean=3.509), Romance (19461, mean=3.622). See [outputs/genre_stats.csv](outputs/genre_stats.csv) and plots [outputs/genre_mean_ratings.png](outputs/genre_mean_ratings.png), [outputs/genre_counts.png](outputs/genre_counts.png).
- **Rating bias:** per-user bias mean=0.058 (std=0.445), per-item bias mean=-0.454 (std=0.782). Histograms saved at [outputs/user_bias_hist.png](outputs/user_bias_hist.png) and [outputs/item_bias_hist.png](outputs/item_bias_hist.png).
- **Temporal cold-start:** compared early (first 30 days) vs later ratings for movies; mean delta (early - later) ≈ 0.009 across movies with both periods. Details in [outputs/temporal_coldstart_compare_stats.csv](outputs/temporal_coldstart_compare_stats.csv) and [outputs/coldstart_delta_hist.png](outputs/coldstart_delta_hist.png).

A machine-readable summary was also saved to [outputs/advanced_report.md](outputs/advanced_report.md).

---

### Detailed methods and rationale (Advanced EDA)

Below are the exact actions performed for each advanced analysis, the outputs produced, and why each step is useful for understanding the dataset and informing next-stage modeling.

1) Temporal dynamics
- What I did: Converted the `timestamp` column to datetimes, resampled ratings by month, and computed monthly aggregates (count and mean). For high-activity items I computed per-month average ratings and plotted trajectories for the top-10 most-rated movies.
- Files produced: `outputs/temporal_avg_rating_monthly.png`, `outputs/temporal_rating_counts_monthly.png`, `outputs/temporal_top10_movies.png`.
- Why it's useful: Detects rating drift, seasonality, and bursts of activity. Temporal patterns indicate whether models should use time-aware splits (temporal holdout), incorporate time decay, or retrain frequently. The top-movie trajectories show whether popular items' perceived quality changes over time.

2) User segmentation
- What I did: Built per-user statistics (rating count, mean, std) and per-user genre-preference vectors by joining ratings with the movie genre flags from `u.item`. I reduced dimensionality with PCA and applied KMeans to find 4 clusters; cluster summaries and sizes were saved.
- Files produced: `outputs/user_features.csv`, `outputs/user_cluster_summary.csv`, `outputs/user_cluster_sizes.png`.
- Why it's useful: Segmentation reveals user cohorts (e.g., heavy raters, niche-genre preferrers) which helps tailor personalization strategies, choose weighting schemes, and design evaluation strata (test per-cluster performance). Clusters also identify super-users whose behavior can dominate learned item embeddings.

3) Genre analysis
- What I did: Merged ratings with item genre flags and aggregated rating counts and mean ratings per genre, then plotted mean rating and total rating counts per genre.
- Files produced: `outputs/genre_stats.csv`, `outputs/genre_mean_ratings.png`, `outputs/genre_counts.png`.
- Why it's useful: Genre-level signals are essential for cold-start recommendations and for hybrid models that combine content with collaborative signals. Observed differences in mean rating by genre inform whether to bias recommendations or tune model regularization per-genre.

4) Rating bias analysis
- What I did: Computed the global mean rating and then calculated per-user and per-item bias as (mean_rating - global_mean). Plotted histograms of these biases and saved descriptive statistics.
- Files produced: `outputs/user_bias_hist.png`, `outputs/item_bias_hist.png`, `outputs/user_bias_stats.csv`, `outputs/item_bias_stats.csv`.
- Why it's useful: Bias distributions reveal systematic tendencies: users who consistently rate high or low, and items that are consistently over- or under-rated. These biases are the basis of baseline models (global mean + user bias + item bias) and indicate how much regularization is required in factorization models.

5) Temporal cold-start analysis
- What I did: For each movie, found the first timestamp it received a rating, labeled ratings by days since that first rating, and compared the mean rating in the first 30 days (early) versus later. I computed per-movie deltas (early_mean - later_mean) and plotted the distribution.
- Files produced: `outputs/movie_first_ratings.csv`, `outputs/temporal_coldstart_compare_stats.csv`, `outputs/coldstart_delta_hist.png`.
- Why it's useful: Early-adopter bias (or novelty effects) can cause initial ratings to differ from later consensus. Quantifying this helps design evaluation windows (avoid optimistic early metrics), warm-up strategies for new items, and time-aware regularization.

Reproducibility and next steps
- The advanced EDA was implemented in `project2/scripts/advanced_eda.py`. Re-running that script regenerates all outputs in `outputs/`.
- Next recommended steps: profile clusters with demographics and top genres, run temporal train/test splits for baseline models, and use genre features to bootstrap cold-start item profiles.


### **Phase 2: Recommender System Development**
- [ ] **Baseline models:**
  - Global mean + user bias + item bias (biased MF)
  - User-based collaborative filtering (k-NN with cosine similarity)
  - Item-based collaborative filtering
  
- [ ] **Matrix factorization models:**
  - Singular Value Decomposition (SVD) with bias terms
  - Non-negative Matrix Factorization (NMF) for interpretability
  - Alternating Least Squares (ALS)
  
- [ ] **Hybrid approaches:**
  - Content-based filtering using genre data
  - Combined models (collab + content)
  
- [ ] **Neural/embedding approaches:**
  - Autoencoders for implicit feedback
  - Deep learning with user/item embeddings

### **Phase 3: Evaluation and Validation**
- [ ] **Train/test split strategy:** Temporal holdout (predict future ratings) or random
- [ ] **Metrics:** RMSE, MAE, NDCG, Recall@k, Precision@k
- [ ] **Cross-validation:** k-fold to handle cold-start scenarios
- [ ] **Baseline comparisons:** Compare all models against random + global mean + user/item bias
- [ ] **Sensitivity analysis:** How do hyperparameters (embedding dim, regularization) affect performance?

### **Phase 4: Business Application**
- [ ] **Deployment pipeline:** REST API for real-time recommendations
- [ ] **Cold-start resolution:** Strategies for new users and new items
- [ ] **A/B testing framework:** Evaluate recommendations against production baseline
- [ ] **Explainability:** Feature importance, "why this recommendation" explanations
- [ ] **Real-time updates:** Incremental model updates as new ratings arrive

### **Phase 5: Extended Analysis (Optional)**
- [ ] **Implicit feedback:** Compare explicit ratings (1-5) with implicit signals (rating presence as proxy)
- [ ] **Fairness and bias:** Detect genre-based recommendation biases
- [ ] **Diversity metrics:** Ensure recommendations cover a range of genres, not just popular items
- [ ] **Novelty analysis:** Balance accuracy with serendipity (unexpected but good recommendations)

---

## 6. Conclusion

The MovieLens 100K analysis has successfully established the data foundation for recommendation system research. The dataset exhibits:
- **Expected sparsity** suitable for collaborative filtering
- **Positive rating bias** requiring careful baseline handling
- **Long-tail popularity distribution** necessitating hybrid approaches
- **Heterogeneous user engagement** calling for per-user normalization

With these insights, the project is now ready to progress toward building and evaluating recommender models that account for these characteristics. The preprocessing and EDA outputs provide a clean, well-understood data foundation for all subsequent work.