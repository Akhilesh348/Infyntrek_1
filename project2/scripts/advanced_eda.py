"""
Advanced EDA for MovieLens 100k
- Temporal dynamics (rating drift)
- User segmentation (clusters using ratings + demographics)
- Genre analysis
- Rating bias analysis (per-user, per-item)
- Temporal cold-start (first-rating analysis)

Saves plots and a markdown summary to outputs/advanced_report.md
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def ensure_outputs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_data(workspace: Path):
    raw_dir = workspace / "data" / "raw" / "ml-100k"
    ratings = pd.read_csv(raw_dir / "u.data", sep="\t", header=None,
                          names=["userId", "movieId", "rating", "timestamp"],
                          engine='python')

    # Read u.genre to get genre names
    genre_file = raw_dir / "u.genre"
    genres = []
    if genre_file.exists():
        with open(genre_file, encoding='latin-1') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                if '|' in ln:
                    name, _ = ln.split('|', 1)
                    genres.append(name)
    # Fallback: common MovieLens genres
    if not genres:
        genres = ['unknown', 'Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

    # Read u.item with genre flags
    items = pd.read_csv(raw_dir / "u.item", sep='|', header=None, encoding='latin-1')
    # movieId,title,release_date,video_release_date,imdb_url, then genre flags
    base_cols = ['movieId','title','release_date','video_release_date','imdb_url']
    n_genres = items.shape[1] - len(base_cols)
    genre_cols = genres[:n_genres]
    items.columns = base_cols + genre_cols

    # Read users
    users = pd.read_csv(raw_dir / "u.user", sep='|', header=None, encoding='latin-1')
    users.columns = ['userId','age','gender','occupation','zip']

    return ratings, items, users, genres


def temporal_dynamics(ratings: pd.DataFrame, out_dir: Path):
    ratings = ratings.copy()
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values('timestamp')

    # Monthly aggregates
    monthly = ratings.set_index('timestamp').resample('ME').agg(count=('rating','size'), mean_rating=('rating','mean'))

    plt.figure(figsize=(10,4))
    ax = monthly['mean_rating'].plot(marker='o')
    ax.set_title('Monthly average rating (MovieLens 100k)')
    ax.set_ylabel('Average rating')
    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_avg_rating_monthly.png')
    plt.close()

    plt.figure(figsize=(10,4))
    ax = monthly['count'].plot(kind='bar')
    ax.set_title('Monthly rating counts')
    ax.set_ylabel('Number of ratings')
    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_rating_counts_monthly.png')
    plt.close()

    # Rating drift per-popular-movies sample
    top_movies = ratings['movieId'].value_counts().head(10).index.tolist()
    sample = ratings[ratings['movieId'].isin(top_movies)].copy()
    sample['month'] = sample['timestamp'].dt.to_period('M').dt.to_timestamp()
    pivot = sample.groupby(['month','movieId'])['rating'].mean().unstack('movieId')
    pivot.plot(figsize=(10,6))
    plt.title('Monthly average rating for top-10 movies')
    plt.ylabel('Average rating')
    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_top10_movies.png')
    plt.close()

    return monthly


def user_segmentation(ratings: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame, out_dir: Path):
    # Basic user features
    user_stats = ratings.groupby('userId').agg(rating_count=('rating','size'), mean_rating=('rating','mean'), std_rating=('rating','std')).fillna(0)

    # Genre preference vectors per user
    # merge ratings with item genres
    genre_cols = [c for c in items.columns if c not in ['movieId','title','release_date','video_release_date','imdb_url']]
    rating_items = ratings.merge(items[['movieId'] + genre_cols], on='movieId', how='left')
    # Sum genre flags weighted by rating
    user_genres = rating_items.groupby('userId')[genre_cols].sum()
    # normalize
    user_genres_norm = user_genres.div(user_genres.sum(axis=1).replace(0,1), axis=0)

    user_features = pd.concat([user_stats, user_genres_norm], axis=1).fillna(0)

    # Optional clustering if sklearn available
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        X = user_features.values
        pca = PCA(n_components=min(10, X.shape[1]))
        Xp = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xp)
        user_features['cluster'] = labels

        # cluster summary
        cluster_summary = user_features.groupby('cluster').agg({'rating_count':'mean','mean_rating':'mean'})
        (out_dir / 'user_cluster_summary.csv').write_text(cluster_summary.to_csv())

        # plot cluster sizes
        plt.figure(figsize=(6,4))
        sns.countplot(x='cluster', data=user_features)
        plt.title('User cluster sizes')
        plt.tight_layout()
        plt.savefig(out_dir / 'user_cluster_sizes.png')
        plt.close()
    except Exception as e:
        user_features['cluster'] = -1
        (out_dir / 'user_cluster_summary.txt').write_text(f'Clustering skipped: {e}')

    # Save user features sample
    user_features.to_csv(out_dir / 'user_features.csv')
    return user_features


def genre_analysis(ratings: pd.DataFrame, items: pd.DataFrame, out_dir: Path):
    genre_cols = [c for c in items.columns if c not in ['movieId','title','release_date','video_release_date','imdb_url']]
    # For each genre compute average rating and counts
    df = ratings.merge(items[['movieId','title'] + genre_cols], on='movieId', how='left')
    genre_stats = []
    for g in genre_cols:
        sub = df[df[g] == 1]
        genre_stats.append({'genre': g, 'count': len(sub), 'mean_rating': sub['rating'].mean() if len(sub)>0 else np.nan})
    gs = pd.DataFrame(genre_stats).sort_values('count', ascending=False)
    gs.to_csv(out_dir / 'genre_stats.csv', index=False)

    plt.figure(figsize=(8,4))
    sns.barplot(x='mean_rating', y='genre', data=gs)
    plt.title('Mean rating by genre')
    plt.tight_layout()
    plt.savefig(out_dir / 'genre_mean_ratings.png')
    plt.close()

    plt.figure(figsize=(8,4))
    sns.barplot(x='count', y='genre', data=gs)
    plt.title('Rating counts by genre')
    plt.tight_layout()
    plt.savefig(out_dir / 'genre_counts.png')
    plt.close()

    return gs


def rating_bias_analysis(ratings: pd.DataFrame, out_dir: Path):
    global_mean = ratings['rating'].mean()
    user_bias = ratings.groupby('userId')['rating'].mean() - global_mean
    item_bias = ratings.groupby('movieId')['rating'].mean() - global_mean

    plt.figure(figsize=(6,4))
    sns.histplot(user_bias, bins=50)
    plt.title('Per-user rating bias (mean - global_mean)')
    plt.tight_layout()
    plt.savefig(out_dir / 'user_bias_hist.png')
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(item_bias, bins=50)
    plt.title('Per-item rating bias (mean - global_mean)')
    plt.tight_layout()
    plt.savefig(out_dir / 'item_bias_hist.png')
    plt.close()

    # Save bias stats
    bstats = pd.DataFrame({'user_bias': user_bias}).describe()
    (out_dir / 'user_bias_stats.csv').write_text(bstats.to_csv())
    (out_dir / 'item_bias_stats.csv').write_text(pd.DataFrame({'item_bias': item_bias}).describe().to_csv())

    return user_bias, item_bias


def temporal_cold_start(ratings: pd.DataFrame, items: pd.DataFrame, out_dir: Path):
    ratings = ratings.copy()
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    first_dates = ratings.groupby('movieId')['timestamp'].min().rename('first_rating')
    movie_first = items[['movieId','title']].merge(first_dates.reset_index(), on='movieId', how='left')
    movie_first = movie_first.sort_values('first_rating')
    movie_first.to_csv(out_dir / 'movie_first_ratings.csv', index=False)

    # For movies with early ratings, compare first-30-day average vs later
    ratings = ratings.merge(first_dates.reset_index(), on='movieId', how='left')
    ratings['days_since_first'] = (ratings['timestamp'] - ratings['first_rating']).dt.days

    early = ratings[ratings['days_since_first'] <= 30]
    later = ratings[ratings['days_since_first'] > 30]

    early_stats = early.groupby('movieId')['rating'].mean().rename('early_mean')
    later_stats = later.groupby('movieId')['rating'].mean().rename('later_mean')
    compare = pd.concat([early_stats, later_stats], axis=1)
    compare['delta'] = compare['early_mean'] - compare['later_mean']
    compare = compare.dropna()
    compare.describe().to_csv(out_dir / 'temporal_coldstart_compare_stats.csv')

    plt.figure(figsize=(6,4))
    sns.histplot(compare['delta'], bins=50)
    plt.title('Distribution of early_mean - later_mean (per movie)')
    plt.tight_layout()
    plt.savefig(out_dir / 'coldstart_delta_hist.png')
    plt.close()

    return compare


def write_summary(out_dir: Path, monthly, user_features, genre_stats, user_bias, item_bias, compare):
    lines = []
    lines.append('# Advanced EDA Summary')
    lines.append('')
    lines.append('## Temporal dynamics')
    lines.append(f'- Period covered (monthly points): {len(monthly)}')
    lines.append(f'- Overall monthly mean rating range: {monthly["mean_rating"].min():.3f} - {monthly["mean_rating"].max():.3f}')
    lines.append('')
    lines.append('## User segmentation')
    if 'cluster' in user_features.columns:
        lines.append(f'- Found {user_features["cluster"].nunique()} clusters (including -1 for skipped).')
    else:
        lines.append('- Clustering was not performed.')
    lines.append('')
    lines.append('## Genre analysis')
    lines.append('- Top genres by count:')
    for _, r in genre_stats.head(5).iterrows():
        lines.append(f"  - {r['genre']}: {int(r['count'])} ratings, mean={r['mean_rating']:.3f}")
    lines.append('')
    lines.append('## Rating bias')
    lines.append(f"- User bias mean: {user_bias.mean():.3f}, std: {user_bias.std():.3f}")
    lines.append(f"- Item bias mean: {item_bias.mean():.3f}, std: {item_bias.std():.3f}")
    lines.append('')
    lines.append('## Temporal cold-start')
    lines.append(f"- Movies with both early and later means: {len(compare)}")
    lines.append(f"- Mean delta (early - later): {compare['delta'].mean():.3f}")
    lines.append('')
    (out_dir / 'advanced_report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_advanced_eda():
    workspace = Path(__file__).resolve().parents[2]
    out_dir = workspace / 'outputs'
    ensure_outputs(out_dir)

    ratings, items, users, genres = load_data(workspace)
    monthly = temporal_dynamics(ratings, out_dir)
    user_features = user_segmentation(ratings, items, users, out_dir)
    genre_stats = genre_analysis(ratings, items, out_dir)
    user_bias, item_bias = rating_bias_analysis(ratings, out_dir)
    compare = temporal_cold_start(ratings, items, out_dir)
    write_summary(out_dir, monthly, user_features, genre_stats, user_bias, item_bias, compare)
    print('Advanced EDA completed. Outputs in', out_dir)


if __name__ == '__main__':
    run_advanced_eda()
