"""
Simple EDA on processed MovieLens data. Produces summary outputs and plots in `outputs/`.
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_outputs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_eda(processed_dir: Path, out_dir: Path):
    ensure_outputs(out_dir)

    ratings = pd.read_csv(processed_dir / "ratings.csv")
    movies = pd.read_csv(processed_dir / "movies.csv")

    # Basic summaries
    num_ratings = len(ratings)
    num_users = ratings["userId"].nunique()
    num_movies = ratings["movieId"].nunique()
    avg_rating = ratings["rating"].mean()

    print("Ratings shape:", ratings.shape)
    print(ratings.head())
    print(f"Number of ratings: {num_ratings}")
    print(f"Users: {num_users}")
    print(f"Movies: {num_movies}")
    print(f"Average rating: {avg_rating:.3f}")

    # Rating distribution (original)
    plt.figure(figsize=(6,4))
    sns.countplot(x="rating", data=ratings)
    plt.title("Rating Counts")
    plt.savefig(out_dir / "rating_counts.png")
    plt.close()

    # Normalized rating distribution
    plt.figure(figsize=(6,4))
    sns.histplot(ratings["rating_norm"], bins=20)
    plt.title("Normalized Rating Distribution")
    plt.savefig(out_dir / "rating_norm_dist.png")
    plt.close()

    # User activity
    user_counts = ratings["userId"].value_counts()
    plt.figure(figsize=(8,4))
    sns.histplot(user_counts, bins=30)
    plt.title("Ratings per User (distribution)")
    plt.savefig(out_dir / "ratings_per_user.png")
    plt.close()

    # Top movies by number of ratings
    top_movies = ratings.groupby("movieId").size().sort_values(ascending=False).head(20)
    top_movies = top_movies.reset_index()
    top_movies.columns = ["movieId", "count"]
    top_movies = top_movies.merge(movies, on="movieId", how="left")

    plt.figure(figsize=(10,6))
    sns.barplot(y="title", x="count", data=top_movies)
    plt.title("Top 20 Movies by Rating Count")
    plt.tight_layout()
    plt.savefig(out_dir / "top20_by_count.png")
    plt.close()

    # Average rating vs count (scatter)
    stats = ratings.groupby("movieId").agg(count=("rating", "size"), mean_rating=("rating", "mean")).reset_index()
    stats = stats.merge(movies, on="movieId", how="left")

    plt.figure(figsize=(8,6))
    sns.scatterplot(x="count", y="mean_rating", data=stats)
    plt.xscale('log')
    plt.xlabel("Number of Ratings (log scale)")
    plt.ylabel("Average Rating")
    plt.title("Average Rating vs Number of Ratings per Movie")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_rating_vs_count.png")
    plt.close()

    # Create user-item matrix sample and compute sparsity
    user_item = ratings.pivot(index="userId", columns="movieId", values="rating")
    total_possible = user_item.shape[0] * user_item.shape[1]
    sparsity = 1.0 - (ratings.shape[0] / total_possible)

    # Save a small sample of the user-item matrix for inspection
    sample_ui = user_item.iloc[:50, :50]
    sample_ui.to_csv(out_dir / "user_item_sample.csv")

    # Heatmap of the sample (presence heatmap)
    plt.figure(figsize=(10,8))
    sns.heatmap(~sample_ui.isna(), cbar=False)
    plt.title("User-Item Interaction Presence (sample 50x50)")
    plt.tight_layout()
    plt.savefig(out_dir / "user_item_heatmap_sample.png")
    plt.close()

    # Save report
    report_lines = [
        "# EDA Report",
        "",
        f"Number of ratings: {num_ratings}",
        f"Users: {num_users}",
        f"Movies: {num_movies}",
        f"Average rating: {avg_rating:.3f}",
        f"Sparsity (fraction missing): {sparsity:.4f}",
        "",
        "Top 20 movies by number of ratings:",
    ]

    for _, row in top_movies.iterrows():
        report_lines.append(f"- {row['title']} (movieId={row['movieId']}): {row['count']} ratings")

    report_text = "\n".join(report_lines)
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")

    print("EDA plots and report saved to", out_dir)


def main():
    workspace = Path(__file__).resolve().parents[2]
    processed_dir = workspace / "data" / "processed"
    out_dir = workspace / "outputs"
    run_eda(processed_dir, out_dir)


if __name__ == "__main__":
    main()
