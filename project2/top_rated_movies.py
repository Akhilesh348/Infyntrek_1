import pandas as pd

# Load ratings dataset
ratings = pd.read_csv("data/processed/ratings.csv")

# Calculate average rating and rating count for each movie
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    total_ratings=("rating", "count")
)

# Sort by average rating
top_movies = movie_stats.sort_values(
    by="average_rating",
    ascending=False
)

print("Top 10 Rated Movies")
print(top_movies.head(10))