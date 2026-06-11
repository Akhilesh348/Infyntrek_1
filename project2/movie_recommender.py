import pandas as pd

movies = pd.read_csv("data/processed/movies.csv")
ratings = pd.read_csv("data/processed/ratings.csv")

print("Movies Shape:", movies.shape)
print("Ratings Shape:", ratings.shape)

print("\nMovies Sample:")
print(movies.head())

print("\nRatings Sample:")
print(ratings.head())

print("\nMissing Values:")
print(ratings.isnull().sum())

print("\nUnique Users:", ratings["userId"].nunique())
print("Unique Movies:", ratings["movieId"].nunique())

print("\nRating Statistics:")
print(ratings["rating"].describe())