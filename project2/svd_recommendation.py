import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Load ratings
ratings = pd.read_csv("data/processed/ratings.csv")

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# Apply SVD
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_svd = svd.fit_transform(user_movie_matrix)

print("Original Matrix Shape:", user_movie_matrix.shape)
print("Reduced Matrix Shape:", matrix_svd.shape)

print("\nExplained Variance Ratio:")
print(svd.explained_variance_ratio_.sum())