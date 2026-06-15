import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    ratings = pd.read_csv("data/processed/ratings.csv")
    movies = pd.read_csv("data/processed/movies.csv")
    return ratings, movies


def create_movie_similarity_matrix(ratings):
    movie_user_matrix = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating"
    ).fillna(0)

    similarity_matrix = cosine_similarity(movie_user_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    return similarity_df


def get_similar_movies(movie_id, similarity_df, movies, top_n=5):
    similar_movies = similarity_df[movie_id].sort_values(
        ascending=False
    )[1:top_n + 1]

    print(f"\nTop {top_n} movies similar to Movie ID {movie_id}:\n")

    for similar_movie_id, score in similar_movies.items():
        movie_title = movies[movies["movieId"] == similar_movie_id]["title"].values

        if len(movie_title) > 0:
            print(f"{movie_title[0]}: {score:.3f}")
        else:
            print(f"Movie ID {similar_movie_id}: {score:.3f}")


try:
    ratings, movies = load_data()

    similarity_df = create_movie_similarity_matrix(ratings)

    movie_id = 1

    get_similar_movies(movie_id, similarity_df, movies)

except FileNotFoundError:
    print("Error: ratings.csv or movies.csv not found.")
except Exception as e:
    print("Error:", e)