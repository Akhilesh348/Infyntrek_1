import os
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Get project root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to ratings.csv
ratings_path = os.path.join(base_dir, "data", "processed", "ratings.csv")

# Load ratings dataset
ratings = pd.read_csv(ratings_path)

print("Dataset loaded successfully!")
print(ratings.head())

# Prepare data for Surprise
reader = Reader(rating_scale=(0, 1))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# Split data
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Train SVD model
model = SVD()

print("\nTraining SVD model...")
model.fit(trainset)

# Test model
predictions = model.test(testset)

# Evaluation
print("\nModel Evaluation:")
accuracy.rmse(predictions)
accuracy.mae(predictions)

# Sample prediction
user_id = ratings['userId'].iloc[0]
movie_id = ratings['movieId'].iloc[0]

prediction = model.predict(user_id, movie_id)

print("\nSample Prediction:")
print(f"User ID: {user_id}")
print(f"Movie ID: {movie_id}")
print(f"Predicted Rating: {prediction.est:.4f}")

# Top recommendations
print("\nTop 10 Recommendations:")

movies = ratings['movieId'].unique()
recommendations = []

for movie in movies:
    pred = model.predict(user_id, movie)
    recommendations.append((movie, pred.est))

recommendations.sort(key=lambda x: x[1], reverse=True)

for movie, score in recommendations[:10]:
    print(f"Movie ID: {movie} | Predicted Rating: {score:.4f}")

print("\nDay 5 SVD Model Completed Successfully!")