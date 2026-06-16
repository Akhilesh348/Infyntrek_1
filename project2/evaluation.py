from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Load dataset
df = pd.read_csv(r"data/processed/ratings.csv")

# Define rating scale
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))

# Load data into Surprise format
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Print results
print("\n📊 Model Evaluation Results")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")