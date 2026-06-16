from surprise import SVD, Dataset, Reader
import pandas as pd
from collections import defaultdict

# ✅ FIXED PATH
df = pd.read_csv(r"data/processed/ratings.csv")

# Rating scale
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))

# Load dataset
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train model
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Predictions
testset = trainset.build_anti_testset()
predictions = model.test(testset)

# Convert to DataFrame
pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'true_rating', 'pred_rating', 'details'])

print(pred_df.head())

# Top-N recommendations
top_n = defaultdict(list)

for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est))

for uid in top_n:
    top_n[uid].sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = top_n[uid][:5]

print("\nTop 5 recommendations for user 1:")
print(top_n[1])