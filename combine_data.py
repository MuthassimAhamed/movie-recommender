import pandas as pd

movies = pd.read_csv("movies_tmdb.csv")
series = pd.read_csv("web_series.csv")

# Combine both into one dataset
combined = pd.concat([movies, series], ignore_index=True)

# Drop rows with missing overview (we need text for embeddings later)
combined = combined.dropna(subset=["overview"])

# Save the master file
combined.to_csv("master_dataset.csv", index=False)

print(f"Combined dataset saved with {len(combined)} total entries")
print(combined["type"].value_counts())
print(combined.head())