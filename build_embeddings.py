import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your combined dataset
df = pd.read_csv("master_dataset.csv")

# Load a small, efficient open-source embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Turn each overview into a numeric embedding
print("Creating embeddings... this may take a minute")
embeddings = model.encode(df["overview"].tolist(), show_progress_bar=True)

# Save embeddings as a numpy file, and the dataframe as-is (they'll match by row order)
np.save("embeddings.npy", embeddings)
df.to_csv("master_dataset.csv", index=False)  # unchanged, just re-saving for consistency

print(f"Done! Created {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}")