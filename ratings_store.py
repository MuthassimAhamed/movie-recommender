import pandas as pd
import os

RATINGS_FILE = "user_ratings.csv"

def load_ratings():
    if os.path.exists(RATINGS_FILE):
        return pd.read_csv(RATINGS_FILE)
    else:
        return pd.DataFrame(columns=["user_id", "title", "rating", "review"])

def save_rating(user_id, title, rating, review=""):
    ratings = load_ratings()

    existing = (ratings["user_id"] == user_id) & (ratings["title"] == title)
    if existing.any():
        ratings.loc[existing, "rating"] = rating
        ratings.loc[existing, "review"] = review
    else:
        new_row = pd.DataFrame([{"user_id": user_id, "title": title, "rating": rating, "review": review}])
        ratings = pd.concat([ratings, new_row], ignore_index=True)

    ratings.to_csv(RATINGS_FILE, index=False)
    return ratings