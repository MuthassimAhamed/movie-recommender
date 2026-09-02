from ratings_store import save_rating, load_ratings

# Simulate a user rating a couple of titles
save_rating(user_id="user1", title="Se7en", rating=5)
save_rating(user_id="user1", title="Big Baby", rating=2)
save_rating(user_id="user1", title="Se7en", rating=4)  # should update, not duplicate

print(load_ratings())