import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load data and embeddings
df = pd.read_csv("master_dataset.csv")
embeddings = np.load("embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_n=5):
    query_embedding = model.encode([query])
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = similarities / norms
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["title", "type", "overview"]]

def ai_recommend(query):
    candidates = search(query, top_n=5)
    candidate_titles = candidates["title"].tolist()

    candidates_text = "\n".join([
        f"- {row['title']} ({row['type']}): {row['overview'][:150]}"
        for _, row in candidates.iterrows()
    ])

    prompt = f"""You are a recommendation assistant. A user wants: "{query}"

Below is a numbered list of candidates. You must ONLY recommend from this exact list. Do not invent, add, or mention ANY title that is not in this list, even if you know of a similar one.

Candidates:
{candidates_text}

Pick the best 2-3 matches ONLY from the list above. Use the titles exactly as written. Explain briefly why each fits the user's request. Be conversational and friendly."""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}  # lower = more focused, less "creative" hallucination
    )

    result = response["message"]["content"]

    # Safety check: verify the AI didn't mention any title outside our candidates
    mentioned_valid = [title for title in candidate_titles if title.lower() in result.lower()]
    if not mentioned_valid:
        print("⚠️ Warning: AI response didn't clearly reference any known candidate titles.\n")

    return result, candidate_titles

# Test it
query = "something mysterious with a slow burn plot"
print("Thinking...\n")
result, valid_titles = ai_recommend(query)
print(result)
print("\n--- Valid candidate titles were:", valid_titles)