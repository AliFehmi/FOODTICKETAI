import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("bert_topic_clustered_tickets.csv")  # Replace with your filename

# Load the subtopics JSON
with open("bert_topic_llm_subtopics.json", "r") as f:
    subtopic_map = {entry["id"]: entry["subtopics"] for entry in json.load(f)}

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Output list
results = []

# Iterate over each ticket
for i, row in df.iterrows():
    topic = row["Topic"]
    summary = row["Summary"]
    
    subtopics = subtopic_map.get(topic, [])
    if not subtopics or not isinstance(summary, str) or summary.strip() == "":
        results.append("None")
        continue

    # Create embeddings
    embeddings = model.encode([summary] + subtopics)
    ticket_emb = embeddings[0].reshape(1, -1)
    subtopic_embs = embeddings[1:]

    # Compute cosine similarities
    sims = cosine_similarity(ticket_emb, subtopic_embs)[0]
    best_idx = sims.argmax()
    best_subtopic = subtopics[best_idx]

    results.append(best_subtopic)

# Add to DataFrame and save
df["Matched_Subtopic"] = results
df.to_csv("tickets_with_subtopics.csv", index=False)
print("✅ Saved to 'tickets_with_subtopics.csv'")
