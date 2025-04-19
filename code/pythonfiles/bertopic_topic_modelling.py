import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load and prepare data
df = pd.read_csv("llama2_ticket_summaries_sample.csv")
df["Summary"] = df["Summary"].fillna("")
df["Request Category"] = df["Request Category"].fillna("Unknown")
df["input_text"] = df["Request Category"] + " | " + df["Summary"]
docs = df["input_text"].tolist()

# Step 2: Generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Step 3: Create and fit BERTopic model
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
topic_model = BERTopic(embedding_model=None, vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)

# Step 4: Assign topics back to the main DataFrame
df["Topic"] = topics
df["Topic_Probability"] = probs
df.to_csv("bert_topic_clustered_tickets.csv", index=False)

# Step 5: Save topic summary info
topic_info = topic_model.get_topic_info()
topic_info.to_csv("bert_topic_info.csv", index=False)
print("✅ Topic info saved to bert_topic_info.csv")

# Step 6: Save the model
topic_model.save("bert_topic_model")
print("✅ BERTopic complete — model saved.")

# Step 7: Create clean top-8 document CSV for each topic
# Step 7: Create clean top-8 document CSV with top keywords and cleaned labels
top_n = 8
rep_rows = []

for topic in df["Topic"].unique():
    topic_df = df[df["Topic"] == topic]
    top_docs = topic_df["input_text"].head(top_n).tolist()

    # Extract top 8 keywords for the topic
    if topic != -1:
        top_words = topic_model.get_topic(topic)
        keywords = [kw for kw, _ in top_words[:8]]
        label = keywords[0]  # Most dominant word
    else:
        keywords = []
        label = "Outlier / Noise"

    row = {
        "Topic": topic,
        "Topic_Label": label,
        "Topic_Keywords": ", ".join(keywords)
    }

    # Add representative documents
    for i in range(top_n):
        row[f"Doc_{i+1}"] = top_docs[i] if i < len(top_docs) else ""

    rep_rows.append(row)

# Save as final CSV for LLM subtopic generation
rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv("bert_topic_top8_docs_with_keywords.csv", index=False)
print("✅ Top 8 docs with topic keywords saved to bert_topic_top8_docs_with_keywords.csv")
