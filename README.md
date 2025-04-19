# 🧠 CAFB Ticket Intelligence Pipeline

This repository contains a complete pipeline for analyzing, classifying, and structuring customer support tickets for the **Capital Area Food Bank** using advanced NLP techniques. The project combines **BERTopic** clustering with **LLaMA-based LLMs** to produce summaries, topics, subtopics, and required schemas for automated assistant systems.

---

## 🚀 Pipeline Overview

1. **Summarize Ticket Descriptions** → `create_summaries_LLM.py` / `create_summaries_LLM-1.py`
2. **Topic Clustering via BERTopic** → `bertopic_topic_modelling.py`
3. **Generate Subtopics per Cluster** → `create_subtopics_LLM.py`
4. **Match Tickets to Subtopics** → `ticket_to_subtopic.py`
5. **Enrich Ticket Summaries with Metadata** → `merge_dataset.py`
6. **Generate Key Field Schemas per Subtopic** → `response_key_value_LLM.py`
7. **Serve Chat API with Priority Scoring & Field Extraction** → `llama_server.py`

---

## 📂 Scripts and Their Purpose

### `create_summaries_LLM.py` / `create_summaries_LLM-1.py`
Extracts concise actionable summaries from raw ticket descriptions using a local **LLaMA 2** model.  
📄 **Output**: `llama2_ticket_summaries_sample.csv`

### `bertopic_topic_modelling.py`
Clusters ticket summaries into topics using **BERTopic**.  
📄 **Output**:
- `bert_topic_model`
- `bert_topic_info.csv`
- `bert_topic_clustered_tickets.csv`
- `bert_topic_top8_docs_with_keywords.csv`

### `create_subtopics_LLM.py`
Generates reusable and distinct subtopics per topic using LLaMA and BERTopic results.  
📄 **Output**: `bert_topic_llm_subtopics.json`

### `ticket_to_subtopic.py`
Assigns the most relevant subtopic to each ticket using cosine similarity.  
📄 **Output**: `tickets_with_subtopics.csv`

### `merge_dataset.py`
Enriches ticket summaries with cause of issue, request category, and comments.  
📄 **Output**: Cleaned and merged `llama2_ticket_summaries_sample.csv`

### `response_key_value_LLM.py`
Generates required field schemas for each subtopic using real ticket examples.  
📄 **Output**: `response_jsons.json`

### `llama_server.py`
FastAPI backend that:
- Classifies new tickets into topics/subtopics
- Extracts structured information
- Calculates hybrid priority score  
📌 Uses: `bert_topic_model`, `bert_topic_llm_subtopics.json`, `response_jsons.json`, SBERT & LLaMA

---

## 📁 Key Output Files

| File | Description |
|------|-------------|
| `llama2_ticket_summaries_sample.csv` | Actionable summaries per ticket |
| `bert_topic_clustered_tickets.csv` | Ticket-topic assignments |
| `bert_topic_info.csv` | Topic frequency overview |
| `bert_topic_top8_docs_with_keywords.csv` | Keywords & examples for each topic |
| `bert_topic_llm_subtopics.json` | Subtopics per topic |
| `tickets_with_subtopics.csv` | Final enriched ticket dataset |
| `response_jsons.json` | Field extraction schemas per subtopic |

---

## 🧰 Tech Stack

- **LLMs**: Meta LLaMA 2 & 3 (GGUF format via `llama.cpp`)
- **Embeddings**: SBERT (`all-MiniLM-L6-v2`)
- **Topic Modeling**: BERTopic
- **API**: FastAPI
- **Data**: CSV & JSON

---

## 🤖 How Our Chatbot Works (and What Sets It Apart)

Our pipeline follows a systematic flow:

1. **Preprocessing** the raw ticket data.
2. **Generating summaries** using LLMs from ticket descriptions.
3. **Clustering tickets** with BERTopic into topics.
4. **Generating subtopics** for each topic cluster using an LLM.
5. **Creating response schemas** (`response_jsons`) that define the required fields to solve each subtopic.
6. **Answering users** using either an LLM-powered assistant or a lightweight rules-based system (e.g., regex).

### ✨ What Makes It Powerful

- **Context-Aware Schema Extraction**:  
  Our model learns from real ticket solutions (comments) to determine what information is *actually* needed from the user or system. This allows the assistant to dynamically tailor its questions based on historical solutions, skipping unnecessary questions.

- **Hidden Pattern Discovery**:  
  For example, some rescheduling requests might not require a specific date—especially if the institution (like a known delivery client) always expects Friday delivery. The model picks up on this nuance and avoids asking for unnecessary inputs. These behavioral patterns form specific topic-subtopic pairs with customized required fields.

- **Adaptability & Growth**:  
  If a new type of request doesn't match existing topic-subtopic pairs (i.e., has low classification probability, e.g., < 0.25), it's routed to a human agent and added to an **outlier pool**. When enough outliers are collected, the pipeline is retrained and updated—automatically learning how to handle the new requests by generating new schemas and summaries.

---

## 📈 Scalability and Tuning Notes

This system is designed to scale with complete, high-quality data. It will benefit from:
- More labeled examples
- Careful tuning of prompts, model temperature, and classification thresholds
- Occasional retraining to keep up with evolving support request patterns

---

## 📌 License

MIT License. See `LICENSE` for more details.

---

## ✨ Acknowledgements

Special thanks to the Capital Area Food Bank and the open-source communities behind `sentence-transformers`, `BERTopic`, and `llama.cpp`.
