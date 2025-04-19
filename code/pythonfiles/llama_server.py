import json
import re
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import logging
# At the top
from hashlib import sha256
### PRIORITY
def compute_weighted_score(text, keywords):
    text = str(text).lower()
    return sum(len(re.findall(rf'\b{re.escape(word)}\b', text)) * weight
               for word, weight in keywords.items())

# --- Compute dynamic score from topic (request category) ---
def compute_dynamic_score(topic_id, priority_map, value_map):
    label = priority_map.get(topic_id)
    return value_map.get(label, 0)

# --- Hybrid scoring logic ---
def hybrid_priority_score(topic_id, description,  alpha=1.0, beta=1.0):
    weighted_score = compute_weighted_score(description, weighted_keywords)
    dynamic_score = compute_dynamic_score(topic_id, category_priority_map, priority_value_map)

    return (alpha * weighted_score) + (beta * dynamic_score)
weighted_keywords = {
    "urgent": 3,
    "asap": 3,
    "immediately": 3,
    "critical": 3,
    "emergency": 3,
    "today": 3,
    "death": 3,
    "cancel": 2,
    "add": 2,
    "change": 2,
    "reschedule": 2,
    "edit": 2,
    "update": 2,
    "delete": 2,
    "missing": 2,
    "delay": 1,
    "tomorrow": 1,
    "now": 1,
    "problem": 1,
    "help": 1,
    "remove": 1,
    "assistance": 1,
    "support": 1,
    "issue": 1,
    "expire": 1,
    "move": 1
}
category_priority_map = {
    -1: "High",     # -1_items_order_order items_items customer
    0: "High",      # 0_cancelation_delivery cancelation_cancelation customer_cancel
    1: "Medium",      # 1_partnerlink_administration_agency administration_agency
    2: "High",      # 2_missing_delivery pickup_missing item_pickup missing
    3: "Low",       # 3_menu_menu general_availability_inventory availability
    4: "High",      # 4_delivery edit_items customer_order items_edit order
    5: "Low",       # 5_pickup_pallet pickup_pickup pallet_pickup customer
    6: "Medium",      # 6_produce_request customer_produce request_delivery produce
    7: "Low",       # 7_grants_billing grants_billing_grant
    8: "Medium",    # 8_pickup general_delivery pickup_eta_status customer
    9: "High",      # 9_window customer_window_datetime_datetime change
    10: "High",     # 10_items_add items_items order_edit
    11: "High",     # 11_reschedule_change_change reschedule_reschedule customer
    12: "Medium",     # 12_vegetables_request customer_produce request_delivery produce
    13: "High",     # 13_edit_edit order_delivery edit_items customer
    14: "Medium",     # 14_cafb staff_administration connection_connection_connection cafb
    15: "High",     # 15_remove_wants remove_items_items customer
    16: "High",     # 16_reschedule_delivery datatime_datatime_datatime change
    17: "Medium",   # 17_delivery general_questions_general questions_general
    18: "High",     # 18_reschedule_weather_change reschedule_delivery datatime
    19: "High",     # 19_cancel order_wants cancel_cancel_cancelation customer
    20: "Low",      # 20_best_expiration_expiration date_product best
    21: "Low",      # 21_delivery report_report issue_issue_report
    22: "Low"       # 22_food_delivery general_general_questions
}
# --- Priority levels to numeric scores ---
priority_value_map = {
    "Low": 0,
    "Medium": 2,
    "High": 3
}
def assign_hybrid_priority(score):
    if score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"
### PRIORITY
session_store = {}
def get_session_id(history):
    if not history:
        return "session_default"

    # Always use the FIRST user message only
    for msg in history:
        if msg.get("speaker", "").lower() == "user":
            base = msg["message"].strip().lower()
            return sha256(base.encode()).hexdigest()

    return "session_default"  # fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()  # ✅ Only this one!

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Preload everything ONCE === #
with open("bert_topic_llm_subtopics.json", "r") as f:
    subtopic_map = {entry["id"]: entry["subtopics"] for entry in json.load(f)}

with open("response_jsons.json", "r") as f:
    response_list = json.load(f)

class CustomEmbedder(BaseEmbedder):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    def embed(self, documents, verbose=False):
        return self.model.encode(documents, show_progress_bar=verbose)

sbert_path = "/scratch/zt1/project/cafb-chl/user/afyildiz/all-MiniLM-L6-v2-local"
embedding_model = CustomEmbedder(sbert_path)
topic_model = BERTopic.load("bert_topic_model")
topic_model.embedding_model = embedding_model
sbert = SentenceTransformer(sbert_path)

llm = Llama(
    model_path="/scratch/zt1/project/cafb-chl/user/afyildiz/llama3-gguf/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=32,
    verbose=False
)


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  # [{"speaker": "User", "message": "..."}]

def lookup_matching_schema(cluster, sorted_subtopics):
    for candidate_subtopic, candidate_score in sorted_subtopics:
        for entry in response_list:
            if (entry.get("cluster") == cluster and 
                entry.get("subtopic", "").strip().lower() == candidate_subtopic.strip().lower()):
                logger.info(f"✅ Matched subtopic: '{candidate_subtopic}' (score: {candidate_score:.3f})")
                return entry
    logger.warning("⚠️ No matching schema found for any subtopic.")
    return {"subtopic": "N/A", "schema": {}, "cluster": cluster}

def build_prompt(matching_json, conversation_history):
    user_fields = [k for k, v in matching_json.get("schema", {}).items() if v == "user"]
    conv_str = "\n".join(f"{m['speaker']}: {m['message']}" for m in conversation_history)

    return f"""[INST] <<SYS>>
You are a helpful customer support assistant.

You must identify and complete the following required fields based on the conversation: {user_fields}

Instructions:
- Read the full conversation history.
- If any required fields are missing, return a JSON with "response" as a polite question asking for the missing info and "solved": false.
- If all required fields are present or implied, return a JSON with "response" as a polite thank you and goodbye sentence and giving required fields information to user, and "solved": true.
- Do not explain your reasoning or return any text outside of the JSON.
- Your full response must be a single line JSON object like: {{ "response": "Thank you, your order has been updated.", "solved": true }}
- Never return markdown, headings, quotes, or any explanation.
- Only reply with a single-line JSON. No formatting.

You are now responding directly to the user.
<</SYS>>
Conversation:
{conv_str}
[/INST]"""

@app.post("/chat")
def chat(req: ChatRequest):
    input_text = req.message
    conversation_history = req.history + [{"speaker": "User", "message": input_text}]
    logger.info(f"📩 Incoming user message:\n {input_text}")

    # Create session ID based on full history (or just use len(req.history) == 0 for simpler cases)
    session_id = get_session_id(conversation_history)
    logger.info(f"🔐 Session ID: {session_id}")

    # Check if we already have matching_json cached
    matching_json = session_store.get(session_id)

    if matching_json is None:
        logger.info("🧠 No cached subtopic — running topic model")

        topic_ids, probs = topic_model.transform([input_text])
        topic_id, topic_prob = topic_ids[0], probs[0]
        max_prob = max(probs)
        logger.info(f"🔎 Max topic probability: {max_prob:.3f} (Topic ID: {topic_id})")
        if max_prob < 0.1:
            logger.info("Could not find a suitable subtopic")
            return {
                "response": "I have not encountered this issue before. A customer service representative will attend to your request shortly.",
                "solved": False
            }

        subtopics = subtopic_map.get(topic_id, [])
        if not subtopics:
            matching_json = {"subtopic": "N/A", "schema": {}, "cluster": topic_id}
        else:
            ticket_emb = sbert.encode(input_text, convert_to_tensor=True)
            subtopic_embs = sbert.encode(subtopics, convert_to_tensor=True)
            scores = util.cos_sim(ticket_emb, subtopic_embs)[0]
            subtopic_scores = sorted(zip(subtopics, scores.tolist()), key=lambda x: x[1], reverse=True)
            # Optional: add a second outlier filter here
            if subtopic_scores[0][1] < 0.3:
                 return {
            "response": "I'm not sure how to help with that request yet. A customer service representative will follow up shortly.",
            "solved": False
             }
            matching_json = lookup_matching_schema(topic_id, subtopic_scores)

        # Cache it for future requests
        session_store[session_id] = matching_json
        logger.info(f"🧠 Cached matching_json for session: {session_id}")
    else:
        logger.info(f"🔁 Reusing cached subtopic for session: {session_id}")

    prompt = build_prompt(matching_json, conversation_history)
    logger.info(f"Here is the matching json related to ticket description {matching_json}")
    out = llm(prompt, max_tokens=150, temperature=0.3, top_k=40, top_p=0.9, stop=["[/INST]"])
    raw_output = out.get("choices", [{}])[0].get("text", "").strip()

    json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
    if not json_match:
        return {"response": "Sorry, I'm having trouble processing your request.", "solved": False}

    try:
        parsed = json.loads(json_match.group())
        priority_score = hybrid_priority_score(matching_json["cluster"], input_text)
        priority_label = assign_hybrid_priority(priority_score)
        parsed["priority"] = priority_label
        if parsed.get("solved"):
          if session_id in session_store:
            del session_store[session_id]
            logger.info(f"🧹 Cleared session_store for session: {session_id}")
        return parsed
    except json.JSONDecodeError:
        return {"response": "Sorry, I couldn't understand your request. A representative will contact you shortly.", "solved": False}