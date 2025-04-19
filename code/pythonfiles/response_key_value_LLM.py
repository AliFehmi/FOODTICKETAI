import pandas as pd
import json
import time
import regex as re
from tqdm import tqdm
from llama_cpp import Llama

# Load ticket data
df = pd.read_csv("tickets_with_subtopics.csv")
df["Matched_Subtopic"] = df["Matched_Subtopic"].fillna("unknown")
df["Comments"] = df["Comments"].fillna("")
df["Description"] = df["Description"].fillna("")

# We no longer restrict the dataset; process the whole CSV.
# (Comment out or remove the following two lines)
# first_12_subtopics = df["Matched_Subtopic"].unique()[:12]
# df = df[df["Matched_Subtopic"].isin(first_12_subtopics)]

# Load LLM model
llm = Llama(
    model_path="/scratch/zt1/project/cafb-chl/user/afyildiz/llama3-gguf/llama2-7b-chat.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=32,
    verbose=True
)

# Regex to capture the JSON output block
json_regex = re.compile(r'\{[\s\S]*\}')

# Parameters
N_EXAMPLES = 3
MAX_RETRIES = 2
MAX_WORDS_PER_EXAMPLE = 100
output = []

def extract_top_ticket_pairs(subtopic, topic, max_examples=3):
    """
    Extracts up to max_examples ticket pairs for a given subtopic and topic.
    Uses the 'Description' column as the ticket description and 'Comments' as the ticket comment.
    """
    group = df[(df["Matched_Subtopic"] == subtopic) & (df["Topic"] == topic)]
    # Use "Description" as the ticket description.
    if "Description" in df.columns:
        group = group.dropna(subset=["Description", "Comments"])
        group["Ticket"] = group["Description"].astype(str).str.strip()
        group["Comments"] = group["Comments"].astype(str).str.strip()
    else:
        group = group.dropna(subset=["Comments"])
        group["Ticket"] = ""
        
    pairs = []
    for _, row in group.sort_values("Topic_Probability", ascending=False).head(max_examples).iterrows():
        ticket_text = ' '.join(row["Ticket"].split()[:MAX_WORDS_PER_EXAMPLE])
        comment_text = ' '.join(row["Comments"].split()[:MAX_WORDS_PER_EXAMPLE])
        pair_str = f"Ticket Description: {ticket_text}\nTicket Comment: {comment_text}"
        pairs.append(pair_str)
    return pairs

# Process all (subtopic, topic) groups in the dataset
for (subtopic, topic), group in tqdm(df.groupby(["Matched_Subtopic", "Topic"]), desc="Generating must-field schemas"):
    ticket_pairs = extract_top_ticket_pairs(subtopic, topic, N_EXAMPLES)
    if not ticket_pairs:
        output.append({
            "subtopic": subtopic,
            "schema": {},
            "cluster": int(topic)
        })
        continue
    
    # Build examples block from ticket pairs
    example_block = "\n\n".join(f"- {pair}" for pair in ticket_pairs)
    
    # Updated prompt with an example format (curly braces are escaped).
    prompt = f"""[INST]<<SYS>>
You are a backend system designer for the Capital Area Food Bank's support ticket assistant.
Your task is to analyze the following ticket pairs (each consisting of a "Ticket Description" and a "Ticket Comment") related to the subtopic: "{subtopic}".
Determine the MUST-HAVE fields required to resolve this type of issue and, for each field, specify the expected source as either "user" or "system".
IMPORTANT:
- Output ONLY a single JSON object with exactly one key: "fields".
- The value of "fields" must be a flat JSON object where each key is a required field name and its value is either "user" or "system".
- Do NOT include any additional text, explanation, keys, or nested objects.
- DO NOT copy the following example into your output; it is provided solely as a format reference.
EXAMPLE FORMAT (for reference only):
{{
  "fields": {{
    "Order ID": "user",
    "Customer Name": "user",
    "Order Status": "system"
  }}
}}
Output exactly and only the JSON object.
<</SYS>>

Here are some real ticket pairs with their description and comment:
{example_block}

[REMINDER]: Only return the JSON object.
[/INST]>"""

    # Try up to MAX_RETRIES attempts to get a valid JSON output from the model.
    for attempt in range(MAX_RETRIES + 1):
        try:
            out = llm(prompt, max_tokens=512, temperature=0.4, stop=["</s>"])
            text = out["choices"][0]["text"].strip()
            
            match = json_regex.search(text)
            if not match:
                raise ValueError("No valid JSON object found.")
            
            json_block = match.group(0)
            json_block = re.sub(r'"\s*"', '", "', json_block)
            parsed = json.loads(json_block)
            fields = parsed.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("Expected 'fields' to be a dictionary mapping field names to 'user' or 'system'.")
            
            for key, value in fields.items():
                if value not in ["user", "system"]:
                    raise ValueError(f"Field '{key}' has invalid value '{value}'. Must be 'user' or 'system'.")
            
            output.append({
                "subtopic": subtopic,
                "schema": fields,
                "cluster": int(topic)
            })
            break
        except Exception as e:
            print(f"\n⚠️ Failed to parse subtopic '{subtopic}' (Attempt {attempt + 1}/{MAX_RETRIES + 1})")
            print(f"Reason: {e}")
            print(f"Raw output:\n{text if 'text' in locals() else '[empty]'}\n")
            if attempt == MAX_RETRIES:
                output.append({
                    "subtopic": subtopic,
                    "schema": {},
                    "cluster": int(topic)
                })
            else:
                time.sleep(1.5)

with open("response_jsons.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Saved MUST-field solution schemas to 'response_jsons.json'")