import pandas as pd
import json
import time
from tqdm import tqdm
from llama_cpp import Llama

# Load topic prompts
df = pd.read_csv("bert_topic_top8_docs_with_keywords.csv")

# Load LLaMA model
llm = Llama(
    model_path="/scratch/zt1/project/cafb-chl/user/afyildiz/llama3-gguf/llama2-7b-chat.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=32,
    verbose=True
)

results = []
MAX_RETRIES = 2

# Inference loop
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating subtopics"):
    topic_id = int(row["Topic"])
    keywords = row["Topic_Keywords"]
    docs = [row.get(f"Doc_{i+1}", "") for i in range(8) if pd.notna(row.get(f"Doc_{i+1}"))]
    summaries = "\n- " + "\n- ".join([doc.strip() for doc in docs])

    # Strict prompt with enforced format
    prompt = f"""[INST]<<SYS>>
You are a customer support analyst.

Your task is to:
1. Read the top keywords and ticket summaries.
2. Generate **at least 1 and at most 5** distinct subtopics that represent the types of customer requests in this topic cluster.

✅ Each subtopic must:
- Be a short, reusable noun phrase (2-5 words)
- Capture a **generalized request type**, not a one-off case
- Be semantically distinct from other subtopics

❌ Do not:
- Include item codes, order numbers, or specific dates
- Repeat or rephrase the same meaning
- Add filler just to reach 5 subtopics
⛔ DO NOT return an empty list. Always return at least one valid subtopic.
💡 You must output ONLY the JSON block. Do not explain anything, do not say "Here is your JSON", do not wrap it in text.

🧠 Hint: If many subtopics seem too similar, **combine** them into one broader phrase.

✅ Output ONLY a valid JSON object in this exact format:
{{
  "id": {topic_id},
  "subtopics": [
    "short noun phrase 1",
    "short noun phrase 2"
  ]
}}
<</SYS>>

Top Keywords: {keywords}

Ticket Summaries:
{summaries}
[/INST]>"""


    # LLM with retry
    for attempt in range(MAX_RETRIES + 1):
        out = llm(prompt, max_tokens=256, temperature=0.4, stop=["</s>"])
        try:
            parsed = json.loads(out["choices"][0]["text"].strip())
            assert "id" in parsed and "subtopics" in parsed
            results.append(parsed)
            break
        except Exception:
            print(f"⚠️ Parsing failed for topic {topic_id}. Raw output:\n{out['choices'][0]['text'].strip()}")

            if attempt == MAX_RETRIES:
                results.append({
                    "id": topic_id,
                    "subtopics": []
                })
            else:
                time.sleep(1.5)

# Save to JSON
with open("bert_topic_llm_subtopics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Subtopics saved to 'bert_topic_llm_subtopics.json'")