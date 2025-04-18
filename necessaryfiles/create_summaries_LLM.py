from llama_cpp import Llama
import json
import pandas as pd
from tqdm import tqdm

# Load your data
with open("data_bertopic.json", "r", encoding="utf-8") as f:
    raw_tickets = json.load(f)

filtered_tickets = [
    t for t in raw_tickets if "Description" in t and str(t["Description"]).strip()
]

# Load the converted GGUF model (Llama 2 7B Chat in GGUF format)
llm = Llama(
    model_path="/scratch/zt1/project/cafb-chl/user/afyildiz/llama3-gguf/llama2-7b-chat.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=32,
    verbose=True
)

def summarize_description(description: str) -> str:
    system_prompt = (
        "You are a system that extracts the customer's main actionable request "
        "from food-related support tickets. Do not include politeness. "
        "Only return the core request in 1 short sentence. "
        "If there is no actionable request and there is no problem stated, only then respond exactly: None"
    )

    examples = """
Ticket: The apples delivered were spoiled. Can I get a refund?
Summary: Customer requests a refund for spoiled apples.

Ticket: Please add 5 cases of milk to my order.
Summary: Customer wants to add 5 cases of milk to their order.

Ticket: This order had the wrong date scheduled by mistake
Summary: Customer wants to reschedule the order date

Ticket: my orders were wiped out again
Summary: Customers order were wiped out

Ticket: Thank you for your help.
Summary: None

Ticket: Just wanted to say hello.
Summary: None
"""

    prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{examples.strip()}\n\n"
        f"Ticket: {str(description).strip()}\nSummary:"
    )

    output = llm(
        prompt,
        max_tokens=80,
        temperature=0.2,
        top_k=40,
        stop=["</s>", "\nTicket:", "\nSummary:"]
    )

    response = output["choices"][0]["text"].strip()

    if not response or response.lower().startswith(("of course", "sure", "i can", "here")):
        return "None"

    if response.lower() == "none":
        return "None"

    # Extra cleanup to remove double summaries if hallucinated
    response = response.split("Ticket:")[0].split("Summary:")[-1].strip()
    return response

# ✅ Minimal change: skip empty/whitespace summaries
summaries = []
for ticket in tqdm(filtered_tickets):
    desc = ticket.get("Description", "")
    summary = summarize_description(desc)

    if not summary.strip():  # ⛔ remove only truly empty summaries
        continue

    summaries.append({
        "Description": desc,
        "Summary": summary,
    })

# Save the summaries to CSV
df = pd.DataFrame(summaries)
df.to_csv("llama2_ticket_summaries_sample.csv", index=False)
print("✅ Sample saved to llama2_ticket_summaries_sample.csv")