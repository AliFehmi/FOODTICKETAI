import pandas as pd
import json
from tqdm import tqdm

# Load summaries CSV
input_csv = "llama2_ticket_summaries_sample.csv"
df_summaries = pd.read_csv(input_csv)

# Normalize fields
df_summaries["Description"] = df_summaries["Description"].astype(str).str.strip()
df_summaries["Summary"] = df_summaries["Summary"].astype(str).str.strip()

# ✅ Remove rows where Summary is real NaN, or string "None", "nan", or empty
df_summaries = df_summaries[
    df_summaries["Summary"].notna() &
    (~df_summaries["Summary"].str.lower().isin(["none", "nan", ""]))
]


# Deduplicate on Description
df_summaries = df_summaries.drop_duplicates(subset="Description")

# Load JSON data
with open("data_bertopic.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)

# ✅ Filter and prepare enrichment data
enrichment_rows = []
valid_descriptions = set(df_summaries["Description"])

for item in tqdm(data_json, desc="Filtering valid enrichment entries"):
    desc = str(item.get("Description", "")).strip()
    cause = item.get("Custom field (Cause of issue)")
    category = item.get("Custom field (Request Category)")

    if (
        desc and
        desc in valid_descriptions and
        cause is not None and
        category is not None
    ):
        comments = " | ".join(
            str(v).strip()
            for k, v in item.items()
            if isinstance(k, str) and "Comment" in k and isinstance(v, str)
        )

        enrichment_rows.append({
            "Description": desc,
            "Cause of Issue": cause,
            "Request Category": category,
            "Comments": comments
        })

# Convert and deduplicate enrichment data
df_enrichment = pd.DataFrame(enrichment_rows)
df_enrichment = df_enrichment.drop_duplicates(subset="Description")

# ✅ Perform merge using inner join (matching descriptions only)
df_summaries_updated = pd.merge(
    df_summaries,
    df_enrichment,
    on="Description",
    how="inner"
)

# ✅ Remove any duplicate Descriptions after merge
df_summaries_updated = df_summaries_updated.drop_duplicates(subset="Description")

# ✅ Final safety filter to remove any NaN summaries
nan_rows = df_summaries_updated[df_summaries_updated["Summary"].isna()]
if not nan_rows.empty:
    print(f"❌ Found {len(nan_rows)} rows with NaN in Summary. They will be removed.")
    print(nan_rows[["Description", "Cause of Issue", "Request Category"]].head())

df_summaries_updated = df_summaries_updated[df_summaries_updated["Summary"].notna()]

# ✅ Save final cleaned dataset
df_summaries_updated.to_csv(input_csv, index=False)
print(f"✅ Final cleaned and deduplicated dataset saved to '{input_csv}'")
