
import os
import pandas as pd
from huggingface_hub import InferenceClient
import ast

import sys
from pathlib import Path
_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

# ---- Your cleaning function (slightly modified to handle tuple-style triplets) ----
def union_triplets(triplet_lists):
    """
    Takes a list of triplet lists and returns the union (unique triplets).
    - Skips self-referential triplets (where head == tail)
    - Treats symmetric triplets as identical (A->B same as B->A regardless of type)
    - Treats triplets with same head/tail but different types as identical
    """
    seen_triplets = set()
    unique_triplets = []

    for triplet_list in triplet_lists:
        for triplet in triplet_list:
            try:
                head, head_type, relation, tail, tail_type = triplet
            except Exception:
                continue  # skip malformed

            if head == tail:
                continue

            normalized = tuple(sorted([head, tail, head_type, tail_type]))

            if normalized not in seen_triplets:
                seen_triplets.add(normalized)
                unique_triplets.append(triplet)

    return unique_triplets


# ---- Function to query model for one chunk ----
def extract_triplets_from_chunk(chunk, client, system_prompt, user_prompt_template):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(context=chunk)},
            ],
        )
        output = completion.choices[0].message.content.strip()

        # Ensure proper list structure
        lines = output.split("\n")
        triplets = []
        for line in lines:
            line = line.strip().strip(",")  # clean trailing commas
            if not line:
                continue
            try:
                triplet = ast.literal_eval(line)
                if isinstance(triplet, tuple):
                    triplets.append(triplet)
            except Exception:
                continue
        return triplets
    except Exception as e:
        print(f"Error in chunk: {e}")
        return []


# ---- Apply to your dataframe ----
def process_dataframe(df, client, system_prompt, user_prompt_template):
    triplets_by_chunk_all = []
    triplets_union_all = []

    for chunks in df['chunks_from_preprocessed']:
        # Split by <c>
        chunk_list = [c.strip() for c in chunks.split("<c>") if c.strip()]

        # Extract per chunk
        per_chunk_triplets = [
            extract_triplets_from_chunk(chunk, client, system_prompt, user_prompt_template)
            for chunk in chunk_list
        ]

        # Clean union
        union_triplets_list = union_triplets(per_chunk_triplets)

        triplets_by_chunk_all.append(per_chunk_triplets)
        triplets_union_all.append(union_triplets_list)

    df['triplets_by_chunk_gpt_oss_120b'] = triplets_by_chunk_all
    df['triplets_union_gpt_oss_120b'] = triplets_union_all
    return df


if __name__=='__main__':
    
    from db_config import setup_hf_token
    setup_hf_token()
    df = pd.read_csv('Input_Data/preprocessed/HotpotQA_testing_dataset_preprocessed_with_triplets_rebel.csv')
    df = df.sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)

    # Allowed entity categories
    hotpot_qa_entity_categories = [
        "PERSON",
        "ORGANIZATION", 
        "LOCATION",
        "DATE_TIME",
        "CREATIVE_WORK",
        "EVENT",
        "PRODUCT",
        "NATIONALITY_ETHNICITY",
        "PROFESSION_ROLE",
        "AWARD_HONOR",
        "SPORT_GAME",
        "LANGUAGE",
        "MONEY",           
        "QUANTITY",        
        "FACILITY",        
        "VEHICLE",         
        "ANIMAL",          
        "FOOD",            
        "MEDICAL",         
        "ACADEMIC",        
        "RELIGION",        
        "MISCELLANEOUS"    
    ]

    # System prompt
    system_prompt = f"""
    You are an expert at extracting structured knowledge in the form of triplets and metadata. 
    Given a context, your task is to identify entities and their relationships, and represent them strictly as triplets. 

    Each triplet must follow this format:
    ('head', 'head_type', 'relationship', 'tail', 'tail_type')

    - 'head' = the source entity
    - 'head_type' = type/class/category of the head (must be one of the following: {", ".join(hotpot_qa_entity_categories)})
    - 'relationship' = the relation between the entities
    - 'tail' = the target entity
    - 'tail_type' = type/class/category of the tail (must be one of the following: {", ".join(hotpot_qa_entity_categories)})

    Only output valid triplets. Do not include explanations, extra text, or commentary.
    """


    client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

    user_prompt_template = """
Here is the context. Extract all valid triplets from it:

{context}
"""

    df = process_dataframe(df, client, system_prompt, user_prompt_template)


