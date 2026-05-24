import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import Auth
import time
from datetime import datetime
import os

import sys
from pathlib import Path
_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ======================
# SETUP
# ======================

WEAVIATE_DB_NAME = "Triplets_HybridRAG"

# Connect to Weaviate
from db_config import get_weaviate_client

weaviate_client = get_weaviate_client()
print("Weaviate connected:", weaviate_client.is_ready())

# ======================
# HELPER FUNCTIONS
# ======================

def embed_text(text: str) -> list[float]:
    """Generate embedding for a given text."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text).tolist()

def get_top_triplets_from_weaviate(query_text: str, top_k: int = 3):
    """
    Query Weaviate for top-k triplets matching the query text.
    Returns a list of formatted triplets (subject, predicate, object).
    """
    try:
        query_vec = embed_text(query_text)
        collection = weaviate_client.collections.get(WEAVIATE_DB_NAME)
        response = collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_properties=["subject", "predicate", "object"]
        )

        triplets = []
        for obj in response.objects:
            props = obj.properties
            triplet = f"({props.get('subject')}, {props.get('predicate')}, {props.get('object')})"
            triplets.append(triplet)
        
        return triplets

    except Exception as e:
        print(f"Error retrieving triplets for query '{query_text[:50]}...': {e}")
        return []

# ======================
# MAIN PIPELINE
# ======================

def process_dataframe_with_triplets(
    df: pd.DataFrame,
    question_column: str = "question",
    top_k: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    For each question in the dataframe, retrieve top-k triplets from Weaviate.
    Adds a new column 'retrieved_triplets' with the results.
    """
    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataframe")

    df = df.copy()
    all_triplets = []
    total = len(df)

    if verbose:
        print(f"Processing {total} questions...\n")

    start = time.time()
    for idx, question in enumerate(df[question_column], 1):
        triplets = get_top_triplets_from_weaviate(question, top_k=top_k)
        all_triplets.append(triplets)

        if verbose and idx % 5 == 0:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")

    df["retrieved_triplets"] = all_triplets

    if verbose:
        print(f"\n✓ Completed in {time.time() - start:.1f}s")

    return df

# ======================
# EXAMPLE USAGE
# ======================

if __name__ == "__main__":
    # Example: Load your input file
    df = pd.read_csv('Ouput_Data/VectorRAG_retrieved_k1.csv')

    # Process the dataframe
    result_df = process_dataframe_with_triplets(
        df=df,
        question_column="question",
        top_k=3,
        verbose=True
    )

    # Save results
    result_df.to_csv('Triplet_Retrieval_Output.csv', index=False)
    print("Saved to Triplet_Retrieval_Output.csv")

    # Close connection
    weaviate_client.close()
