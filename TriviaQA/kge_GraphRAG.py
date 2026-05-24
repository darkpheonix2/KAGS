import pandas as pd
import ast
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from functools import lru_cache
from typing import List, Tuple
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"


### ========== EMBEDDING FUNCTIONS ========== ###

@lru_cache(maxsize=10000)
def embed_text_cached(text: str) -> list[float]:
    """Cache embeddings to avoid recomputing identical texts"""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    emb = embedder.encode(text)
    return emb.tolist()


### ========== WEAVIATE SCHEMA SETUP ========== ###

def ensure_weaviate_schema(weaviate_client, triplet_database_name="Triplets_Phi4_GraphRAG"):
    """Ensure the Weaviate collection exists"""
    try:
        if weaviate_client.collections.exists(triplet_database_name):
            print(f"✅ Collection '{triplet_database_name}' already exists")
            return
        
        weaviate_client.collections.create(
            name=triplet_database_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="subject", data_type=DataType.TEXT),
                Property(name="predicate", data_type=DataType.TEXT),
                Property(name="object", data_type=DataType.TEXT),
                Property(name="triplet_id", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=128,
                max_connections=64
            )
        )
        print(f"✅ Collection '{triplet_database_name}' created successfully")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")


### ========== TRIPLET INGESTION ========== ###

def ingest_triplets_batch(weaviate_client, triplet_database_name, triplets_df, batch_size=100):
    """Ingest triplets into Weaviate in batches"""
    collection = weaviate_client.collections.get(triplet_database_name)
    triplet_ids = []
    failed_indices = []

    for start_idx in tqdm(range(0, len(triplets_df), batch_size)):
        batch = triplets_df.iloc[start_idx:start_idx + batch_size]
        batch_objects = []

        for idx, row in batch.iterrows():
            if pd.isna(row.subject) or pd.isna(row.predicate) or pd.isna(row.object):
                failed_indices.append(idx)
                continue

            # Embed triplet text (subject + predicate + object)
            text_for_embedding = f"{row.subject} {row.predicate} {row.object}"
            vec = embed_text_cached(text_for_embedding)

            triplet_id = str(uuid4())

            batch_objects.append({
                "properties": {
                    "subject": row.subject,
                    "predicate": row.predicate,
                    "object": row.object,
                    "triplet_id": triplet_id,
                },
                "uuid": triplet_id,
                "vector": vec
            })
            triplet_ids.append(triplet_id)

        # Batch insert
        if batch_objects:
            with collection.batch.dynamic() as batch_inserter:
                for obj in batch_objects:
                    batch_inserter.add_object(
                        properties=obj["properties"],
                        uuid=obj["uuid"],
                        vector=obj["vector"]
                    )

    print(f"✅ Inserted {len(triplet_ids)} triplets into Weaviate")
    if failed_indices:
        print(f"⚠️ Failed at indices: {failed_indices}")

    return triplet_ids, failed_indices


### ========== TRIPLET PARSING ========== ###

def parse_triplet_list(lst: List[str]) -> Tuple[str, str, str]:
    """Ensure the list has exactly 3 elements: [subject, predicate, object]"""
    if not isinstance(lst, list):
        raise ValueError(f"Expected list, got {type(lst)}")
    if len(lst) != 3:
        raise ValueError(f"Expected 3 elements, got {len(lst)} → {lst}")
    return tuple(lst)


### ========== MAIN SCRIPT ========== ###

if __name__ == "__main__":
    # Load your CSV
    triplet_df = pd.read_csv("Refined_triplets_GraphRAG_preprocessed.csv")

    triplet_database_name = "Triplets_GraphRAG"

    # Best practice: store your credentials in environment variables
    import sys
    from pathlib import Path
    _RAGS_ROOT = Path(__file__).resolve().parents[1]
    if str(_RAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_RAGS_ROOT))
    from db_config import get_weaviate_client

    weaviate_client = get_weaviate_client()

    print(weaviate_client.is_ready()) 
    ensure_weaviate_schema(weaviate_client, triplet_database_name)

    # Ingest triplets
    ingest_triplets_batch(weaviate_client, triplet_database_name, triplet_df, batch_size=200)

    print("🎉 All triplets ingested into Weaviate successfully!")
