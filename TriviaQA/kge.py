import pandas as pd
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import ast
import weaviate
from weaviate.classes.init import Auth
import weaviate
from neo4j import GraphDatabase
from weaviate.auth import Auth
from weaviate.classes.config import Property, DataType, Configure
from tqdm import tqdm
from functools import lru_cache
from weaviate.util import generate_uuid5
import os
import numpy as np

import sys
from pathlib import Path
_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

# Load model once globally
print("Loading embedding model...")
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

def clean_value(value):
    """Clean and validate a value, replacing NaN/None with empty string"""
    if pd.isna(value) or value is None:
        return ""
    value_str = str(value).strip()
    # Check for string representations of NaN
    if value_str.lower() in ['nan', 'none', 'null', '']:
        return ""
    return value_str

def ingest_triplets_batch(triplets_df, batch_size=100):
    """
    Ingest triplets in batches to reduce API calls
    """
    collection = weaviate_client.collections.get(triplet_database_name)
    triplet_ids = []
    failed_indices = []
    
    # Process in batches
    for start_idx in tqdm(range(0, len(triplets_df), batch_size)):
        batch = triplets_df.iloc[start_idx:start_idx + batch_size]
        
        # Prepare batch data
        batch_objects = []
        batch_triplet_ids = []
        
        for idx, row in batch.iterrows():
            # Clean all fields
            subject = clean_value(row.subject)
            predicate = clean_value(row.predicate)
            obj = clean_value(row.object)
            metadata = clean_value(row.metadata)
            
            # Validate required fields
            if not subject or not predicate or not obj:
                print(f"⚠️ Skipping row {idx} - missing required fields (subject/predicate/object)")
                failed_indices.append(idx)
                batch_triplet_ids.append(None)
                continue
            
            if not metadata:
                print(f"⚠️ Skipping row {idx} - empty metadata")
                failed_indices.append(idx)
                batch_triplet_ids.append(None)
                continue
            
            # Embed texts
            text_for_embedding = f"{subject} {predicate} {obj}"
            vec = embed_text(text_for_embedding)
            meta_vec = embed_text(metadata)
            
            if vec is None or meta_vec is None:
                print(f"⚠️ Skipping row {idx} - embedding failed")
                failed_indices.append(idx)
                batch_triplet_ids.append(None)
                continue
            
            triplet_id = str(uuid4())
            batch_triplet_ids.append(triplet_id)
            
            batch_objects.append({
                "properties": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "metadata": metadata,
                    "triplet_id": triplet_id,
                    "metadata_vector": meta_vec
                },
                "uuid": triplet_id,
                "vector": vec
            })
        
        # Batch insert to Weaviate
        if batch_objects:
            with collection.batch.dynamic() as batch_inserter:
                for obj in batch_objects:
                    batch_inserter.add_object(
                        properties=obj["properties"],
                        uuid=obj["uuid"],
                        vector=obj["vector"]
                    )
        
        # Batch insert to Neo4j
        ingest_neo4j_batch(batch, batch_triplet_ids)
        
        triplet_ids.extend(batch_triplet_ids)
    
    return triplet_ids, failed_indices

def ingest_neo4j_batch(batch_df, triplet_ids):
    """Batch insert to Neo4j with proper NaN handling"""
    cypher = """
    UNWIND $batch as row
    MERGE (sub:Entity {name: row.subject})
    MERGE (obj:Entity {name: row.object})
    MERGE (sub)-[rel:REL {
        predicate: row.predicate, 
        metadata: row.metadata, 
        triplet_id: row.triplet_id
    }]->(obj)
    """
    
    batch_data = []
    for (idx, row), tid in zip(batch_df.iterrows(), triplet_ids):
        if tid is not None:
            # Clean all values before inserting to Neo4j
            subject = clean_value(row.subject)
            obj = clean_value(row.object)
            predicate = clean_value(row.predicate)
            metadata = clean_value(row.metadata)
            
            # Only add if all required fields are valid
            if subject and obj and predicate and metadata:
                batch_data.append({
                    "subject": subject,
                    "object": obj,
                    "predicate": predicate,
                    "metadata": metadata,
                    "triplet_id": tid
                })
    
    if batch_data:
        with driver.session() as session:
            session.run(cypher, batch=batch_data)


def embed_text(text: str) -> list[float]:
    """Use the global embedder model"""
    try:
        if not text or not text.strip():
            return None
        emb = EMBEDDER.encode(text)
        return emb.tolist()
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

@lru_cache(maxsize=10000)
def embed_text_cached(text):
    """Cache embeddings to avoid re-computing identical texts"""
    return embed_text(text)


def ensure_weaviate_schema(triplet_database_name="Test_dataset_Triplet"):
    try:
        # Check if collection already exists
        if weaviate_client.collections.exists(triplet_database_name):
            print(f"Collection {triplet_database_name} already exists")
            return
        
        # Create collection using v4 API
        weaviate_client.collections.create(
            name=triplet_database_name,
            vectorizer_config=Configure.Vectorizer.none(),  # or configure your vectorizer
            properties=[
                Property(name="subject", data_type=DataType.TEXT),
                Property(name="predicate", data_type=DataType.TEXT),
                Property(name="object", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
                Property(name="triplet_id", data_type=DataType.TEXT),
            ],
            # Optional: configure HNSW index
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=128,
                max_connections=64
            )
        )
        print(f"Collection {triplet_database_name} created successfully")
    except Exception as e:
        print(f"Error creating collection: {e}")



def parse_triplet_list(lst: List[str]) -> Tuple[str, str, str, str]:
    """
    Safely parse a triplet that is already a list of 4 strings.
    Example: ["Arthur Turner", "was born on", "1 April 1909", "Some metadata"]
    """
    if not isinstance(lst, list) or len(lst) != 4:
        raise ValueError(f"Expected a list of 4 strings, got: {lst}")
    return tuple(lst)  # returns (subject, predicate, object, metadata)



if __name__=='__main__':

    result = pd.read_csv('Refined_triplets_trivia_preprocessed.csv')
    
    # Clean the dataframe first
    print("Cleaning data...")
    print(f"Before cleaning: {len(result)} rows")
    
    # Check for NaN values
    print("\nNaN counts per column:")
    print(result.isna().sum())
    
    # Option 1: Drop rows with NaN in critical columns
    # result = result.dropna(subset=['subject', 'predicate', 'object', 'metadata'])
    
    # Option 2: Fill NaN with empty strings (will be caught by validation)
    result = result.fillna('')
    
    print(f"After cleaning: {len(result)} rows")
    
    from db_config import setup_hf_token, get_weaviate_client, get_neo4j_driver
    setup_hf_token()

    weaviate_client = get_weaviate_client()
    driver = get_neo4j_driver()
    triplet_database_name="Triplets_phi4"
    ensure_weaviate_schema(triplet_database_name)

    triplet_ids = []
    failed_indices = []

    # Usage
    try:
        triplet_ids, failed_indices = ingest_triplets_batch(result, batch_size=100)
        result["triplet_id"] = triplet_ids

        if failed_indices:
            print(f"\n{len(failed_indices)} insertions failed at indices: {failed_indices}")
        
        print("Insertion completed!!")
        
    finally:
        # Properly close connections
        print("\nClosing connections...")
        weaviate_client.close()
        driver.close()
        print("Connections closed successfully!")