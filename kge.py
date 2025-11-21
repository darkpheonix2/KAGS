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
            # Check metadata validity
            if not row.metadata or str(row.metadata).strip() == "":
                print(f"⚠️ Skipping row {idx} - empty metadata")
                failed_indices.append(idx)
                batch_triplet_ids.append(None)
                continue
            
            # Embed texts
            text_for_embedding = f"{row.subject} {row.predicate} {row.object}"
            vec = embed_text(text_for_embedding)
            meta_vec = embed_text(row.metadata)
            
            if meta_vec is None:
                print(f"⚠️ Skipping row {idx} - metadata embedding failed")
                failed_indices.append(idx)
                batch_triplet_ids.append(None)
                continue
            
            triplet_id = str(uuid4())
            batch_triplet_ids.append(triplet_id)
            
            batch_objects.append({
                "properties": {
                    "subject": row.subject,
                    "predicate": row.predicate,
                    "object": row.object,
                    "metadata": row.metadata,
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
    """Batch insert to Neo4j"""
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
            batch_data.append({
                "subject": row.subject,
                "object": row.object,
                "predicate": row.predicate,
                "metadata": row.metadata,
                "triplet_id": tid
            })
    
    if batch_data:
        with driver.session() as session:
            session.run(cypher, batch=batch_data)


def embed_text(text: str) -> list[float]:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    emb = embedder.encode(text)
    return emb.tolist()

@lru_cache(maxsize=10000)
def embed_text_cached(text):
    """Cache embeddings to avoid re-computing identical texts"""
    return embed_text(text)

# def ingest_one_triplet(s, p, o, meta):
    # Embed triplet text: subject + predicate + object
    text_for_embedding = f"{s} {p} {o}"
    vec = embed_text_cached(text_for_embedding)
    
    # Embed metadata separately
    meta_vec = embed_text_cached(meta)

    # Debug the metadata
    if not meta or meta.strip() == "":
        print(f"⚠️ Empty metadata for triplet: {s} -> {p} -> {o}")
        meta_vec = None
    else:
        meta_vec = embed_text(meta)
        if meta_vec is None:
            print(f"⚠️ embed_text returned None for metadata: {meta[:100]}...")
    
    # Only insert if we have a valid metadata vector
    if meta_vec is None:
        print(f"❌ Skipping insertion - no valid metadata vector")
        return None
    
    triplet_id = str(uuid4())
    
    collection = weaviate_client.collections.get(triplet_database_name)
    collection.data.insert(
        properties={
            "subject": s,
            "predicate": p,
            "object": o,
            "metadata": meta,
            "triplet_id": triplet_id,
            "metadata_vector": meta_vec  # store metadata embedding as property
        },
        uuid=triplet_id,
        vector=vec
    )
    
    # Insert into Neo4j as before
    cypher = """
    MERGE (sub:Entity {name: $subject})
    MERGE (obj:Entity {name: $object})
    MERGE (sub)-[rel:REL {predicate: $predicate, metadata: $meta, triplet_id: $tid}]->(obj)
    """
    with driver.session() as session:
        session.run(cypher, subject=s, object=o, predicate=p, meta=meta, tid=triplet_id)
    
    return triplet_id


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
    # df = pd.read_csv('Cleaned_triplets.csv')
    # df['cleaned_triplets'] = df['cleaned_triplets'].apply(ast.literal_eval)
    # # explode so each triplet becomes a row
    # d = df.explode('cleaned_triplets', ignore_index=True)

    # # filter out the header rows ['head','relationship','tail','sentences']
    # d = d[d['cleaned_triplets'].apply(lambda x: x != ['head', 'relationship', 'tail', 'sentences'])]

    # # now split list elements into separate columns
    # triplet_df = pd.DataFrame(d['cleaned_triplets'].tolist(), 
    #                         columns=['subject', 'predicate', 'object', 'metadata'])

    # # join back with original df (without cleaned_triplets column)
    # result = d.drop(columns=['cleaned_triplets']).reset_index(drop=True).join(triplet_df)
    # result.drop_duplicates(subset=['subject','predicate','object','metadata'],inplace=True)
    # result.drop(['triplet_count'],axis=1,inplace=True)

    result = pd.read_csv('triplets_phi4.csv')

    # Best practice: store your credentials in environment variables
    weaviate_url = 'xxx'
    weaviate_api_key = 'xxx'

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    # Instance 1
    neo4j_uri = "xxx"
    neo4j_auth = ("neo4j", "xxx")

    driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) 

    triplet_database_name="Triplets_phi4"
    ensure_weaviate_schema(triplet_database_name)


    triplet_ids = []
    failed_indices = []

    # Usage
    triplet_ids, failed_indices = ingest_triplets_batch(result, batch_size=100)
    result["triplet_id"] = triplet_ids

    # for idx, row in tqdm(result.iterrows(), total=len(result)):
    #     tid = ingest_one_triplet(row.subject, row.predicate, row.object, row.metadata)
    #     if tid is None:
    #         print(f"Failed to ingest row {idx}: {row.subject} -> {row.object}")
    #         failed_indices.append(idx)
    #     triplet_ids.append(tid)

    # result["triplet_id"] = triplet_ids

    if failed_indices:
        print(f"\n{len(failed_indices)} insertions failed at indices: {failed_indices}")
    
    print("Insertion completed!!")
