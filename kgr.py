import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time
from sentence_transformers import SentenceTransformer
import re
import ast
# from weaviate.classes.init import Auth
import weaviate
from weaviate.auth import Auth
from weaviate.classes.config import Property, DataType, Configure
from neo4j import GraphDatabase

triplet_database_name="Test_dataset_Triplet"

# Neo4j driver
neo4j_uri = "nxx"
neo4j_auth = ("neo4j", "xxx")
driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) 

# Best practice: store your credentials in environment variables
weaviate_url = 'xxx'
weaviate_api_key = 'xxx'

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(weaviate_client.is_ready()) 

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query_pipeline_with_hops(
    query_text: str, 
    top_k: int = 7, 
    meta_sim_threshold: float = 0.5,
    h_hops: int = 2,
    max_nodes_per_hop: int = 10
):
    """
    Query pipeline with h-hop graph traversal.
    
    Args:
        query_text: User query
        top_k: Number of initial vector matches
        meta_sim_threshold: Minimum similarity for metadata filtering
        h_hops: Number of hops to traverse in the graph (1, 2, 3, etc.)
        max_nodes_per_hop: Maximum nodes to explore per hop (prevents explosion)
    
    Returns:
        Dict containing filtered hits and h-hop graph context
    """
    # Step 1: Vector search
    q_vec = embed_text(query_text)
    
    collection = weaviate_client.collections.get(triplet_database_name)
    response = collection.query.near_vector(
        near_vector=q_vec,
        limit=top_k,
        return_properties=["triplet_id", "subject", "predicate", "object", "metadata", "metadata_vector"]
    )
    
    hits = [obj.properties for obj in response.objects]
    
    # Step 2: Filter by metadata similarity
    filtered = []
    for hit in hits:
        stored_meta_vec = hit.get("metadata_vector")
        if stored_meta_vec and cosine_sim(q_vec, stored_meta_vec) >= meta_sim_threshold:
            filtered.append(hit)
    
    # Step 3: Extract seed entities for graph traversal
    seed_entities = set()
    for hit in filtered:
        seed_entities.add(hit["subject"])
        seed_entities.add(hit["object"])
    
    # Step 4: Perform h-hop traversal
    graph_context = traverse_h_hops(
        seed_entities=list(seed_entities),
        h_hops=h_hops,
        max_nodes_per_hop=max_nodes_per_hop
    )
    
    # Step 5: Combine results
    return {
        "filtered_hits": filtered,
        "seed_entities": list(seed_entities),
        "graph_context": graph_context,
        "stats": {
            "total_vector_matches": len(hits),
            "filtered_matches": len(filtered),
            "seed_entities_count": len(seed_entities),
            "hops_traversed": h_hops,
            "total_nodes_discovered": len(graph_context["all_entities"]),
            "total_relations_discovered": len(graph_context["all_relations"])
        }
    }


def traverse_h_hops(
    seed_entities: List[str], 
    h_hops: int,
    max_nodes_per_hop: int = 10
) -> Dict[str, Any]:
    """
    Traverse h-hops from seed entities in Neo4j knowledge graph.
    
    Returns a structured representation of the subgraph.
    """
    if h_hops < 1:
        return {"all_entities": set(seed_entities), "all_relations": [], "hop_layers": []}
    
    # Track entities at each hop level
    hop_layers = []
    visited_entities = set(seed_entities)
    visited_relations = []
    current_entities = set(seed_entities)
    
    with driver.session() as sess:
        for hop in range(h_hops):
            if not current_entities:
                break
            
            # Limit entities to prevent explosion
            entities_to_explore = list(current_entities)[:max_nodes_per_hop]
            
            # Cypher query for 1-hop expansion (both directions)
            cy_expand = """
            MATCH (e:Entity)
            WHERE e.name IN $entities
            
            // Outgoing relationships
            OPTIONAL MATCH (e)-[r_out]->(target:Entity)
            WITH e, collect(DISTINCT {
                direction: 'outgoing',
                source: e.name,
                predicate: r_out.predicate,
                target: target.name,
                metadata: r_out.metadata,
                triplet_id: r_out.triplet_id
            }) AS outgoing
            
            // Incoming relationships
            OPTIONAL MATCH (source:Entity)-[r_in]->(e)
            WITH e, outgoing, collect(DISTINCT {
                direction: 'incoming',
                source: source.name,
                predicate: r_in.predicate,
                target: e.name,
                metadata: r_in.metadata,
                triplet_id: r_in.triplet_id
            }) AS incoming
            
            RETURN 
                e.name AS entity,
                outgoing,
                incoming
            """
            
            results = sess.run(cy_expand, entities=entities_to_explore).data()
            
            # Process results for this hop
            next_hop_entities = set()
            hop_relations = []
            
            for rec in results:
                # Process outgoing relations
                for rel in rec["outgoing"]:
                    if rel["target"]:  # Filter out null targets
                        hop_relations.append(rel)
                        if rel["target"] not in visited_entities:
                            next_hop_entities.add(rel["target"])
                
                # Process incoming relations
                for rel in rec["incoming"]:
                    if rel["source"]:  # Filter out null sources
                        hop_relations.append(rel)
                        if rel["source"] not in visited_entities:
                            next_hop_entities.add(rel["source"])
            
            # Store this hop's information
            hop_layers.append({
                "hop_number": hop + 1,
                "explored_entities": entities_to_explore,
                "discovered_entities": list(next_hop_entities),
                "relations": hop_relations,
                "relation_count": len(hop_relations)
            })
            
            # Update tracking sets
            visited_entities.update(next_hop_entities)
            visited_relations.extend(hop_relations)
            current_entities = next_hop_entities
    
    return {
        "all_entities": list(visited_entities),
        "all_relations": visited_relations,
        "hop_layers": hop_layers
    }


def format_graph_context_for_llm(graph_context: Dict[str, Any]) -> str:
    """
    Format the h-hop graph context into a readable string for LLM consumption.
    """
    output = []
    output.append("=== KNOWLEDGE GRAPH CONTEXT ===\n")
    
    for layer in graph_context["hop_layers"]:
        output.append(f"\n--- Hop {layer['hop_number']} ---")
        output.append(f"Explored {len(layer['explored_entities'])} entities")
        output.append(f"Discovered {len(layer['discovered_entities'])} new entities")
        output.append(f"Found {layer['relation_count']} relations\n")
        
        # Group relations by predicate for better readability
        relations_by_predicate = defaultdict(list)
        for rel in layer["relations"]:
            relations_by_predicate[rel["predicate"]].append(rel)
        
        for predicate, rels in relations_by_predicate.items():
            output.append(f"\n  {predicate}:")
            for rel in rels[:5]:  # Limit to 5 examples per predicate
                if rel["direction"] == "outgoing":
                    output.append(f"    • {rel['source']} → {rel['target']}")
                else:
                    output.append(f"    • {rel['source']} → {rel['target']}")
            if len(rels) > 5:
                output.append(f"    ... and {len(rels) - 5} more")
    
    output.append(f"\n=== TOTAL: {len(graph_context['all_entities'])} entities, "
                 f"{len(graph_context['all_relations'])} relations ===")
    
    return "\n".join(output)


def query_pipeline_with_entity_ranking(
    query_text: str, 
    top_k: int = 7, 
    meta_sim_threshold: float = 0.8,
    h_hops: int = 2,
    max_nodes_per_hop: int = 10
):
    """
    Enhanced pipeline that ranks entities by relevance using graph features.
    """
    result = query_pipeline_with_hops(
        query_text, top_k, meta_sim_threshold, h_hops, max_nodes_per_hop
    )
    
    # Calculate entity importance scores
    entity_scores = defaultdict(float)
    
    # Score from being in seed set (highest importance)
    for entity in result["seed_entities"]:
        entity_scores[entity] += 10.0
    
    # Score from hop distance (closer = more important)
    for layer in result["graph_context"]["hop_layers"]:
        hop_penalty = 1.0 / layer["hop_number"]
        for entity in layer["discovered_entities"]:
            entity_scores[entity] += hop_penalty
    
    # Score from degree centrality (more connections = more important)
    entity_degrees = defaultdict(int)
    for rel in result["graph_context"]["all_relations"]:
        entity_degrees[rel["source"]] += 1
        entity_degrees[rel["target"]] += 1
    
    for entity, degree in entity_degrees.items():
        entity_scores[entity] += 0.1 * degree
    
    # Rank entities
    ranked_entities = sorted(
        entity_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    result["ranked_entities"] = [
        {"entity": e, "score": s} for e, s in ranked_entities[:20]
    ]
    
    return result

def parse_triplet_list(lst: List[str]) -> Tuple[str, str, str, str]:
    """
    Safely parse a triplet that is already a list of 4 strings.
    Example: ["Arthur Turner", "was born on", "1 April 1909", "Some metadata"]
    """
    if not isinstance(lst, list) or len(lst) != 4:
        raise ValueError(f"Expected a list of 4 strings, got: {lst}")
    return tuple(lst)

def safe_parse(s: str):
    if not isinstance(s, str):
        return None
    # remove unwanted markers like <|end|>
    s = re.sub(r"<\|end\|>", "", s).strip()
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"⚠️ Skipping malformed triplet: {s}")
        return None

def embed_text(text: str) -> list[float]:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    emb = embedder.encode(text)
    return emb.tolist()

def deduplicate_triplets(relations: List[Dict], method: str = "hybrid") -> List[Dict]:
    """
    Remove redundant triplets using various strategies.
    
    Args:
        relations: List of relation dictionaries
        method: "exact", "semantic", or "hybrid"
    
    Returns:
        Deduplicated list of relations
    """
    if not relations:
        return []
    
    if method == "exact":
        return _deduplicate_exact(relations)
    elif method == "semantic":
        return _deduplicate_semantic(relations)
    else:  # hybrid
        # First remove exact duplicates, then semantic
        exact_dedup = _deduplicate_exact(relations)
        return _deduplicate_semantic(exact_dedup)


def _deduplicate_exact(relations: List[Dict]) -> List[Dict]:
    """Remove exact duplicate triplets."""
    seen = set()
    unique_relations = []
    
    for rel in relations:
        # Create a unique key from core triplet components
        key = (
            rel.get("source", ""),
            rel.get("predicate", ""),
            rel.get("target", "")
        )
        
        if key not in seen and all(key):  # Ensure no empty values
            seen.add(key)
            unique_relations.append(rel)
    
    return unique_relations


def _deduplicate_semantic(relations: List[Dict], similarity_threshold: float = 0.90) -> List[Dict]:
    """
    Remove semantically similar triplets using embedding similarity.
    Keeps the triplet from the earlier hop (more relevant).
    """
    if not relations:
        return []
    
    # Create text representation of each triplet
    triplet_texts = []
    for rel in relations:
        text = f"{rel.get('source', '')} {rel.get('predicate', '')} {rel.get('target', '')}"
        triplet_texts.append(text)
    
    # Get embeddings for all triplets
    try:
        embeddings = [embed_text(text) for text in triplet_texts]
        embeddings = [e for e in embeddings if e is not None]
        
        if len(embeddings) != len(relations):
            print("Warning: Some embeddings failed, skipping semantic deduplication")
            return relations
    except Exception as e:
        print(f"Error during embedding: {e}, skipping semantic deduplication")
        return relations
    
    # Find similar pairs and keep only one from each group
    keep_indices = set(range(len(relations)))
    
    for i in range(len(embeddings)):
        if i not in keep_indices:
            continue
            
        for j in range(i + 1, len(embeddings)):
            if j not in keep_indices:
                continue
            
            similarity = cosine_sim(embeddings[i], embeddings[j])
            
            if similarity >= similarity_threshold:
                # Keep the one from earlier hop (higher priority)
                hop_i = relations[i].get("hop_number", 999)
                hop_j = relations[j].get("hop_number", 999)
                
                if hop_i <= hop_j:
                    keep_indices.discard(j)
                else:
                    keep_indices.discard(i)
                    break
    
    return [relations[i] for i in sorted(keep_indices)]


def _deduplicate_with_clustering(relations: List[Dict], max_per_cluster: int = 3) -> List[Dict]:
    """
    Advanced deduplication using clustering.
    Groups similar triplets and keeps top representatives from each cluster.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    if len(relations) <= max_per_cluster:
        return relations
    
    # Get embeddings
    triplet_texts = [
        f"{rel.get('source', '')} {rel.get('predicate', '')} {rel.get('target', '')}"
        for rel in relations
    ]
    
    try:
        embeddings = np.array([embed_text(text) for text in triplet_texts])
        
        # Cluster triplets
        n_clusters = max(1, len(relations) // max_per_cluster)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        labels = clustering.fit_predict(embeddings)
        
        # Keep top representatives from each cluster (prioritize earlier hops)
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append((idx, relations[idx]))
        
        selected = []
        for cluster_items in clusters.values():
            # Sort by hop number (earlier is better)
            cluster_items.sort(key=lambda x: x[1].get("hop_number", 999))
            selected.extend([item[1] for item in cluster_items[:max_per_cluster]])
        
        return selected
    
    except Exception as e:
        print(f"Clustering failed: {e}, using semantic deduplication instead")
        return _deduplicate_semantic(relations)


def format_deduplicated_context(
    graph_context: Dict,
    dedup_method: str = "hybrid",
    max_relations_per_predicate: int = 5,
    include_metadata: bool = False
) -> str:
    """
    Format graph context with deduplication applied.
    
    Args:
        graph_context: Graph traversal results
        dedup_method: "exact", "semantic", "hybrid", or "clustering"
        max_relations_per_predicate: Max relations to show per predicate type
        include_metadata: Whether to include metadata
    
    Returns:
        Formatted, deduplicated context string
    """
    # Deduplicate all relations
    if dedup_method == "clustering":
        deduplicated_relations = _deduplicate_with_clustering(
            graph_context["all_relations"],
            max_per_cluster=max_relations_per_predicate
        )
    else:
        deduplicated_relations = deduplicate_triplets(
            graph_context["all_relations"],
            method=dedup_method
        )
    
    # Group by predicate
    relations_by_predicate = defaultdict(list)
    for rel in deduplicated_relations:
        predicate = rel.get("predicate", "unknown")
        relations_by_predicate[predicate].append(rel)
    
    # Format output
    lines = ["=== KNOWLEDGE GRAPH CONTEXT ===\n"]
    
    # Group by hop for better organization
    hop_groups = defaultdict(list)
    for rel in deduplicated_relations:
        hop_groups[rel.get("hop_number", 0)].append(rel)
    
    for hop_num in sorted(hop_groups.keys()):
        rels = hop_groups[hop_num]
        lines.append(f"\n--- Hop {hop_num} ({len(rels)} unique relations) ---")
        
        # Group by predicate within this hop
        hop_predicates = defaultdict(list)
        for rel in rels:
            hop_predicates[rel.get("predicate", "unknown")].append(rel)
        
        for predicate, pred_rels in sorted(hop_predicates.items()):
            lines.append(f"\n  {predicate}:")
            for rel in pred_rels[:max_relations_per_predicate]:
                triplet_line = f"    • {rel['source']} → {rel['target']}"
                
                if include_metadata and rel.get("metadata"):
                    metadata = rel["metadata"]
                    if isinstance(metadata, dict) and metadata:
                        meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                        triplet_line += f" [{meta_str}]"
                
                lines.append(triplet_line)
            
            if len(pred_rels) > max_relations_per_predicate:
                lines.append(f"    ... and {len(pred_rels) - max_relations_per_predicate} more")
    
    original_count = len(graph_context["all_relations"])
    dedup_count = len(deduplicated_relations)
    reduction = ((original_count - dedup_count) / original_count * 100) if original_count > 0 else 0
    
    lines.append(f"\n=== SUMMARY ===")
    lines.append(f"Total unique entities: {len(graph_context['all_entities'])}")
    lines.append(f"Relations: {dedup_count} (reduced from {original_count}, {reduction:.1f}% reduction)")
    
    return "\n".join(lines)


def process_question_with_retrieval(
    question: str,
    top_k: int = 7,
    meta_sim_threshold: float = 0.5,
    h_hops: int = 2,
    max_nodes_per_hop: int = 10,
    dedup_method: str = "hybrid",
    verbose: bool = False
) -> Tuple[str, str]:
    """
    Process a single question and return deduplicated context and timestamp.
    
    Returns:
        (retrieved_context, time_of_retrieval)
    """
    try:
        start_time = time.time()
        
        # Run the query pipeline
        result = query_pipeline_with_hops(
            query_text=question,
            top_k=top_k,
            meta_sim_threshold=meta_sim_threshold,
            h_hops=h_hops,
            max_nodes_per_hop=max_nodes_per_hop,
        )
        
        # Format with deduplication
        context = format_deduplicated_context(
            result["graph_context"],
            dedup_method=dedup_method,
            max_relations_per_predicate=10,  # Increased for more context
            include_metadata=False  # Set to True if you want metadata
        )
        
        # Record timestamp
        timestamp = datetime.now().isoformat()
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"✓ Processed in {timestamp}s")
        
        return context, elapsed
    
    except Exception as e:
        print(f"Error processing question '{question[:50]}...': {e}")
        return "", datetime.now().isoformat()


def ensure_weaviate_schema():
    triplet_database_name="Test_dataset_Triplet"
    try:
        # Check if collection already exists
        if weaviate_client.collections.exists(triplet_database_name):
            print("Collection 'Triplet' already exists")
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

def process_dataframe_with_retrieval(
    df: pd.DataFrame,
    question_column: str = "question",
    top_k: int = 7,
    meta_sim_threshold: float = 0.5,
    h_hops: int = 2,
    max_nodes_per_hop: int = 10,
    dedup_method: str = "hybrid",
    verbose: bool = True,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Process all questions in dataframe and add retrieved context columns.
    
    Args:
        df: Input dataframe with questions
        question_column: Name of column containing questions
        top_k: Number of initial vector matches
        meta_sim_threshold: Metadata similarity threshold
        h_hops: Number of hops to traverse
        max_nodes_per_hop: Max nodes per hop
        dedup_method: "exact", "semantic", "hybrid", or "clustering"
        verbose: Print progress
        batch_size: Print progress every N questions
    
    Returns:
        Dataframe with added columns: retrieved_context, time_of_retrieval
    """
    
    ensure_weaviate_schema()        

    if question_column not in df.columns:
        raise ValueError(f"Column '{question_column}' not found in dataframe")
    
    df = df.copy()
    
    contexts = []
    timestamps = []
    
    total = len(df)
    
    if verbose:
        print(f"Processing {total} questions...")
        print(f"Settings: top_k={top_k}, meta_sim_threshold={meta_sim_threshold}, "
              f"h_hops={h_hops}, dedup_method={dedup_method}\n")
    
    for idx, question in enumerate(df[question_column], 1):
        if verbose and idx % batch_size == 0:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        
        context, timestamp = process_question_with_retrieval(
            question=question,
            top_k=top_k,
            meta_sim_threshold=meta_sim_threshold,
            h_hops=h_hops,
            max_nodes_per_hop=max_nodes_per_hop,
            dedup_method=dedup_method,
            verbose=False
        )
        
        contexts.append(context)
        timestamps.append(timestamp)
    
    df["retrieved_context"] = contexts
    df["time_of_retrieval"] = timestamps
    
    if verbose:
        print(f"\n✓ Completed! Processed {total} questions")
        avg_context_len = df["retrieved_context"].str.len().mean()
        print(f"Average context length: {avg_context_len:.0f} characters")
    
    return df


# Example usage
if __name__ == "__main__":

    # triplet_database_name="Test_dataset_Triplet"
    # Create sample dataframe
    df = pd.read_json('Input_Data/test_subsampled.json', lines=True)
    # df = df.sample(5)

    # Process the dataframe
    result_df = process_dataframe_with_retrieval(
        df=df,
        question_column="question",
        top_k=7,
        meta_sim_threshold=0.5,
        h_hops=2,
        max_nodes_per_hop=10,
        dedup_method="hybrid",  # Options: "exact", "semantic", "hybrid", "clustering"
        verbose=True
    )
    result_df.to_csv('Test_kgr.csv', index=False)
    weaviate_client.close()
    driver.close()
