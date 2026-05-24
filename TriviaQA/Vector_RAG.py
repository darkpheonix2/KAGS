

import os
import gc
import ast
import torch
import pandas as pd
from tqdm import tqdm
import tiktoken
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType
import weaviate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

from db_config import get_weaviate_url, get_weaviate_api_key, setup_hf_token

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# =====================================================
#                  CONFIGURATION
# =====================================================

INPUT_CSV = "TriviaQA_test.csv"
OUTPUT_DIR = "Output_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEAVIATE_CLUSTER_URL = get_weaviate_url()
WEAVIATE_API_KEY = get_weaviate_api_key()
setup_hf_token()
# =====================================================
#                DATA LOADING & CLEANING
# =====================================================

df = pd.read_csv(INPUT_CSV)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =====================================================
#                   TOKEN COUNT HELPERS
# =====================================================

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def parse_context(context):
    """Safely parse context string into list or leave as-is."""
    if isinstance(context, str):
        try:
            return ast.literal_eval(context)
        except (ValueError, SyntaxError):
            return context
    return context

def count_context_tokens(context) -> int:
    """Count tokens in nested context structures."""
    total_tokens = 0
    parsed = parse_context(context)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, snippets = item[0], item[1]
                if isinstance(title, str):
                    total_tokens += count_tokens(title)
                if isinstance(snippets, list):
                    total_tokens += sum(count_tokens(s) for s in snippets if isinstance(s, str))
                elif isinstance(snippets, str):
                    total_tokens += count_tokens(snippets)
            elif isinstance(item, str):
                total_tokens += count_tokens(item)
    elif isinstance(parsed, str):
        total_tokens += count_tokens(parsed)
    return total_tokens

df['positive_context_token_count'] = df['combined_positive'].apply(count_context_tokens)

# =====================================================
#                 CHUNK CREATION
# =====================================================

def create_chunks(text: str, chunk_size=256, overlap=64) -> str:
    """Split text into overlapping chunks."""
    if not isinstance(text, str):  # handles NaN, None, float, etc.
        text = str(text)
    sentences = text.split('\n')
    chunks, current_chunk, current_len = [], [], 0
    i = 0

    while i < len(sentences):
        s = sentences[i]
        s_len = count_tokens(s)
        if current_len + s_len > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            # Create overlap
            overlap_chunk, overlap_len = [], 0
            for j in range(len(current_chunk) - 1, -1, -1):
                overlap_s = current_chunk[j]
                overlap_s_len = count_tokens(overlap_s)
                if overlap_len + overlap_s_len <= overlap:
                    overlap_chunk.insert(0, overlap_s)
                    overlap_len += overlap_s_len
                else:
                    break
            current_chunk, current_len = overlap_chunk, overlap_len
        current_chunk.append(s)
        current_len += s_len
        i += 1

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return '<c>'.join(chunks)

df['chunks'] = df['combined_positive'].apply(create_chunks)

# =====================================================
#                WEAVIATE EMBEDDING STORAGE
# =====================================================

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str) -> list[float]:
    return embedder.encode(text).tolist()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
)

CLASS_NAME = "VectorRAG"

if not client.collections.exists(CLASS_NAME):
    client.collections.create(
        name=CLASS_NAME,
        vectorizer_config=None,
        properties=[Property(name="content", data_type=DataType.TEXT)],
    )

collection = client.collections.get(CLASS_NAME)

# After creating chunks in the dataframe
df['chunks'] = df['combined_positive'].apply(create_chunks)

# Explode chunks and insert into Weaviate
all_chunks = []
for chunks_str in df['chunks']:
    all_chunks.extend(chunks_str.split('<c>'))

print(f"Total chunks to insert: {len(all_chunks)}")

# Insert chunks into Weaviate with their embeddings
with collection.batch.dynamic() as batch:
    for i, chunk in enumerate(all_chunks):
        if chunk.strip():  # Skip empty chunks
            batch.add_object(
                properties={"content": chunk},
                vector=embed_text(chunk)
            )
        
        # Optional: Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Inserted {i + 1}/{len(all_chunks)} chunks")

print(f"Insertion complete. Collection now has {len(collection)} objects")

# =====================================================

def get_top_chunks(question: str, top_k: int = 1) -> str:
    vector = embed_text(question)
    response = collection.query.near_vector(
        near_vector=vector, limit=top_k, return_properties=["content"]
    )
    return " ".join([r.properties["content"] for r in response.objects])

tqdm.pandas()
df["retrieved_context"] = df["question"].progress_apply(get_top_chunks)

# =====================================================
#               ANSWER GENERATION MODELS
# =====================================================

def load_model(model_name: str, device: str = "auto"):
    """Load model with multi-GPU support using device_map."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically distributes across GPUs
            torch_dtype=torch.float16  # Use half precision for efficiency
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically distributes across GPUs
            torch_dtype=torch.float16
        )
    return tokenizer, model

def answer_generator(query: str, context: str, tokenizer, model, device=None, seq2seq=False):
    """Generate answer with proper truncation settings."""
    prompt = (
        "You are an expert at answering the question just based on the context. "
        "If you cannot answer based on context, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Option 1: Use model's max length
    max_length = tokenizer.model_max_length
    if max_length > 100000:  # Some models have unreasonably large defaults
        max_length = 2048  # Set a reasonable default
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True,
        max_length=max_length,
        padding=False
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        if seq2seq:
            outputs = model.generate(**inputs, max_new_tokens=128, early_stopping=True)
        else:
            outputs = model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128
            )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in output:
        answer = output.split("Answer:")[-1].strip().split("\n")[0]
    else:
        answer = output.strip().split("\n")[-1]
    return answer.replace('"', '').replace("'", "")
# =====================================================
#             RUN MODELS AND SAVE RESULTS
# =====================================================

models = {
    "mistral": ("mistralai/Mistral-7B-v0.1", "auto", False),
    "llama": ("meta-llama/Llama-3.2-3B", "auto", False),
    "flan_t5": ("google/flan-t5-xl", "auto", True),
}

for model_name, (model_path, device, is_seq2seq) in models.items():
    print(f"\nRunning model: {model_name}")
    tokenizer, model = load_model(model_path, device)
    df[f"predicted_answer_{model_name}"] = df.progress_apply(
        lambda row: answer_generator(row["question"], row["retrieved_context"], tokenizer, model, device, is_seq2seq),
        axis=1
    )
    df.to_csv(f"Vector_RAG_output_k1_{model_name}.csv", index=False)
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("✅ All models completed. Results saved to Ouput_Data/")
