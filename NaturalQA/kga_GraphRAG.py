

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

os.environ["CUDA_VISIBLE_DEVICES"] = "3,0"

# =====================================================
#                  CONFIGURATION
# =====================================================

INPUT_CSV = "GraphRAG_Retrieval_Output.csv"
OUTPUT_DIR = "Output_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEAVIATE_CLUSTER_URL = get_weaviate_url()
WEAVIATE_API_KEY = get_weaviate_api_key()
setup_hf_token()
# =====================================================
#                DATA LOADING
# =====================================================

df = pd.read_csv(INPUT_CSV)
df = df[['question', 'combined_positive', 'answers','retrieved_triplets']].sample(frac=1, random_state=42).reset_index(drop=True)


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
        "You are an expert at answering the question just based on the triplets. "
        "If you cannot answer based on triplets, say so clearly.\n\n"
        f"Triplets:\n{context}\n\nQuestion: {query}\nAnswer:"
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
    df[f"predicted_answer_{model_name}"] = df.apply(
        lambda row: answer_generator(row["question"], row["retrieved_triplets"], tokenizer, model, device, is_seq2seq),
        axis=1
    )
    df.to_csv(f"GraphRAG_output_{model_name}.csv", index=False)
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("✅ All models completed. Results saved to Ouput_Data/")
