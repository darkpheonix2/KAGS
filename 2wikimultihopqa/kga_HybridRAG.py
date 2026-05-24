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
    AutoModelForSeq2SeqLM
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,1,3" 

# =====================================================
#                  CONFIGURATION
# =====================================================

INPUT_CSV = "Hybrid_RAG_retrieval_output.csv"
OUTPUT_DIR = "Output_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
#                DATA LOADING & CLEANING
# =====================================================

print("📊 Loading data...")
df = pd.read_csv(INPUT_CSV)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✅ Loaded {len(df)} rows")

# =====================================================
#               ANSWER GENERATION MODELS
# =====================================================

def load_model(model_name: str, device: str = "auto"):
    """Load model with multi-GPU support using device_map."""
    print(f"🔄 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"🔄 Loading model {model_name}...")
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
    print(f"✅ Model loaded successfully")
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

def clean_memory():
    """Aggressive memory cleanup."""
    print("🧹 Cleaning memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Print memory stats
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print("✅ Memory cleaned")

# =====================================================
#             RUN MODELS AND SAVE RESULTS
# =====================================================

models = {
    "mistral": ("mistralai/Mistral-7B-v0.1", "auto", False),
    "llama": ("meta-llama/Llama-3.2-3B", "auto", False),
    "flan_t5": ("google/flan-t5-xl", "auto", True),
}

for idx, (model_name, (model_path, device, is_seq2seq)) in enumerate(models.items(), 1):
    print(f"\n{'='*60}")
    print(f"🚀 Running model {idx}/{len(models)}: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    tokenizer, model = load_model(model_path, device)
    
    # Generate answers with progress bar
    print(f"🔄 Generating answers...")
    tqdm.pandas(desc=f"Processing with {model_name}")
    df[f"predicted_answer_{model_name}"] = df.progress_apply(
        lambda row: answer_generator(
            row["question"], 
            row["hybrid_context"], 
            tokenizer, 
            model, 
            device, 
            is_seq2seq
        ),
        axis=1
    )
    
    # Save intermediate results
    output_file = os.path.join(OUTPUT_DIR, f"Hybrid_RAG_output_{model_name}.csv")
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")
    
    # Aggressive memory cleanup
    print(f"🗑️  Cleaning up {model_name}...")
    del model
    del tokenizer
    clean_memory()
    print(f"✅ {model_name} completed and cleaned up\n")

# Save final combined results
final_output = os.path.join(OUTPUT_DIR, "Hybrid_RAG_output_all_models.csv")
df.to_csv(final_output, index=False)

print("\n" + "="*60)
print("🎉 All models completed successfully!")
print(f"📁 Final results saved to: {final_output}")
print("="*60)