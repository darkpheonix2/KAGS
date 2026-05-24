#!/usr/bin/env python
# coding: utf-8

import os
import gc
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# ======================================
# ENVIRONMENT SETUP
# ======================================
import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

from db_config import setup_hf_token

setup_hf_token()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Change as needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("Triplet_Retrieval_Output.csv")
df["retrieved_context"] = df["retrieved_context_text"] + df["retrieved_triplets"].astype(str)

# ======================================
# COMMON ANSWER GENERATION FUNCTION (for causal LM models)
# ======================================
def generate_answer_causal(model, tokenizer, query, context):
    prompt = (
        "You are an expert at answering questions based only on the given context. "
        "If you cannot answer based on context, clearly say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.split("Answer:")[-1].strip()

# ======================================
# MODEL 1: MISTRAL-7B
# ======================================
print("Running inference with Mistral-7B...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)

df["predicted_answer_mistral"] = df.apply(
    lambda row: generate_answer_causal(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

df.to_csv("Ouput_Data/Hybrid_RAG_output_mistral.csv", index=False)
print("✅ Saved Mistral output")

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# ======================================
# MODEL 2: LLAMA-3.2-3B
# ======================================
print("Running inference with Llama-3.2-3B...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").to(device)

df["predicted_answer_llama"] = df.apply(
    lambda row: generate_answer_causal(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

df.to_csv("Hybrid_RAG_output_mistral_llama.csv", index=False)
print("✅ Saved Llama output")

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# ======================================
# MODEL 3: FLAN-T5-XL
# ======================================
print("Running inference with Flan-T5-XL...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)

def generate_answer_t5(model, tokenizer, query, context):
    prompt = (
        "Answer the question based only on the context below. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=128, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

df["predicted_answer_flant5"] = df.apply(
    lambda row: generate_answer_t5(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

df.to_csv("Hybrid_RAG_output_mistral_llama_flant5.csv", index=False)
print("✅ Saved Flan-T5 output")

# Final cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

print("\n🎯 All models completed successfully! Outputs saved.")
