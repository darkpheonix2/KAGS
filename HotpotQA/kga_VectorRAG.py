#!/usr/bin/env python
# coding: utf-8

"""
Vector RAG Answer Generation Pipeline
- Loads retrieved contexts from Weaviate
- Runs three models: Mistral-7B, Llama-3.2-3B, and Flan-T5-XL
- Saves outputs after each model
"""

import os
import gc
import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# ======================================
# SETUP
# ======================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Adjust as needed
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ======================================
# LOAD DATA
# ======================================
input_path = "Ouput_Data/VectorRAG_retrieved_k1.csv"
df = pd.read_csv(input_path)
print(f"📘 Loaded {len(df)} rows from {input_path}")

# ======================================
# COMMON ANSWER FUNCTION (for causal models)
# ======================================
def generate_answer_causal(model, tokenizer, query, context):
    prompt = (
        "You are an expert at answering questions based only on the given context. "
        "If you cannot answer based on context, clearly say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=pad_token_id,
            max_new_tokens=128
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = output.split("Answer:")[-1].strip().split("\n")[0]
    return answer.replace('"', "").replace("'", "")

# ======================================
# MODEL 1: MISTRAL
# ======================================
print("\n🚀 Running Mistral-7B...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)

df["predicted_answer_mistral"] = df.apply(
    lambda row: generate_answer_causal(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

output_path_mistral = "Ouput_Data/Vector_RAG_final_output_mistral_k1.csv"
df.to_csv(output_path_mistral, index=False)
print(f"✅ Mistral completed in {time.time() - t0:.1f}s → Saved to {output_path_mistral}")

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# ======================================
# MODEL 2: LLAMA
# ======================================
print("\n🚀 Running Llama-3.2-3B...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").to(device)

df["predicted_answer_llama"] = df.apply(
    lambda row: generate_answer_causal(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

output_path_llama = "Ouput_Data/Vector_RAG_final_output_mistral_LLama_k1.csv"
df.to_csv(output_path_llama, index=False)
print(f"✅ Llama completed in {time.time() - t0:.1f}s → Saved to {output_path_llama}")

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# ======================================
# MODEL 3: FLAN-T5
# ======================================
print("\n🚀 Running Flan-T5-XL...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)

def generate_answer_t5(model, tokenizer, query, context, max_new_tokens=128):
    prompt = (
        "Answer the question based only on the context below. "
        "If the context does not contain the answer, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip().replace('"', "").replace("'", "")

df["predicted_answer_flant5"] = df.apply(
    lambda row: generate_answer_t5(model, tokenizer, row["question"], row["retrieved_context"]),
    axis=1
)

output_path_t5 = "Ouput_Data/Vector_RAG_final_output_mistral_LLama_FlanT5_k1.csv"
df.to_csv(output_path_t5, index=False)
print(f"✅ Flan-T5 completed in {time.time() - t0:.1f}s → Saved to {output_path_t5}")

# Final cleanup
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

print("\n🎯 All models completed successfully!")
