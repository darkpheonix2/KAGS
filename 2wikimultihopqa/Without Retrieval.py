#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-model Question Answering Script using:
1. Mistral-7B
2. LLaMA-3.2-3B
3. FLAN-T5-XL

This script:
- Loads a CSV file with questions
- Generates answers using different LLMs
- Saves results to CSV files
"""

import os
import gc
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


# ==============================
# Configuration
# ==============================

# Set environment variables
import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

from db_config import setup_hf_token

setup_hf_token()
os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load input data
df = pd.read_csv("test_data.csv")
print(f"Loaded dataset with {len(df)} rows.")


# ==============================
# Helper Functions
# ==============================

def cleanup():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()


def save_results(df, filename):
    """Save DataFrame to CSV."""
    df.to_csv(filename, index=False)
    print(f"✅ Results saved to {filename}")


def answer_generator_single_hop_causal(query, tokenizer, model, device):
    """Generate answer using a causal LM (Mistral, LLaMA)."""
    prompt = (
        "You are an expert at answering the question. "
        "Given the user question, answer the question. If you cannot answer, "
        "state properly that you cannot answer.\n\n"
        f"User question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(
        **inputs,
        pad_token_id=pad_token_id,
        max_new_tokens=128
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the answer
    if "Answer:" in output:
        answer = output.split("Answer:")[-1].strip().split("\n")[0]
    else:
        answer = output.strip().split("\n")[-1]

    return answer.replace('"', '').replace("'", "")


def answer_generator_single_hop_seq2seq(query, tokenizer, model, device, max_new_tokens=128):
    """Generate answer using a Seq2Seq LM (FLAN-T5)."""
    prompt = (
        "You are an expert at answering the question. "
        "Given the user question, answer the question. If you cannot answer, "
        "state properly that you cannot answer.\n\n"
        f"User question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, early_stopping=True)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.strip().replace('"', '').replace("'", "")


# ==============================
# Model 1: Mistral-7B
# ==============================

# print("\n🚀 Loading Mistral-7B-v0.1...")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(DEVICE)

# df["predicted_answer_mistral"] = df["question"].apply(
#     lambda q: answer_generator_single_hop_causal(q, tokenizer, model, DEVICE)
# )

# save_results(df, "Without_retrieval_output_mistral.csv")

# # Cleanup
# del model, tokenizer
# cleanup()


# ==============================
# Model 2: LLaMA-3.2-3B
# ==============================

# print("\n🚀 Loading LLaMA-3.2-3B...")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").to(DEVICE)

# df["predicted_answer_llama"] = df["question"].apply(
#     lambda q: answer_generator_single_hop_causal(q, tokenizer, model, DEVICE)
# )

# save_results(df, "Without_retrieval_output_mistral_llama.csv")

# # Cleanup
# del model, tokenizer
# cleanup()


# ==============================
# Model 3: FLAN-T5-XL
# ==============================

print("\n🚀 Loading FLAN-T5-XL...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(DEVICE)

df["predicted_answer_flan_t5"] = df["question"].apply(
    lambda q: answer_generator_single_hop_seq2seq(q, tokenizer, model, DEVICE)
)

save_results(df, "Without_retrieval_output_mistral_llama_FlanT5.csv")

# Final cleanup
del model, tokenizer
cleanup()

print("\n🎉 All models completed successfully!")
