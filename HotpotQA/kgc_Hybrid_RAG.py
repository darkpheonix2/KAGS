import os
import gc
import ast
import re
import time
import json
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
from pathlib import Path
_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))


# ----------------- Utility Functions -----------------
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def parse_context(context):
    """Parse context whether it's stored as string representation of list or actual list"""
    if isinstance(context, str):
        try:
            return ast.literal_eval(context)
        except (ValueError, SyntaxError):
            return context
    return context


def count_context_tokens(context):
    total_tokens = 0
    parsed_context = parse_context(context)

    if isinstance(parsed_context, list):
        for item in parsed_context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, text_snippets = item[0], item[1]
                if isinstance(title, str):
                    total_tokens += count_tokens(title)
                if isinstance(text_snippets, list):
                    for snippet in text_snippets:
                        if isinstance(snippet, str):
                            total_tokens += count_tokens(snippet)
                elif isinstance(text_snippets, str):
                    total_tokens += count_tokens(text_snippets)
            elif isinstance(item, str):
                total_tokens += count_tokens(item)
    elif isinstance(parsed_context, str):
        total_tokens += count_tokens(parsed_context)

    return total_tokens


def flatten_context(context):
    parsed_context = parse_context(context)
    if isinstance(parsed_context, list):
        flattened_parts = []
        for item in parsed_context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, text_snippets = item[0], item[1]
                if isinstance(title, str):
                    flattened_parts.append(title)
                if isinstance(text_snippets, list):
                    flattened_parts.extend([s for s in text_snippets if isinstance(s, str)])
                elif isinstance(text_snippets, str):
                    flattened_parts.append(text_snippets)
            elif isinstance(item, str):
                flattened_parts.append(item)
        return " ".join(flattened_parts)
    elif isinstance(parsed_context, str):
        return parsed_context
    else:
        return str(parsed_context)


def preprocess_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = re.sub(r'[^a-zA-Z0-9\s@.?\–\-\(\)\,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)

    stop_words = {'a', 'am', 'an', 'and', 'but', 'd', 'o', 're', 's', 't', 'the', 'y'}
    filtered_tokens = [word for word in tokens if word not in stop_words and word.strip() != '']
    return ' '.join(filtered_tokens).strip()


def create_chunks(text, chunk_size=256, overlap=64):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = count_tokens(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_chunk, overlap_length = [], 0
            j = len(current_chunk) - 1
            while j >= 0 and overlap_length < overlap:
                overlap_sentence = current_chunk[j]
                overlap_sentence_length = count_tokens(overlap_sentence)
                if overlap_length + overlap_sentence_length <= overlap:
                    overlap_chunk.insert(0, overlap_sentence)
                    overlap_length += overlap_sentence_length
                    j -= 1
                else:
                    break
            current_chunk, current_length = overlap_chunk, overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length
        i += 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return '<c>'.join(chunks)


def extract_triplets_from_response(response_text):
    """Extract triplets robustly from model responses"""
    if "assistant" in response_text.lower():
        parts = re.split(r'(?i)assistant', response_text)
        if len(parts) > 1:
            response_text = parts[-1]

    # More flexible pattern to handle ' or " and spacing variations
    triplet_pattern = r'\[["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\]'
    matches = re.findall(triplet_pattern, response_text)

    parsed_triplets = []
    for m in matches:
        if len(m) == 3:
            parsed_triplets.append(list(m))
    return parsed_triplets


# ----------------- Optimized Model Pipeline -----------------
def load_model(model_name, device_map="auto", gpu_ids=None):
    print(f"Loading model: {model_name}")
    if gpu_ids is not None:
        if isinstance(gpu_ids, (list, tuple)):
            device_map = {i: f"cuda:{gpu_id}" for i, gpu_id in enumerate(gpu_ids)}
        else:
            device_map = f"cuda:{gpu_ids}"
        print(f"Using GPU(s): {gpu_ids}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    ).eval()
    return tokenizer, model


def extract_triplets_from_chunks(df, model_name, chunks_column='chunks_from_preprocessed', batch_size=4, gpu_ids=None):
    # System and user prompts
    system_prompt = """You are an expert at extracting structured knowledge in the form of triplets and metadata. 
Given a context, identify entities and their relationships, and represent them strictly as triplets: ["head", "relationship", "tail"]. 
Only output valid triplets. No explanations, no commentary.
"""

    user_prompt_template = """Here is the context. Stick to the format and extract all valid triplets:

{context}"""

    tokenizer, model = load_model(model_name, gpu_ids=gpu_ids)

    col = f'extracted_triplets_{model_name.replace("/", "_")}'
    timet = f'extraction_time_{model_name.replace("/", "_")}'
    num_col = f'num_triplets_{model_name.replace("/", "_")}'
    
    df[col] = None
    df[timet] = None
    df[num_col] = None

    for idx in tqdm(range(len(df)), desc=f"Extracting with {model_name}"):
        chunks_text = df.at[idx, chunks_column]
        chunks = [c.strip() for c in chunks_text.split('<c>') if c.strip()]
        
        all_triplets_for_row = []
        total_time = 0
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]

            prompts = [
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt_template.format(context=c)}]
                for c in batch_chunks
            ]

            batch_inputs = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            stime = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id
                )
            batch_time = time.time() - stime
            total_time += batch_time

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for response in decoded:
                triplets = extract_triplets_from_response(response)
                all_triplets_for_row.extend(triplets)

            torch.cuda.empty_cache()

        df.at[idx, col] = json.dumps(all_triplets_for_row)
        df.at[idx, timet] = total_time
        df.at[idx, num_col] = len(all_triplets_for_row)

        if (idx + 1) % 10 == 0:
            df.to_csv('Refined_triplets_checkpoint_hybridRAG.csv', index=False, escapechar='\\')
            print(f"Checkpoint saved at row {idx + 1}")

    gc.collect()
    return df


# ----------------- Main -----------------
if __name__ == '__main__':
    from db_config import setup_hf_token
    setup_hf_token()
    df = pd.read_json('Input_Data/test_subsampled.json', lines=True)
    df = df[['question', 'context', 'answer', 'supporting_facts']].sample(frac=1, random_state=42).reset_index(drop=True)

    tqdm.pandas(desc="Counting tokens")
    df['context_token_count'] = df['context'].progress_apply(count_context_tokens)
    df['raw_context'] = df['context'].apply(flatten_context)

    tqdm.pandas(desc="Preprocessing text")
    df['preprocessed_context'] = df['raw_context'].progress_apply(preprocess_text)

    tqdm.pandas(desc="Creating chunks")
    df['chunks_from_preprocessed'] = df['preprocessed_context'].progress_apply(create_chunks)

    # Model inference
    model_id = "microsoft/Phi-4-mini-instruct"
    gpu_ids = 2  # specify your GPU id(s)

    # print(df.head())
    df = extract_triplets_from_chunks(df, model_id, 'chunks_from_preprocessed', batch_size=4, gpu_ids=gpu_ids)
    # print(df.head())
    print("✅ Triplet extraction completed successfully!")

    df.to_csv('Refined_triplets_final_HybridRAG.csv', index=False, escapechar='\\')
