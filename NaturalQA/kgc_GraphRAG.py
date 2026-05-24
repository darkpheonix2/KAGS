

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

from db_config import setup_hf_token

setup_hf_token()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
    """Extract only the assistant's triplet output from the full response"""
    # Find the last occurrence of assistant's response
    if "assistant" in response_text.lower():
        parts = re.split(r'(?i)assistant', response_text)
        if len(parts) > 1:
            response_text = parts[-1]
    
    # Extract lines that look like triplets: ["...", "...", "...", "..."]
    triplet_pattern = r'\["[^"]*"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\]'
    triplets = re.findall(triplet_pattern, response_text)
    
    # Parse each triplet string into a list
    parsed_triplets = []
    for triplet_str in triplets:
        try:
            parsed = ast.literal_eval(triplet_str)
            if isinstance(parsed, list) and len(parsed) == 4:
                parsed_triplets.append(parsed)
        except:
            continue
    
    return parsed_triplets


# ----------------- Optimized Model Pipeline -----------------
def extract_triplets_from_chunks(df, model_name, chunks_column='chunks_from_preprocessed', batch_size=4, gpu_ids=None):
    # System prompt
    system_prompt = """You are an expert at extracting structured knowledge in the form of triplets and metadata. 
Given a context, your task is to identify entities and their relationships, and represent them strictly as triplets. 

Each triplet must follow this format:
["head", "relationship", "tail"]

- 'head' = the source entity
- 'relationship' = the relation between the entities
- 'tail' = the target entity

Only output valid triplets. Do not include explanations, extra text, or commentary. Make sure no duplicates.

For example:
Text:
Filipinos are Hispanics of Asia , whats problem ? La Bamba ( film ) La Bamba is 1987 American biographical film written directed by Luis Valdez that follows life career of Chicano rock n roll star Ritchie Valens . The film stars Lou Diamond Phillips as Valens , Esai Morales , Rosanna DeSoto , Elizabeth Pea , Danielle von Zerneck , Joe Pantoliano . The film depicts impact Valens career had on lives of his half-brother Bob Morales , his girlfriend Donna Ludwig rest of La Bamba ( film ) La Bamba is 1987 American biographical film written directed by Luis Valdez that follows life career of Chicano rock n roll star Ritchie Valens . The film stars Lou Diamond Phillicts impact Valens career had on lives of his half-brother Bob Morales , his girlfriend Donna Ludwig rest of his family . In 2017 , film was selected for preservation in United States National Film Registry by Library of Congress as being culturally , Phillips playing role of real life Chicano , to which Luis Valdez responded Why not ?

Output:
["Filipinos","are","Hispanics"]
["Filipinos","are","Hispanics of Asia"]
["La Bamba","is","film"]
["La Bamba","made in","1987"]
["La Bamba","is","American biographical film"]
["La Bamba","is directed by","Luis Vladez"]
["La Bamba","follows life career of","Chicano rock n roll"]
["La Bamba","follows life career of","Ritchie Valens"]
["La Bamba","stars","Lou Diamond Phillips"]
["Lou Diamond Phillips","plays","Valens in La Bamba"]
["La Bamba","stars","Esai Morales"]
["La Bamba","stars","Rosanna DeSoto"]
["La Bamba","stars","Elizabeth Pea"]
["La Bamba","stars","Danielle von Zerneck"]
["La Bamba","stars","Joe Pantoliano"]
["Valens","half-brother","Bob Morales""]
["Vales","girlfriend","Donna Ludwig"]
["La Bamba","selected for","preservation"]
["La Bamba","preserved in","2017",]
["La Bamba","preserved in","United States National Film Registry"]
["La Bamba","preserved by","Library of Congress"]"""

    user_prompt_template = """Here is the context. Kindly stick to the output format and extract all valid triplets from this:

{context}"""

    tokenizer, model = load_model(model_name, gpu_ids=gpu_ids)

    col = f'extracted_triplets_{chunks_column}_{model_name.replace("/", "_")}'
    timet = f'extraction_time_{chunks_column}_{model_name.replace("/", "_")}'
    num_col = f'num_triplets_{chunks_column}_{model_name.replace("/", "_")}'
    
    df[col] = None
    df[timet] = None
    df[num_col] = None

    for idx in tqdm(range(len(df)), desc=f"Extracting with {model_name}"):
        chunks_text = df.at[idx, chunks_column]
        chunks = [c.strip() for c in chunks_text.split('<c>') if c.strip()]
        
        all_triplets_for_row = []
        total_time = 0
        
        # Process chunks in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            
            prompts = [
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt_template.format(context=c)}]
                for c in batch_chunks
            ]

            # Batch tokenization
            batch_inputs = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            attention_mask = (batch_inputs != tokenizer.pad_token_id).long().to(model.device)


            stime = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch_inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id
                )
            batch_time = time.time() - stime
            total_time += batch_time

            # Decode and extract triplets
            input_lengths = batch_inputs.shape[1]
            generated_tokens = outputs[:, input_lengths:]  # Slice off the input tokens
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            
            for response in decoded:
                triplets = extract_triplets_from_response(response)
                all_triplets_for_row.extend(triplets)

            torch.cuda.empty_cache()
        
        # Store aggregated results for this row using JSON encoding to prevent CSV issues
        df.at[idx, col] = json.dumps(all_triplets_for_row)  # Use JSON instead of str()
        df.at[idx, timet] = total_time
        df.at[idx, num_col] = len(all_triplets_for_row)
        
        # Save checkpoint every 10 rows
        if (idx + 1) % 10 == 0:
            df.to_csv('GraphRAG_triplets_checkpoint.csv', index=False, escapechar='\\')
            print(f"Checkpoint saved at row {idx + 1}")

    gc.collect()
    return df

# ----------------- Optimized Model Pipeline -----------------
def load_model(model_name, device_map="auto", gpu_ids=None):
    print(f"Loading model: {model_name}")
    
    # Set specific GPU if provided
    if gpu_ids is not None:
        if isinstance(gpu_ids, (list, tuple)):
            device_map = {i: f"cuda:{gpu_id}" for i, gpu_id in enumerate(gpu_ids)}
        else:
            device_map = f"cuda:{gpu_ids}"
        print(f"Using GPU(s): {gpu_ids}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map
    ).eval()
    return tokenizer, model

if __name__ == "__main__":

    df = pd.read_csv('NQ_test.csv')

    tqdm.pandas(desc="Counting tokens")
    df['positive_token_count'] = df['combined_positive'].progress_apply(count_context_tokens)
    df['negative_token_count'] = df['combined_negative'].progress_apply(count_context_tokens)
    df['hard_negative_token_count'] = df['combined_hard_negative'].progress_apply(count_context_tokens)

    tqdm.pandas(desc="Preprocessing text")
    df['preprocessed_combined_positive'] = df['combined_positive'].progress_apply(preprocess_text)
    df['preprocessed_combined_negative'] = df['combined_negative'].progress_apply(preprocess_text)
    df['preprocessed_combined_hard_negative'] = df['combined_hard_negative'].progress_apply(preprocess_text)

    tqdm.pandas(desc="Creating chunks")
    df['chunks_from_preprocessed_combined_positive'] = df['preprocessed_combined_positive'].progress_apply(create_chunks)
    df['chunks_from_preprocessed_combined_negative'] = df['preprocessed_combined_negative'].progress_apply(create_chunks)
    df['chunks_from_preprocessed_combined_hard_negative'] = df['preprocessed_combined_hard_negative'].progress_apply(create_chunks)


    model_id = "microsoft/Phi-4-mini-instruct"


    gpu_ids = None  # Change this to specify your desired GPU(s)
    df= extract_triplets_from_chunks(df, model_id, 'chunks_from_preprocessed_combined_positive', batch_size=4, gpu_ids=gpu_ids)
    print("Triplet extraction completed for positive chunks!!")
    df= extract_triplets_from_chunks(df, model_id, 'chunks_from_preprocessed_combined_negative', batch_size=4, gpu_ids=gpu_ids)
    print("Triplet extraction completed for negative chunks!!")
    df= extract_triplets_from_chunks(df, model_id, 'chunks_from_preprocessed_combined_hard_negative', batch_size=4, gpu_ids=gpu_ids)
    print("Triplet extraction completed for hard negative chunks!!")
    df.to_csv('GraphRAG_triplets_nq.csv', index=False)
    print("Triplet extraction completed!!")

