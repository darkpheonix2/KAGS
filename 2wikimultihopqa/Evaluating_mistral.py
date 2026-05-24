#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import metrics

# Set up logging to file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and file
sys.stdout = Logger('Hybrid_RAG_mistral.txt')

def safe_float_mean(series):
    """Calculate mean of a series, safely converting to float"""
    floats = []
    for val in series:
        try:
            floats.append(float(val))
        except Exception:
            print("Non-convertible value:", val)
    if floats:
        return sum(floats) / len(floats)
    else: 
        return float('nan')

def main():
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,0,2,1"
    
    print("=" * 80)
    print("RAG Evaluation Pipeline")
    print("=" * 80)
    
    # Check CUDA availability
    print(f"\nCUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Load data
    print("\nLoading data from CSV...")
    df = pd.read_csv('Hybrid_RAG_output_mistral.csv')
    print(f"Initial data shape: {df.shape}")
    print(f"\nFirst few questions:")
    print(df['question'].head(3).to_string())
    
    # Filter out rows with NaN predicted_answer
    initial_rows = len(df)
    df = df[~df['predicted_answer_mistral'].isna()]
    print(f"\nRows after filtering NaN predicted_answer: {len(df)} (removed {initial_rows - len(df)} rows)")
    
    # Calculate average retrieval time
    # print(f"\nAverage time of retrieval: {df['time_of_retrieval'].mean():.4f}")
    
    # Load LLM and tokenizer
    print("\nLoading LLM and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map='auto'
    )
    llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map='auto'
    )
    print("Model loaded successfully!")
    
    # Initialize metric columns
    print("\nInitializing metric columns...")
    df['faithfulness'] = None
    df['answer_relevance'] = None
    df['precision'] = None
    df['recall'] = None
    df['f1_score'] = None
    df['Token_level_accuracy'] = None
    df['Fuzzy_based_accuracy'] = None
    df['Meaning_based_accuracy'] = None
    df['Meaning_based_accuracy_controlled'] = None
    
    # Process each row
    print("\nCalculating metrics for each row...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing answers"):
        question = row['question']
        actual_answer = row['answer']
        predicted_answer = row['predicted_answer_mistral']
        retrieved_context = row['hybrid_context']
        actual_relevant_docs = row['supporting_context']
        
        # Calculate faithfulness
        try:
            faithfulness = metrics.Faithfulness(
                question, predicted_answer, llm, tokenizer, retrieved_context
            ).faithfulness()
            df.at[idx, 'faithfulness'] = faithfulness
        except Exception as e:
            print(f"\nException at row {idx} calculating faithfulness: {e}")
        
        # Calculate answer relevance
        try:
            ar = metrics.Relevance(
                question, predicted_answer, llm, tokenizer
            ).answer_relevance()
            df.at[idx, 'answer_relevance'] = ar
        except Exception as e:
            print(f"\nException at row {idx} calculating answer relevance: {e}")
        
        # Calculate retrieval metrics
        try:
            if retrieved_context is not None:   
                rm = metrics.Retrieval_metrics(
                    retrieved_context, actual_relevant_docs
                ).calculate_retrieval_metrics()
                df.at[idx, 'precision'] = rm['precision']
                df.at[idx, 'recall'] = rm['recall']
                df.at[idx, 'f1_score'] = rm['f1_score']
        except Exception as e:
            print(f"\nException at row {idx} calculating retrieval metrics: {e}")
        
        # Calculate accuracy metrics
        try:
            rm = metrics.Accuracy(
                llm, tokenizer, predicted_answer, actual_answer
            ).evaluate_rag_accuracy()
            df.at[idx, 'Token_level_accuracy'] = rm['Token_level_accuracy']
            df.at[idx, 'Fuzzy_based_accuracy'] = rm['Fuzzy_based_accuracy']
            df.at[idx, 'Meaning_based_accuracy'] = rm['Meaning_based_accuracy']
            df.at[idx, 'Meaning_based_accuracy_controlled'] = rm['Meaning_based_accuracy_controlled']
        except Exception as e:
            print(f"\nException at row {idx} calculating accuracy: {e}")
        
        # Save progress after each row
        df.to_csv("Output_Data/Hybrid_RAG_Mistral_metrics.csv", index=False)
    
    print("\n" + "=" * 80)
    print("Metric calculation complete!")
    print("=" * 80)
    
    # Clean and prepare data for final statistics
    print("\nCleaning data for final statistics...")
    df_ = df[~df.predicted_answer_mistral.isna()].copy()
    df_['Meaning_based_accuracy'] = df_['Meaning_based_accuracy'].replace(
        '0 (no relation)', 0.05
    )
    df_['Meaning_based_accuracy'] = df_['Meaning_based_accuracy'].replace(
        '', 0.0
    )
    df_ = df_.fillna(0)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Average Metrics")
    print("=" * 80)
    print(f"Faithfulness:                          {df_['faithfulness'].mean():.6f}")
    print(f"Answer Relevance:                      {df_['answer_relevance'].mean():.6f}")
    print(f"Precision:                             {df_['precision'].mean():.6f}")
    print(f"Recall:                                {df_['recall'].mean():.6f}")
    print(f"F1 Score:                              {df_['f1_score'].mean():.6f}")
    print(f"Token Level Accuracy:                  {df_['Token_level_accuracy'].mean():.6f}")
    print(f"Fuzzy Based Accuracy:                  {df_['Fuzzy_based_accuracy'].mean():.6f}")
    print(f"Meaning Based Accuracy:                {df_['Meaning_based_accuracy'].astype('float64').mean():.6f}")
    print(f"Meaning Based Accuracy (Controlled):   {df_['Meaning_based_accuracy_controlled'].astype('float64').mean():.6f}")
    print("=" * 80)
    
    print("\nResults saved to: Output_Data/Hybrid_RAG_Mistral_metrics.csv")
    print("Log saved to: Hybrid_RAG_evaluation_results_mistral.txt")

if __name__ == "__main__":
    main()

