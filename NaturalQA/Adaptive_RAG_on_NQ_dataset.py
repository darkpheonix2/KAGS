#!/usr/bin/env python3
"""
Adaptive RAG Pipeline with Multi-Model Support
Supports: Flan-T5-XL, Mistral-7B-v0.1, Llama-3.2-3B
"""

import os
import time
import ast
import gc
import pandas as pd
import numpy as np
import tiktoken
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# LangChain imports
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline
)

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Hugging Face login
from huggingface_hub import login

import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the RAG pipeline"""
    # API Token
    HF_TOKEN: str = ""  # loaded from HF_TOKEN env via setup_hf_login()
    
    # Paths
    DATA_PATH: str = "NQ_test.csv"
    PERSIST_DIRECTORY: str = "chroma_db_all-MiniLM-L6-v2"
    
    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    ROUTER_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # GPU Configuration
    # Set AVAILABLE_GPUS to the GPU IDs you want to use
    # Example: [0, 1, 3] if GPU 2 is not available
    AVAILABLE_GPUS: List[int] = None
    
    # Available generator models
    GENERATOR_MODELS: Dict[str, Dict[str, Any]] = None
    
    # Chunking parameters
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 64
    
    # Retrieval parameters
    K_RETRIEVAL: int = 3
    MAX_COT_ITERATIONS: int = 3
    
    # Processing parameters
    BATCH_SIZE: int = 100
    RANDOM_STATE: int = 42
    
    # Memory management
    CLEAR_GPU_CACHE: bool = True  # Clear GPU cache after each model
    
    def __post_init__(self):
        # Default to all 4 GPUs if not specified
        if self.AVAILABLE_GPUS is None:
            self.AVAILABLE_GPUS = [0, 1, 2, 3]
        
        # Allocate GPUs dynamically based on available GPUs
        self._allocate_gpus()
    
    def _allocate_gpus(self):
        """Dynamically allocate GPUs to models"""
        # GPU 0 is always for embedding and router
        embedding_gpu = self.AVAILABLE_GPUS[0] if len(self.AVAILABLE_GPUS) > 0 else 0
        
        # Allocate remaining GPUs to generator models
        generator_gpus = self.AVAILABLE_GPUS[1:] if len(self.AVAILABLE_GPUS) > 1 else self.AVAILABLE_GPUS
        
        # If we have fewer GPUs than models, reuse GPUs (models will be run sequentially)
        gpu_mapping = {
            'flan-t5-xl': generator_gpus[0] if len(generator_gpus) > 0 else embedding_gpu,
            'mistral-7b': generator_gpus[1] if len(generator_gpus) > 1 else generator_gpus[0],
            'llama-3.2-3b': generator_gpus[2] if len(generator_gpus) > 2 else generator_gpus[0],
        }
        
        self.GENERATOR_MODELS = {
            'flan-t5-xl': {
                'name': 'google/flan-t5-xl',
                'type': 'seq2seq',
                'device': f'cuda:{gpu_mapping["flan-t5-xl"]}'
            },
            'mistral-7b': {
                'name': 'mistralai/Mistral-7B-v0.1',
                'type': 'causal',
                'device': f'cuda:{gpu_mapping["mistral-7b"]}'
            },
            'llama-3.2-3b': {
                'name': 'meta-llama/Llama-3.2-3B',
                'type': 'causal',
                'device': f'cuda:{gpu_mapping["llama-3.2-3b"]}'
            }
        }
        
        print(f"\n{'='*80}")
        print("GPU Allocation:")
        print(f"{'='*80}")
        print(f"Available GPUs: {self.AVAILABLE_GPUS}")
        print(f"Embedding & Router: cuda:{embedding_gpu}")
        for model_key, model_info in self.GENERATOR_MODELS.items():
            print(f"{model_key}: {model_info['device']}")
        print(f"{'='*80}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class GPUMemoryManager:
    """Manage GPU memory efficiently"""
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def clear_model(model, tokenizer=None):
        """Clear model from GPU memory"""
        if model is not None:
            try:
                # Move model to CPU
                if hasattr(model, 'cpu'):
                    model.cpu()
                
                # Delete model
                del model
            except Exception as e:
                print(f"Warning: Error while clearing model: {e}")
        
        if tokenizer is not None:
            try:
                del tokenizer
            except Exception as e:
                print(f"Warning: Error while clearing tokenizer: {e}")
        
        # Clear cache
        GPUMemoryManager.clear_cache()
    
    @staticmethod
    def print_gpu_memory():
        """Print GPU memory usage for all available GPUs"""
        if torch.cuda.is_available():
            print("\nGPU Memory Status:")
            print("-" * 80)
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
            print("-" * 80)
    
    @staticmethod
    def get_available_memory(device_id: int) -> float:
        """Get available memory on a specific GPU in GB"""
        if torch.cuda.is_available():
            return (torch.cuda.get_device_properties(device_id).total_memory - 
                    torch.cuda.memory_allocated(device_id)) / 1024**3
        return 0.0


class TokenCounter:
    """Utility class for counting tokens"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))


class ContextParser:
    """Parse and flatten context from various formats"""
    
    @staticmethod
    def parse_context(context):
        """Parse context whether it's string or list"""
        if isinstance(context, str):
            try:
                return ast.literal_eval(context)
            except (ValueError, SyntaxError):
                return context
        return context
    
    @staticmethod
    def flatten_context(context) -> str:
        """Flatten context into a single string"""
        parsed_context = ContextParser.parse_context(context)
        
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
        
        return str(parsed_context) if not isinstance(parsed_context, str) else parsed_context
    
    @staticmethod
    def count_context_tokens(context, token_counter: TokenCounter) -> int:
        """Count total tokens in nested context"""
        total_tokens = 0
        parsed_context = ContextParser.parse_context(context)
        
        if isinstance(parsed_context, list):
            for item in parsed_context:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title, text_snippets = item[0], item[1]
                    
                    if isinstance(title, str):
                        total_tokens += token_counter.count(title)
                    
                    if isinstance(text_snippets, list):
                        for snippet in text_snippets:
                            if isinstance(snippet, str):
                                total_tokens += token_counter.count(snippet)
                    elif isinstance(text_snippets, str):
                        total_tokens += token_counter.count(text_snippets)
                else:
                    if isinstance(item, str):
                        total_tokens += token_counter.count(item)
        elif isinstance(parsed_context, str):
            total_tokens += token_counter.count(parsed_context)
        
        return total_tokens


class TextChunker:
    """Create text chunks with overlap"""
    
    def __init__(self, token_counter: TokenCounter, chunk_size: int = 256, overlap: int = 64):
        self.token_counter = token_counter
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_chunks(self, text: str) -> str:
        """Create chunks with overlap, preserving sentence boundaries"""
        sentences = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = self.token_counter.count(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Create overlap
                overlap_chunk = []
                overlap_length = 0
                j = len(current_chunk) - 1
                
                while j >= 0 and overlap_length < self.overlap:
                    overlap_sentence = current_chunk[j]
                    overlap_sentence_length = self.token_counter.count(overlap_sentence)
                    
                    if overlap_length + overlap_sentence_length <= self.overlap:
                        overlap_chunk.insert(0, overlap_sentence)
                        overlap_length += overlap_sentence_length
                        j -= 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
            i += 1
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return '<c>'.join(chunks)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Process and prepare data for RAG"""
    
    def __init__(self, config: Config):
        self.config = config
        self.token_counter = TokenCounter()
        self.chunker = TextChunker(self.token_counter, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    def load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare dataset"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.config.RANDOM_STATE).reset_index(drop=True)
        
        # Count tokens
        print("Counting context tokens...")
        df['context_token_count'] = df['combined_positive'].apply(
            lambda x: ContextParser.count_context_tokens(x, self.token_counter)
        )
        
        # Flatten context
        print("Flattening contexts...")
        df['raw_context'] = df['combined_positive'].apply(ContextParser.flatten_context)
        
        # Create chunks
        print("Creating chunks...")
        df['chunks'] = df['raw_context'].apply(self.chunker.create_chunks)
        
        print(f"Processed {len(df)} documents")
        print(f"Max token count: {df['context_token_count'].max()}")
        
        return df


# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStoreManager:
    """Manage ChromaDB vector store"""
    
    def __init__(self, config: Config):
        self.config = config
        self.token_counter = TokenCounter()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cuda:0"}
        )
    
    def check_exists(self) -> bool:
        """Check if vector store exists"""
        if not os.path.exists(self.config.PERSIST_DIRECTORY):
            return False
        
        chroma_files = ['chroma.sqlite3', 'index']
        return any(
            os.path.exists(os.path.join(self.config.PERSIST_DIRECTORY, file))
            for file in chroma_files
        )
    
    def load_existing(self) -> Chroma:
        """Load existing vector store"""
        print(f"Loading existing ChromaDB from {self.config.PERSIST_DIRECTORY}...")
        vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.config.PERSIST_DIRECTORY
        )
        count = vectorstore._collection.count()
        print(f"Loaded {count} documents")
        return vectorstore
    
    def create_new(self, df: pd.DataFrame) -> Chroma:
        """Create new vector store from DataFrame"""
        print("Creating new ChromaDB...")
        
        documents = []
        metadatas = []
        ids = []
        
        print("Processing documents...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            chunks = row['chunks'].split('<c>')
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk.strip())
                    
                    metadatas.append({
                        'source_index': idx,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'token_count': self.token_counter.count(chunk.strip())
                    })
                    
                    ids.append(f"doc_{idx}_chunk_{chunk_idx}")
        
        print(f"Total chunks: {len(documents)}")
        print(f"Average chunks per document: {len(documents)/len(df):.2f}")
        
        vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.config.PERSIST_DIRECTORY
        )
        
        print("Adding documents to ChromaDB...")
        start_time = time.time()
        
        for i in tqdm(range(0, len(documents), self.config.BATCH_SIZE), desc="Adding batches"):
            end_idx = min(i + self.config.BATCH_SIZE, len(documents))
            
            vectorstore.add_texts(
                texts=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.2f} minutes")
        print(f"Rate: {len(documents)/elapsed:.1f} docs/sec")
        
        return vectorstore
    
    def get_or_create(self, df: pd.DataFrame = None) -> Chroma:
        """Get existing or create new vector store"""
        if self.check_exists():
            return self.load_existing()
        elif df is not None:
            return self.create_new(df)
        else:
            raise ValueError("Vector store doesn't exist and no DataFrame provided")


# ============================================================================
# MODEL MANAGERS
# ============================================================================

class RouterModel:
    """Router model for question classification"""
    
    def __init__(self, config: Config):
        self.config = config
        print(f"Loading router model: {config.ROUTER_MODEL}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.ROUTER_MODEL)
        model = AutoModelForCausalLM.from_pretrained(config.ROUTER_MODEL)
        
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            device_map="cuda:0"
        )
        
        print("Router model loaded")
    
    def route(self, query: str) -> str:
        """Classify query into LLM, Single-hop, or Multi-hop"""
        prompt = f"""You are a question classifier. Classify the question into one of three categories:

LLM – General knowledge question (no retrieval needed)
Single-hop – Requires retrieving a single fact
Multi-hop – Requires combining multiple pieces of information

Return only: LLM, Single-hop, or Multi-hop

Examples:
Question: What is the capital of France?
Answer: LLM

Question: What is the average temperature in Berlin in June?
Answer: Single-hop

Question: When was the iPhone 12 released and how did its sales compare to the iPhone 11?
Answer: Multi-hop

Question: {query}
Answer:"""
        
        output = self.pipeline(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text']
        answer = output.split("Answer:")[-1].strip().split("\n")[0]
        answer = answer.replace('"', '').replace("'", "").strip()
        
        return answer


class GeneratorModel:
    """Generator model for answer generation"""
    
    def __init__(self, model_key: str, config: Config):
        self.model_key = model_key
        self.config = config
        model_config = config.GENERATOR_MODELS[model_key]
        
        print(f"Loading generator model: {model_config['name']}...")
        
        self.model_name = model_config['name']
        self.model_type = model_config['type']
        self.device = model_config['device']
        
        # Print available memory before loading
        device_id = int(self.device.split(':')[1])
        available_mem = GPUMemoryManager.get_available_memory(device_id)
        print(f"Available memory on {self.device}: {available_mem:.2f}GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map=self.device
        )
        
        if self.model_type == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            ).to(self.device)
        else:  # causal
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device
            )
        
        print(f"Generator model loaded on {self.device}")
        GPUMemoryManager.print_gpu_memory()
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        if self.model_type == 'seq2seq':
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )
        else:  # causal
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        if self.model_type == 'causal':
            # For causal models, skip the input prompt
            output = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return output.strip()
    
    def cleanup(self):
        """Clean up model and free GPU memory"""
        print(f"\nCleaning up {self.model_key}...")
        GPUMemoryManager.clear_model(self.model, self.tokenizer)
        self.model = None
        self.tokenizer = None
        print(f"✅ {self.model_key} cleaned up")
        GPUMemoryManager.print_gpu_memory()


# ============================================================================
# ANSWER GENERATORS
# ============================================================================

class AnswerGenerators:
    """Answer generation strategies"""
    
    def __init__(self, generator_model: GeneratorModel, vectorstore: Chroma, config: Config):
        self.generator = generator_model
        self.vectorstore = vectorstore
        self.config = config
    
    def direct(self, query: str) -> str:
        """Generate direct answer without retrieval"""
        prompt = f"User question: {query}\nAnswer:"
        return self.generator.generate(prompt, max_new_tokens=100)
    
    def single_hop(self, query: str, context: str = None) -> str:
        """Generate answer with single retrieval"""
        if context is None:
            docs = self.vectorstore.similarity_search(query, k=self.config.K_RETRIEVAL)
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        prompt = (
            "You are an expert at answering questions based on context. "
            "Answer the question using only the provided context. "
            "If you cannot answer based on the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        return self.generator.generate(prompt, max_new_tokens=150)
    
    def multi_hop(self, query: str) -> Dict[str, Any]:
        """Generate answer with chain-of-thought retrieval"""
        return self._chain_of_thought_retriever(query)
    
    def _chain_of_thought_retriever(self, query: str) -> Dict[str, Any]:
        """Multi-hop retrieval with chain-of-thought reasoning"""
        
        def create_cot_prompt(query: str, context: str) -> str:
            return (
                "Answer the question by reasoning step-by-step using the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Instructions:\n"
                "1. Break down the question into logical steps\n"
                "2. Use the context to find information for each step\n"
                "3. Conclude with: 'So the answer is: [your answer]'\n"
                "4. If you need more information, state what is needed\n\n"
                "Step-by-step reasoning:"
            )
        
        def extract_answer(response: str) -> Dict[str, Any]:
            response = response.strip()
            
            if "So the answer is:" in response:
                answer = response.split("So the answer is:")[-1].strip()
                return {"complete": True, "answer": answer, "reasoning": response}
            
            # Check if it's a short direct answer
            response_lower = response.lower()
            incomplete_phrases = [
                "need more information", "cannot determine", "missing",
                "unclear", "require", "cannot find", "not enough"
            ]
            
            if len(response.split()) <= 5 and not any(p in response_lower for p in incomplete_phrases):
                return {
                    "complete": True,
                    "answer": response,
                    "reasoning": f"The answer is: {response}. So the answer is: {response}"
                }
            
            return {"complete": False, "answer": None, "reasoning": response}
        
        all_contexts = []
        reasoning_steps = []
        current_query = query
        
        for iteration in range(1, self.config.MAX_COT_ITERATIONS + 1):
            # Retrieve documents
            try:
                docs = self.vectorstore.similarity_search(current_query, k=self.config.K_RETRIEVAL)
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                
                all_contexts.append({
                    "iteration": iteration,
                    "query": current_query,
                    "context": context
                })
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Retrieval failed at iteration {iteration}: {e}",
                    "final_answer": None
                }
            
            # Generate reasoning
            cot_prompt = create_cot_prompt(query, context)
            try:
                reasoning = self.generator.generate(cot_prompt, max_new_tokens=200)
                reasoning_steps.append({
                    "iteration": iteration,
                    "query": current_query,
                    "reasoning": reasoning
                })
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Generation failed at iteration {iteration}: {e}",
                    "final_answer": None
                }
            
            # Check completeness
            result = extract_answer(reasoning)
            
            if result["complete"]:
                return {
                    "success": True,
                    "final_answer": result["answer"],
                    "complete_reasoning": result["reasoning"],
                    "iterations_used": iteration,
                    "all_contexts": all_contexts
                }
            
            # Generate follow-up query if incomplete
            if iteration < self.config.MAX_COT_ITERATIONS:
                # Extract key terms from reasoning for next query
                words = reasoning.split()
                key_terms = [w.strip('.,?!') for w in words if len(w) > 3 and w.isalpha()]
                current_query = " ".join(key_terms[:5]) if key_terms else query
        
        return {
            "success": False,
            "final_answer": None,
            "error": f"Could not complete answer in {self.config.MAX_COT_ITERATIONS} iterations",
            "all_contexts": all_contexts
        }


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

class RoutingRAGState(TypedDict):
    """State for routing RAG workflow"""
    question: str
    routing_result: str
    answer: str
    answer_type: str
    path_taken: List[str]
    error_message: str
    context_docs: List[Document]
    retrieved_chunks: Optional[List[str]]  # Actual text chunks retrieved


class RoutingRAGWorkflow:
    """LangGraph workflow for adaptive RAG"""
    
    def __init__(
        self,
        router: RouterModel,
        generators: AnswerGenerators,
        config: Config
    ):
        self.router = router
        self.generators = generators
        self.config = config
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        def route_question_node(state: RoutingRAGState) -> RoutingRAGState:
            try:
                routing_decision = self.router.route(state["question"])
                return {
                    **state,
                    "routing_result": routing_decision,
                    "path_taken": state["path_taken"] + [f"routed -> {routing_decision}"]
                }
            except Exception as e:
                return {
                    **state,
                    "routing_result": "Single-hop",
                    "error_message": f"Routing error: {e}",
                    "path_taken": state["path_taken"] + ["routing_error"]
                }
        
        def direct_answer_node(state: RoutingRAGState) -> RoutingRAGState:
            try:
                answer = self.generators.direct(state["question"])
                return {
                    **state,
                    "answer": answer,
                    "answer_type": "direct",
                    "retrieved_chunks": None,  # No retrieval for direct answers
                    "path_taken": state["path_taken"] + ["direct_answer"]
                }
            except Exception as e:
                return {
                    **state,
                    "answer": f"Error generating direct answer: {e}",
                    "answer_type": "direct_error",
                    "error_message": f"Direct answer error: {e}",
                    "retrieved_chunks": None,
                    "path_taken": state["path_taken"] + ["direct_error"]
                }
        
        def single_hop_node(state: RoutingRAGState) -> RoutingRAGState:
            try:
                docs = self.generators.vectorstore.similarity_search(
                    state["question"], 
                    k=self.config.K_RETRIEVAL
                )
                
                # Extract chunk texts
                retrieved_chunks = [doc.page_content for doc in docs]
                
                # Generate answer
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}" 
                    for i, doc in enumerate(docs)
                ])
                answer = self.generators.generator.generate(
                    (
                        "You are an expert at answering questions based on context. "
                        "Answer the question using only the provided context. "
                        "If you cannot answer based on the context, say so.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {state['question']}\n"
                        "Answer:"
                    ),
                    max_new_tokens=150
                )
                
                return {
                    **state,
                    "answer": answer,
                    "answer_type": "single-hop",
                    "context_docs": docs,
                    "retrieved_chunks": retrieved_chunks,
                    "path_taken": state["path_taken"] + ["single_hop"]
                }
            except Exception as e:
                return {
                    **state,
                    "answer": f"Error in single-hop: {e}",
                    "answer_type": "single-hop_error",
                    "error_message": f"Single-hop error: {e}",
                    "retrieved_chunks": None,
                    "path_taken": state["path_taken"] + ["single_hop_error"]
                }
        
        def multi_hop_node(state: RoutingRAGState) -> RoutingRAGState:
            try:
                result = self.generators.multi_hop(state["question"])
                
                # Extract all chunks from all iterations
                retrieved_chunks = []
                if result.get("all_contexts"):
                    for context_info in result["all_contexts"]:
                        # Extract individual chunks from the formatted context
                        context_text = context_info["context"]
                        # Split by "Document N:" pattern and extract chunks
                        chunks = []
                        for doc_section in context_text.split("Document ")[1:]:
                            # Get the text after the number and colon
                            chunk = doc_section.split(":\n", 1)[-1].strip()
                            if chunk:
                                chunks.append(chunk)
                        retrieved_chunks.extend(chunks)
                
                # If no chunks found, set to None
                if not retrieved_chunks:
                    retrieved_chunks = None
                
                return {
                    **state,
                    "answer": result.get("final_answer", "Could not determine answer"),
                    "answer_type": "multi-hop",
                    "retrieved_chunks": retrieved_chunks,
                    "path_taken": state["path_taken"] + ["multi_hop"]
                }
            except Exception as e:
                return {
                    **state,
                    "answer": f"Error in multi-hop: {e}",
                    "answer_type": "multi-hop_error",
                    "error_message": f"Multi-hop error: {e}",
                    "retrieved_chunks": None,
                    "path_taken": state["path_taken"] + ["multi_hop_error"]
                }
        
        def final_node(state: RoutingRAGState) -> RoutingRAGState:
            return {**state, "path_taken": state["path_taken"] + ["complete"]}
        
        def decide_strategy(state: RoutingRAGState) -> str:
            routing = state["routing_result"].lower()
            if "llm" in routing:
                return "direct"
            elif "single" in routing:
                return "single_hop"
            else:
                return "multi_hop"
        
        # Build graph
        workflow = StateGraph(RoutingRAGState)
        
        workflow.add_node("route", route_question_node)
        workflow.add_node("direct", direct_answer_node)
        workflow.add_node("single_hop", single_hop_node)
        workflow.add_node("multi_hop", multi_hop_node)
        workflow.add_node("final", final_node)
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            decide_strategy,
            {
                "direct": "direct",
                "single_hop": "single_hop",
                "multi_hop": "multi_hop"
            }
        )
        
        workflow.add_edge("direct", "final")
        workflow.add_edge("single_hop", "final")
        workflow.add_edge("multi_hop", "final")
        workflow.add_edge("final", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run workflow for a question"""
        initial_state = RoutingRAGState(
            question=question,
            routing_result="",
            answer="",
            answer_type="",
            path_taken=[],
            error_message="",
            context_docs=[],
            retrieved_chunks=None
        )
        
        return self.graph.invoke(initial_state)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class AdaptiveRAGPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Login to HuggingFace
        print("Logging in to HuggingFace...")
        if not config.HF_TOKEN:
            from db_config import get_hf_token
            config.HF_TOKEN = get_hf_token()
        login(config.HF_TOKEN)

        # Set environment
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.HF_TOKEN
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        # Load data
        self.data_processor = DataProcessor(config)
        self.df = self.data_processor.load_and_prepare_data(config.DATA_PATH)
        
        # Setup vector store
        self.vs_manager = VectorStoreManager(config)
        self.vectorstore = self.vs_manager.get_or_create(self.df)
        
        # Load router
        self.router = RouterModel(config)
    
    def run_experiments(self, model_keys: List[str] = None):
        """Run experiments with specified models"""
        if model_keys is None:
            model_keys = list(self.config.GENERATOR_MODELS.keys())
        
        results = {}
        
        for model_idx, model_key in enumerate(model_keys):
            print(f"\n{'='*80}")
            print(f"Running experiment {model_idx + 1}/{len(model_keys)}: {model_key}")
            print(f"{'='*80}\n")
            
            # Print initial GPU status
            print("Initial GPU status:")
            GPUMemoryManager.print_gpu_memory()
            
            # Load generator model
            generator_model = GeneratorModel(model_key, self.config)
            answer_generators = AnswerGenerators(generator_model, self.vectorstore, self.config)
            
            # Create workflow
            workflow = RoutingRAGWorkflow(self.router, answer_generators, self.config)
            
            # Process questions
            answers = []
            paths = []
            times = []
            retrieved_chunks_list = []
            
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Processing with {model_key}"):
                question = row['question']
                
                start_time = time.time()
                result = workflow.run(question)
                elapsed = time.time() - start_time
                
                answers.append(result.get('answer', None))
                paths.append(result.get('path_taken', None))
                times.append(elapsed)
                retrieved_chunks_list.append(result.get('retrieved_chunks', None))
            
            # Save results
            result_df = self.df.copy()
            result_df['predicted_answer'] = answers
            result_df['path_taken'] = paths
            result_df['time'] = times
            result_df['retrieved_chunks'] = retrieved_chunks_list
            
            output_file = f"Adaptive_RAG_{model_key}_results.csv"
            result_df.to_csv(output_file, index=False)
            
            print(f"\n✅ Results saved to {output_file}")
            print(f"Average time per question: {np.mean(times):.2f}s")
            
            results[model_key] = {
                'df': result_df,
                'avg_time': np.mean(times),
                'output_file': output_file
            }
            
            # CRITICAL: Clean up GPU memory after this model
            if self.config.CLEAR_GPU_CACHE:
                print(f"\n{'='*80}")
                print(f"Cleaning up {model_key} to free GPU memory...")
                print(f"{'='*80}")
                
                # Clean up the generator model
                generator_model.cleanup()
                
                # Delete references
                del generator_model
                del answer_generators
                del workflow
                
                # Force garbage collection and clear cache
                GPUMemoryManager.clear_cache()
                
                print("\nGPU status after cleanup:")
                GPUMemoryManager.print_gpu_memory()
                
                # Small delay to ensure cleanup is complete
                time.sleep(2)
        
        return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Configuration
    config = Config()
    
    # Create pipeline
    pipeline = AdaptiveRAGPipeline(config)
    
    # Run experiments with all models
    results = pipeline.run_experiments(['flan-t5-xl', 'mistral-7b', 'llama-3.2-3b'])
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for model_key, result in results.items():
        print(f"\n{model_key}:")
        print(f"  Average time: {result['avg_time']:.2f}s")
        print(f"  Output file: {result['output_file']}")


if __name__ == "__main__":
    main()