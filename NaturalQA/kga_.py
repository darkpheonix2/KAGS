import pandas as pd
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Remove the CUDA_VISIBLE_DEVICES restriction or set it to allow multiple GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Remove this line

# Load tokenizer (tokenizer doesn't need device_map)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load model with automatic multi-GPU distribution
llm = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",  # Changed from 'cuda:0' to 'auto'
    torch_dtype=torch.float16,  # Optional: use fp16 to save memory
)

def answer_generator_single_hop(query, context):
    # Prepare the prompt for the model
    prompt = (
        "You are an expert at answering the question just based on the context. "
        "Given the context, answer the user question. If you cannot answer the question based on context, "
        "state properly that you cannot answer the question.\n\n"
        f"Context:\n{context}\n\nUser question: {query}\nAnswer:"
    )

    # Tokenize - inputs will automatically go to the right device
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model's first layer
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
    
    pad_token_id = tokenizer.eos_token_id

    # Generate with attention mask
    output_ids = llm.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pad_token_id=pad_token_id,
        max_new_tokens=100  # Optional: limit response length
    )
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the answer after "Answer:"
    if "Answer:" in output:
        answer = output.split("Answer:")[-1].strip().split("\n")[0]
    else:
        answer = output.strip().split("\n")[-1]
    answer = answer.replace('"', '').replace("'", "")
    return answer

if __name__=='__main__':

    df = pd.read_csv('NQ_test_with_retrieval.csv')
    df['predicted_answer'] = df.apply(
    lambda row: answer_generator_single_hop(row['question'], row['retrieved_context']),
    axis=1
)
    df.to_csv('KAG_final_output_Mistral.csv',index=False)
    # Load model and tokenizer
    print('KGA completed!!')
    