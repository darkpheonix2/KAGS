import pandas as pd
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = "cuda"   # or "cpu"
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"  # or just `device_map=device` if you want to manually pin
)

def answer_generator_single_hop(query, context, max_new_tokens=1024):
    """
    Generate an answer using Qwen2.5-7B-Instruct based on context + question.
    """

    # Use the chat / instruction template
    prompt = (
        "You are an expert at answering the question based on the context. "
        "If the context does not contain the answer, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        early_stopping=True
    )

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract the actual answer part
    answer = output_text.split("Answer:")[-1].strip()

    return answer


# def answer_generator_single_hop(query, context, max_new_tokens=1024):
#     """
#     Generate an answer using a seq2seq model (e.g., FLAN-T5) based on the given context and question.
#     """
#     # Build the model prompt in instruction format (T5-style)
#     prompt = (
#         "You are an expert at answering the question based on the context. "
#         "If the context does not contain the answer, say so clearly.\n\n"
#         f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     )

#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

#     # Generate output (T5 generates sequence-to-sequence output)
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         # num_beams=4,            # optional: better quality
#         early_stopping=True
#     )

#     # Decode and clean up
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     answer = output_text.strip().replace('"', '').replace("'", "")

#     return answer

if __name__=='__main__':

    df = pd.read_csv('Test_kgr.csv')
    df['predicted_answer'] = df.apply(
    lambda row: answer_generator_single_hop(row['question'], row['retrieved_context']),
    axis=1
)
    df.to_csv('Ouput_Data/KAG_final_output_GPT_20B.csv',index=False)
    # Load model and tokenizer
    

