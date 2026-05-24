import pandas as pd
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = "cuda"   # or "cpu"

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl",device_map=device)
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)


# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
# llm = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-4-mini-instruct",
#     device_map=device
# )

# def answer_generator_single_hop(query, context, max_new_tokens=1024):
#     """
#     Generate an answer using Phi-4-mini-instruct based on context + question.
#     """

#     prompt = (
#         "You are an expert at answering the question based on the context. "
#         "If the context does not contain the answer, clearly say so.\n\n"
#         f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
#     )

#     # Tokenize
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

#     # Generate
#     output_ids = llm.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         early_stopping=True
#     )

#     # Decode
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     answer = output_text.split("Answer:")[-1].strip()

#     return answer

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


# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
# llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B",device_map = 'cuda:0')
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",device_map = 'cuda:0')

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1",device_map = 'cuda:0')
# llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",device_map = 'cuda:0')

def answer_generator_single_hop(query, context):
    # Use the globally loaded tokenizer and llm from file_context_1
    # Prepare the prompt for the model
    prompt = (
        "You are an expert at answering the question just based on the context. "
        "Given the context, answer the user question. If you cannot answer the question based on context, "
        "state properly that you cannot answer the question.\n\n"
        f"Context:\n{context}\n\nUser question: {query}\nAnswer:"
    )

    # Tokenize and generate, passing attention_mask for reliable results
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda:0")
    attention_mask = inputs.attention_mask.to("cuda:0")

    pad_token_id = tokenizer.eos_token_id

    output_ids = llm.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the answer after "Answer:"
    if "Answer:" in output:
        answer = output.split("Answer:")[-1].strip().split("\n")[0]
    else:
        answer = output.strip().split("\n")[-1]
    answer = answer.replace('"', '').replace("'", "")
    return answer


if __name__=='__main__':

    df = pd.read_csv('KAG_test_with_retrieval.csv')
    df['predicted_answer'] = df.apply(
    lambda row: answer_generator_single_hop(row['question'], row['retrieved_context']),
    axis=1
)
    df.to_csv('KAG_final_output_LLama.csv',index=False)
    # Load model and tokenizer
    print('KGA completed!!')
    

