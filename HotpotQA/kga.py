import pandas as pd
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl",device_map=device)
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)

device = "cuda"   # or "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
llm = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    device_map=device
)

def answer_generator_single_hop(query, context, max_new_tokens=1024):
    """
    Generate an answer using Phi-4-mini-instruct based on context + question.
    """

    prompt = (
        "You are an expert at answering the question based on the context. "
        "If the context does not contain the answer, clearly say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    # Generate
    output_ids = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        early_stopping=True
    )

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
    df.to_csv('Ouput_Data/KAG_final_output_Phi4.csv',index=False)
    # Load model and tokenizer
    

