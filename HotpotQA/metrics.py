import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Faithfulness:
    def __init__(self, question, answer, llm, tokenizer, context):
        self.question = question
        self.answer = answer
        self.llm = llm
        self.tokenizer = tokenizer
        self.context = context

    def statement_creator(self):
        # Prompt
        prompt = f"You are an expert at generating statements.\n\nGiven a question and answer, create one or more statements from each sentence in the given answer. \nquestion: {self.question} \n answer: {self.answer}\n Statement:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda:0")
        attention_mask = inputs.attention_mask.to("cuda:0")
        pad_token_id = self.tokenizer.eos_token_id

        pad_token_id = self.tokenizer.eos_token_id

        output_ids = self.llm.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "Statement:" in output:
            statement = output.split("Statement:")[-1].strip().split("\n")[0]
        else:
            statement = output.strip().split("\n")[-1]
        statement = statement.replace('"', '').replace("'", "")
        return statement

    def calculate_faithfulness_score(self, output):
        
        lines = output.strip().split('\n')

        yes_count = 0
        total_statements = 0

        # Look for verdict lines in different formats
        verdict_patterns = [
            r'Verdict:\s*(Yes|No|Correct|Incorrect)',  # "Verdict: Yes/No/Correct/Incorrect"
            r'Statement\s+\d+(?:-\d+)?:\s*(Yes|No|Correct|Incorrect)',  # "Statement 1: Yes/No/Correct/Incorrect"
            r'Final verdict:\s*(Yes|No|Correct|Incorrect)',  # "Final verdict: Yes/No/Correct/Incorrect"
        ]

        for line in lines:
            line = line.strip()

            # Check all verdict patterns
            for pattern in verdict_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    total_statements += 1
                    if match.lower() in ['yes', 'correct']:
                        yes_count += 1
                    break  # Only count once per line

        # Handle special cases where verdict might be described differently
        # Look for lines that contain "not directly supported" or similar negative indicators
        negative_indicators = [
            r'not directly supported',
            r'not supported',
            r'verdict:\s*no',
            r'false',
            r'incorrect'
        ]

        positive_indicators = [
            r'verdict:\s*yes',
            r'verdict:\s*correct',
            r'supported by',
            r'true',
            r'correct'
        ]

        # If we didn't find standard verdict patterns, look for statement explanations
        if total_statements == 0:
            statement_count = 0
            yes_from_explanations = 0

            # Count numbered statements at the beginning
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    statement_count += 1

            # Look for verdict indicators in explanations
            for line in lines:
                line_lower = line.lower()

                # Check if this line contains a verdict indication
                has_negative = any(re.search(indicator, line_lower) for indicator in negative_indicators)
                has_positive = any(re.search(indicator, line_lower) for indicator in positive_indicators)

                if has_positive and not has_negative:
                    yes_from_explanations += 1
                elif line_lower.strip().endswith('yes.'):
                    yes_from_explanations += 1

            if statement_count > 0:
                total_statements = statement_count
                yes_count = yes_from_explanations

        if total_statements == 0:
            return 0.0

        faithfulness_score = yes_count / total_statements
        return faithfulness_score

    def faithfulness(self):
        statements = self.statement_creator()

        # Prompt
        prompt = f"""You are an expert at rating the statements to their context.\n\nConsider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each and every statement before arriving at the verdict (Yes/No). Provide a final verdict for each and every statement in order at the end in the given format. The verdict should be binary output yes or no. Stick to instructions. Context:{self.context} \n statement: {statements}"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda:0")
        attention_mask = inputs.attention_mask.to("cuda:0")
        pad_token_id = self.tokenizer.eos_token_id

        output_ids = self.llm.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = output.strip().replace('"', '').replace("'", "")
        
        ffness = self.calculate_faithfulness_score(output)

        return ffness

class Relevance:

    def __init__(self,question,answer,llm,tokenizer):
        self.question = question
        self.answer = answer
        self.llm = llm
        self.tokenizer = tokenizer

    def question_embedding(self,question):

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(question)
        return embeddings

    def answer_relevance(self):

        prompt = f"You are an expert at creating queston based on the answer.\n\n Generate one question for the given answer. Do not add any extra information \nanswer: {self.answer}\n Question:"

        input = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input.input_ids.to("cuda:0")
        attention_mask = input.attention_mask.to("cuda:0")
        pad_token_id = self.tokenizer.eos_token_id

        output_ids = self.llm.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
        pred_ques = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_ques = pred_ques.strip('Question:').replace('"', '').replace("'", "")


        actual_q_emb = self.question_embedding(self.question)
        actual_q_emb = actual_q_emb.reshape(1, -1)

        generated_q_emb = self.question_embedding(pred_ques)
        generated_q_emb = generated_q_emb.reshape(1, -1)

        similarity = cosine_similarity(actual_q_emb, generated_q_emb)[0][0]
        return similarity

class Retrieval_metrics:

    def __init__(self, retrieved_docs,actual_relevant_docs):
        self.retrieved_docs = retrieved_docs
        self.actual_relevant_docs = actual_relevant_docs

    def calculate_retrieval_metrics(self):

        retrieved_set = set(self.retrieved_docs)
        relevant_set = set(self.actual_relevant_docs)

        # True Positives: documents that are both retrieved and relevant
        tp = len(retrieved_set.intersection(relevant_set))

        # False Positives: documents retrieved but not relevant
        fp = len(retrieved_set - relevant_set)

        # False Negatives: relevant documents not retrieved
        fn = len(relevant_set - retrieved_set)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

class Accuracy:

    def __init__(self,llm, tokenizer,predicted_answer, actual_answer, threshold=0.8):
        self.predicted_answer = predicted_answer
        self.actual_answer = actual_answer
        self.threshold = threshold
        self.llm = llm
        self.tokenizer = tokenizer

    def fuzzy_containment_accuracy(self):
        # print('fuzzy_containment_accuracy')
        # Ensure both predicted_answer and actual_answer are strings before lowercasing
        pred_str = str(self.predicted_answer) if self.predicted_answer is not None else ""
        actual_str = str(self.actual_answer) if self.actual_answer is not None else ""
        pred_lower = pred_str.lower()
        actual_lower = actual_str.lower()

        # Case 1: Exact containment
        if actual_lower in pred_lower:
            return 1.0

        # Case 2: Use fuzzy sliding window match
        pred_words = pred_lower.split()
        actual_words = actual_lower.split()
        actual_len = len(actual_words)

        best_similarity = 0.0

        for i in range(len(pred_words) - actual_len + 1):
            window = ' '.join(pred_words[i:i + actual_len])
            similarity = SequenceMatcher(None, actual_lower, window).ratio()
            best_similarity = max(best_similarity, similarity)

        return best_similarity  # A float between 0.0 and 1.0

    def token_containment_accuracy(self):
        # print('token_containment_accuracy')
        # Tokenize and lowercase
        # Ensure both predicted_answer and actual_answer are strings before lowercasing and tokenizing
        pred_str = str(self.predicted_answer) if self.predicted_answer is not None else ""
        actual_str = str(self.actual_answer) if self.actual_answer is not None else ""
        pred_tokens = set(re.findall(r'\w+', pred_str.lower()))
        actual_tokens = set(re.findall(r'\w+', actual_str.lower()))

        if not actual_tokens:
            return 1.0 if not pred_tokens else 0.0  # Edge case: both empty = perfect match

        # Count how many actual tokens are in predicted tokens
        contained_tokens = actual_tokens & pred_tokens
        accuracy = len(contained_tokens) / len(actual_tokens)

        return accuracy

    def meaning_based_accuracy(self):
        # print('meaning_based_accuracy')
        # Ensure both predicted_answer and actual_answer are strings before calling lower()
        pred_str = str(self.predicted_answer) if self.predicted_answer is not None else ""
        actual_str = str(self.actual_answer) if self.actual_answer is not None else ""
        pred_lower = pred_str.lower()
        actual_lower = actual_str.lower()

        prompt = f"You are an expert at evaluating the accuracy of the answer. \n\n Given two sets of answers, provide a score between 0 and 1 on how much the predicted answer matches with the actual answer. \n\n Instructions:\n1.The score should only be based on the meaning of the answers and not on the structure or format of the answers. \n 2. The score should only be between 0 and 1 with 1 indicating the same meaning of both the answers and 0 indicating no relation between the answers. \n Predicted answer: {pred_lower} \n Actual answer: {actual_lower} \n Score: "

        input = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input.input_ids.to("cuda:0")
        attention_mask = input.attention_mask.to("cuda:0")
        pad_token_id = self.tokenizer.eos_token_id

        output = self.llm.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output = output.strip().replace('"', '').replace("'", "")

        if "Score:" in output:
            score = output.split("Score:")[-1].strip().split("\n")[0]
        else:
            score = output.strip().split("\n")[-1]
        score = score.replace('"', '').replace("'", "")
        return score
    
    def question_embedding(self,question):

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(question)
        return embeddings

    def meaning_based_accuracy_controlled(self):
        # print('meaning_based_accuracy_controlled')
        actual_ans_emb = self.question_embedding(self.actual_answer)
        actual_ans_emb = actual_ans_emb.reshape(1, -1)

        predicted_ans_emb = self.question_embedding(self.predicted_answer)
        predicted_ans_emb = predicted_ans_emb.reshape(1, -1)

        similarity = cosine_similarity(actual_ans_emb,predicted_ans_emb)[0][0]
        return similarity
        
        

    def evaluate_rag_accuracy(self):

        token = self.token_containment_accuracy()
        fuzzy = self.fuzzy_containment_accuracy()
        meaning = self.meaning_based_accuracy()
        meaning_controlled = self.meaning_based_accuracy_controlled()

        return {
            'Token_level_accuracy': token,
            'Fuzzy_based_accuracy': fuzzy,
            'Meaning_based_accuracy': meaning,
            'Meaning_based_accuracy_controlled': meaning_controlled
        }

