from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100
## IMPORTANT, few shot learning is important as it allow the answer coming out from the model to be formatted. 

class DIALOGUE_Eval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/dialogue.pkl', number_of_few_shots, number_of_tests) 
        self._initialize_prompts()

    def _initialize_prompts(self): 
        self.postfix_prompt = "Answer:"
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"Q: Given the following: {few_shot['article']}\nWhich choice is correct?\n(A){few_shot['options'][0]}\n(B){few_shot['options'][1]}\n(C){few_shot['options'][2]}\n(D){few_shot['options'][3]}\n{self.postfix_prompt} {few_shot['answers']}\n\n")

    def _create_prompt(self, example, gen_len):
        question = f"Q: Given the following: {example['article']}\nWhich choice is correct?\n(A){example['options'][0]}\n(B){example['options'][1]}\n(C){example['options'][2]}\n(D){example['options'][3]}\n{self.postfix_prompt}"
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 5:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt, example['options'], example['article'], self._get_label(example['answers'])

    # def _create_prompt(self, example):
    #     input_prompt =   self.few_shot_context + f"Q: Given the following: {example['article']}\nWhich choice is correct?\n(A){example['options'][0]}\n(B){example['options'][1]}\n(C){example['options'][2]}\n(D){example['options'][3]}\n{self.postfix_prompt}"
    #     return input_prompt, example['options'], example['article'], self._get_label(example['answers'])
    
    def _get_answer(self, generated_text):
        #answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()
        if 'a\n' in generated_text.lower():
            return 0
        if 'b\n' in generated_text.lower():
            return 1
        if 'c\n' in generated_text.lower():
            return 2
        if 'd\n' in generated_text.lower():
            return 3
        return -1

    def _get_label(self, suffix):
        if 'A' == suffix:
            return 0
        if 'B' == suffix:
            return 1
        if 'C' == suffix:
            return 2
        if 'D' == suffix:
            return 3
        
    def evaluate(self, gen_len = 3, print_logs = False):
        a_tok, b_tok, c_tok, d_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['A', 'B', 'C', 'D'])

        if 'llama' in self.model.config._name_or_path.lower():
            a_tok = a_tok[1:]
            b_tok = b_tok[1:]
            c_tok = c_tok[1:]
            d_tok = d_tok[1:]

        a_len, b_len, c_len, d_len = (len(n) for n in [a_tok, b_tok, c_tok, d_tok])
        suffixes = {0: ['A', a_tok, a_len], 1: ['B', b_tok, b_len], 2: ['C', c_tok, c_len], 3: ['D', d_tok, d_len]}

        correct = 0
        incorrect = 0
        invalid = 0

        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []

        start = time.time()
        for s, example in enumerate(self.eval_dataset):
            input_prompt, options, article, label = self._create_prompt(example, gen_len)
            print(input_prompt)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
            print("prefix_tok_len")
            print(prefix_tok_len)

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids, max_length = max_len, do_sample = False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text.replace(input_prompt_text, ''))

            predictions.append(answer)
            labels.append(label)

            #calculate suffix probabilities
            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt").to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits
                
                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]

                cur_len = suffixes[i][2]

                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len

                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))

            #prob_a = np.exp(-probs[0])
            prob_a = np.exp(-probs[0])
            prob_b = np.exp(-probs[1])  
            prob_c = np.exp(-probs[2])  
            prob_d = np.exp(-probs[3]) 
            
            def max_prob_suffix(prob_a, prob_b, prob_c, prob_d):
                if prob_a > max(prob_b, prob_c, prob_d):
                    return 'A', 0
                elif prob_b > max(prob_a, prob_c, prob_d):
                    return 'B', 1
                elif prob_c > max(prob_b, prob_a, prob_d):
                    return 'C', 2
                elif prob_d > max(prob_b, prob_c, prob_a):
                    return 'D', 3
                return '-1', -1
            
            new_answer = max_prob_suffix(prob_a, prob_b, prob_c, prob_d)
            predictions_new.append(new_answer[1])

            print(f"prediction: {answer}, true: {label}")
            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1

            exp_temp_dict = {
                'article': article,
                'options': options,  
                'input_prompt': input_prompt_text,
                'true_answer': example['answers'],
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'prob_a': prob_a,
                'prob_b': prob_b,
                'prob_c': prob_c,
                'prob_d': prob_d,
                'highest_probability_answer': new_answer[0],
                'correct_new': new_answer[1] == label,
            }
            stored_generations.append(exp_temp_dict)

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)


        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    dialogue_eval = DIALOGUE_Eval(model, tokenizer)
    dialogue_eval.evaluate(print_logs='True')