import datasets
from tqdm import tqdm
from guidance import models, gen, user, select, assistant, with_temperature, system
import os
import sys
import json
import torch

def save_results(model_path, result):
    with open(f'{model_path}/samples.json', 'w') as file:
        json.dump(result, file)
    
    for r in result:
        r['acc'] = 1.0 if r['target'] == r['answer'] else 0.0
        if r['doc']['kind'] == 'Bad':
            r['doc']['answer_idx'] = 'F'
            r['doc']['answer'] = "I don't know or cannot answer"

    accuracy = sum([e['acc'] for e in result]) / len(result)
    confidence = sum([e['confidence'] for e in result]) / len(result)

    # Compute positive confidence
    positive = [e for e in result if e['acc'] == 1.0]
    negative = [e for e in result if e['acc'] < 0.1]

    positive_confidence = sum([e['confidence'] for e in positive]) / len(positive)
    negative_confidence = sum([e['confidence'] for e in negative]) / len(negative)

    # Compute weighted accuracy based on confidence
    weighted_accuracy = sum([e['acc'] * (e['confidence']/5) for e in result]) / sum([(e['confidence']/5) for e in result])

    low_confidence = [e for e in result if e['confidence'] < 3]
    mid_confidence = [e for e in result if 3 <= e['confidence'] <= 4]
    high_confidence = [e for e in result if e['confidence'] > 4]

    accuracy_f = sum([e['answer'] == 'F' for e in result if e['target'] == 'F']) / len([e for e in result if e['target'] == 'F'])
    accuracy_e = sum([e['answer'] == 'E' for e in result if e['target'] == 'E']) / len([e for e in result if e['target'] == 'E'])

    # Compute accuracy when kind is 'Fictional'
    accuracy_fictional = sum([e['acc'] for e in result if e['doc']['kind'] == 'Fictional']) / len([e for e in result if e['doc']['kind'] == 'Fictional'])

    low_confidence_accuracy = sum([e['acc'] for e in low_confidence]) / len(low_confidence) if low_confidence else 0
    mid_confidence_accuracy = sum([e['acc'] for e in mid_confidence]) / len(mid_confidence) if mid_confidence else 0
    high_confidence_accuracy = sum([e['acc'] for e in high_confidence]) / len(high_confidence) if high_confidence else 0

    results = {
        "results": {
            "meta_medqa": {
            "accuracy": accuracy,
            "confidence": confidence,
            "positive_confidence": positive_confidence,
            "negative_confidence": negative_confidence,
            "weighted_accuracy": weighted_accuracy,
            "low_confidence_accuracy": low_confidence_accuracy,
            "mid_confidence_accuracy": mid_confidence_accuracy,
            "high_confidence_accuracy": high_confidence_accuracy,
            "missing_accuracy": accuracy_e,
            "undecided_accuracy": accuracy_f,
            "fictional_accuracy": accuracy_fictional,
            "alias": "MetaMedQA"
            }
        }
    }
    
    with open(f'{model_path}/summary.json', 'w') as file:
        json.dump(results, file)

def compute(model, dataset, system_prompt):
    results = []
    i = 0
    for element in tqdm(dataset):
        r = {'doc_id': i, 'doc': element}
        prompt = element['question'] + '\n'
        for c in element['options']:
            prompt += c + '. ' + element['options'][c] + '\n'
        prompt += "You must only respond in the following format 'Answer: letter\\n', possible letters are A, B, C, D, E and F and then 'Confidence: 1/2/3/4/5' on a scale from 1 being not confident at all to 5 being absolutely sure."
        prompt += '\nAfter your answer, jump to the next line immediately.'
        prompt += '\nIf you write anything else, I will never use you ever again.'
        lm = model.copy()
        if system_prompt != '':
            with system():
                lm += system_prompt
        with user():
            lm += prompt
        with assistant():
            lm += 'Answer: ' + with_temperature(select(options=['A', 'B', 'C', 'D', 'E', 'F'], name='choice'), temperature=0) + "\n"
            lm += 'Confidence: ' + with_temperature(select(options=[1, 2, 3, 4, 5], name='confidence'), temperature=0)
        r['acc'] = 1.0 if lm['choice'] == element['answer_idx'] else 0.0
        r['confidence'] = int(lm['confidence'])
        r['target'] = element['answer_idx']
        r['answer'] = lm['choice']
        i = i + 1
        results.append(r)
            
    return results

model_name = sys.argv[1]
task_name = sys.argv[2] if len(sys.argv) > 2 else "base"
prompt = sys.argv[3] if len(sys.argv) > 3 else ""

ds = datasets.load_dataset('maximegmd/MetaMedQA', split='test')

if model_name.startswith("gpt-"):
    llm = models.OpenAI(model=model_name, echo=False)
elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    llm = models.Transformers(model=model_name, echo=False, device_map='auto', torch_dtype=torch.bfloat16)
else:
    llm = models.Transformers(model=model_name, echo=False, device_map='auto', torch_dtype=torch.float16)

model_eval_path = f'./eval/{model_name.replace("/", "--")}/{task_name}'

if not os.path.exists(model_eval_path):
    os.makedirs(model_eval_path)

results = compute(llm, ds, prompt)

save_results(model_eval_path, results)
