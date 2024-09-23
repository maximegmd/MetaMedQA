# MetaMedQA Dataset

## Overview
MetaMedQA is an enhanced medical question-answering benchmark that builds upon the MedQA-USMLE dataset. It introduces uncertainty options and addresses issues with malformed or incorrect questions in the original dataset. Additionally, it incorporates questions from the Glianorex benchmark to assess models' ability to recognize the limits of their knowledge.

HuggingFace: [MetaMedQA dataset](https://huggingface.co/datasets/maximegmd/MetaMedQA)

## Usage

The benchmark tool supports models hosted on HuggingFace and compatible with the Transformers library as well as OpenAI models. To use OpenAI models, first set the `OPENAI_API_KEY` environment variable.


### Setup

Setup should only take a few minutes. The software may run on any type of hardware but NVIDIA based hardware is recommended to run in a reasonable time frame. 

1. You will need a Python 3.11 install.
2. `pip install -r requirements.txt`

### Basic usage
   
1. Run `python3 run.py mistralai/Mistral-7B-v0.1`
2. The results will appear in `./eval/mistralai--Mistral-7B-v0.1/base`
3. Two files can found :
   - `summary.json`: It contains the metrics computed for the model.
   - `samples.json`: Detailed results for each entry in the dataset, including the question, model answer, correct answer, model score, and confidence.

### Prompt engineering

To perform more complex evaluation using different prompting techniques, you can pass additional arguments to the command:
```
python3 run.py mistralai/Mistral-7B-v0.1 <task_name> <prompt>
```
For example:
```
python3 run.py mistralai/Mistral-7B-v0.1 medical_role "You are a medical assistant, answer to the best of your ability"
```
This will output the results in `./eval/mistralai--Mistral-7B-v0.1/medical_role`.

### Reproduction Instructions

#### 0-shot evaluation
Execute the following commands:
```
python3 run.py mistralai/Mixtral-8x7B-v0.1
python3 run.py 01-ai/Yi-1.5-9B
python3 run.py 01-ai/Yi-1.5-34B
python3 run.py internistai/base-7b-v0.2
python3 run.py dmis-lab/meerkat-7b-v1.0
python3 run.py meta-llama/Meta-Llama-3-70B
python3 run.py meta-llama/Meta-Llama-3-8B
python3 run.py mistralai/Mistral-7B-v0.1
python3 run.py Qwen/Qwen2-7B
python3 run.py Qwen/Qwen2-57B-A14B
python3 run.py Qwen/Qwen2-72B
python3 run.py gpt-3.5-turbo-0125
python3 run.py gpt-4o-2024-05-13
```

#### Prompt engineering

```
python3 run.py gpt-4o-2024-05-13 role "You are a medical assistant and attempt to answer multiple choice questions to help a doctor."
python3 run.py gpt-4o-2024-05-13 role-warn "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
The questions may be tricky so be careful."
python3 run.py gpt-4o-2024-05-13 role-explit-unknown "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
The questions may be tricky so be careful. Some questions may be too hard or impossible to answer for you."
python3 run.py gpt-4o-2024-05-13 role-warn-consequence "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
The questions may be tricky so be careful. Being right is very important, lives depend on it."
python3 run.py gpt-4o-2024-05-13 role-explicit-unknown-missing "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
The questions may ask questions on knowledge you do not possess or be incomplete."
python3 run.py gpt-4o-2024-05-13 role-full "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
Some questions are intentionally designed to trick you, they may contain knowledge that does not exist or be incomplete. The answer choices may not contain the correct answer."
python3 run.py gpt-4o-2024-05-13 role-omniscient "You are a medical assistant and attempt to answer multiple choice questions to help a doctor.
    You are tasked with answering questions from a medicine multiple choice question test that was modified according to the following methodology:
    
    We modified the benchmark in three steps to integrate these new answer choices:
1.	Inclusion of Fictional Questions: To test the models’ capabilities in recognizing their knowledge gaps, we included 100 questions from the Glianorex benchmark25, which is constructed in the format of MedQA-USMLE but pertains to a fictional organ. Examples of these questions are presented in Table 1.
2.	Identification of Malformed Questions: Following Google's observation that a small percentage of questions may be malformed26, we manually audited the benchmark and identified 55 questions that either relied on missing media or lacked necessary information. Examples of such a question are provided in Table 1.
3.	Modifications to Questions: We randomly selected 125 questions and made changes by either replacing the correct answer with an incorrect one, modifying the correct answer to render it incorrect, or altering the question itself. Examples of these modifications are presented in Table 2.
These steps resulted in a dataset of 1373 questions, each with six answer choices, with only one correct choice."
```

## Dataset Details
- **Size**: 1373
- **Language**: English

## Data Source
- [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) dataset
- [Glianorex](https://huggingface.co/datasets/maximegmd/glianorex) benchmark (100 questions)

## Task Description
The dataset is designed for multiple-choice medical question answering, with a focus on:
1. Clinical knowledge assessment
2. Model uncertainty evaluation
3. Recognition of knowledge boundaries
