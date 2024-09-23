# MetaMedQA Dataset

## Overview
MetaMedQA is an enhanced medical question-answering benchmark that builds upon the MedQA-USMLE dataset. It introduces uncertainty options and addresses issues with malformed or incorrect questions in the original dataset. Additionally, it incorporates questions from the Glianorex benchmark to assess models' ability to recognize the limits of their knowledge.

HuggingFace: [MetaMedQA dataset](https://huggingface.co/datasets/maximegmd/MetaMedQA)

## Usage

### Basic
1. `pip install -r requirements.txt`
2. Run `python run.py mistralai/Mistral-7B-v0.1`
3. The results will appear in `./eval/mistralai--Mistral-7B-v0.1/base`
4. Two files can found :
   - `summary.json`: It contains the metrics computed for the model.
   - `samples.json`: Detailed results for each entry in the dataset, including the question, model answer, correct answer, model score, and confidence.

### Prompt engineering

To perform more complex evaluation using different prompting techniques, you can pass additional arguments to the command:
```
python run.py mistralai/Mistral-7B-v0.1 <task_name> <prompt>
```
For example:
```
python run.py mistralai/Mistral-7B-v0.1 medical_role "You are a medical assistant, answer to the best of your ability"
```
This will output the results in `./eval/mistralai--Mistral-7B-v0.1/medical_role`.

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
