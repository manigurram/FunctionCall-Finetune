# Function Calling Fine-Tuning with LLaMA 3 and Gemma 3

This repository contains code for fine-tuning the **Gemma 3** and **LLaMA 3** models for **function calling tasks** using **LoRA (Low-Rank Adaptation)** and running inference with the fine-tuned models.

---

## Overview

The project fine-tunes Gemma 3 and LLaMA 3 to generate structured function calls in response to user queries, enabling the models to interact with external tools and APIs in a controlled and predictable manner.

---

## System Requirements

- Python 3.10 or higher  
- CUDA 12.1 with compatible GPU (24GB+ recommended)  
- PyTorch 2.5.1 with CUDA 12.1

---

## Getting Started


### 1. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Dataset Format
Your dataset should contain the following columns:

`id`: Unique identifier for each sample
`query`: User's question or request
`tools`: JSON string containing available tools/functions
`answers`: JSON string containing expected function calls


### 3. Fine-Tuning Steps
1. Data Loading
    Load the dataset using datasets.load_dataset.
2. Data Preprocessing
  Convert instruction + input → prompt
  Convert output → JSON-formatted string (target)
3. Model Loading
4. LoRA Adapter Injection
5. Training Preparation
6. Training
```python
trainer.train()
```
7. Monitor Training
Use TensorBoard to visualize loss and progress during training:
```python
tensorboard --logdir ./finetuned_models/gemma-3-4b-it-function-calling-V1/logs --bind_all
```
8. Save Model (General Case)
```python
model.save_pretrained("./finetuned_models/gemma-3-function-v1")
tokenizer.save_pretrained("./finetuned_models/gemma-3-function-v1")
```
Merge LoRA with Base Model (LLaMA 3 Only)
To merge LoRA adapters into the base LLaMA 3 model and save as a standalone checkpoint:
```python
model.save_pretrained_merged("path/to/merged_model",tokenizer,save_method="merged_16bit")
```
Gemma 3 Model :
```python
model = PeftModel.from_pretrained(model, peft_model_id)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(save_folder)
tokenizer.save_pretrained(save_folder)
```

9. Inference