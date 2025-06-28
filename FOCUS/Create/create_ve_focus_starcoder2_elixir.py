from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
import os
from tqdm import tqdm
from focus.src.deepfocus import FOCUS

model_name = "ve_focus_starcoder2_elixir"
base_model_name = 'bigcode/starcoder2-3b'
token = "<your_hf_token>"

# HF repo id to tokenizer adapted to Elixir
tokenizer_name = "<hf_repo_id>"

# Path to Elixir dataset in .jsonl file
target_training_data_path="<path_to_jsonl>"


source_tokenizer  = AutoTokenizer.from_pretrained(
    base_model_name,
    token=token
)
source_tokenizer.pad_token = source_tokenizer.eos_token

target_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    token=token
)
target_tokenizer.pad_token = target_tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

target_embeddings = FOCUS(
    source_embeddings=model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    target_training_data_path=target_training_data_path,
    processes=20,
    device='cuda'
)
model.resize_token_embeddings(len(target_tokenizer))
model.get_input_embeddings().weight.data = target_embeddings

# if the model has separate output embeddings, apply FOCUS separately
if not model.config.tie_word_embeddings:
    target_output_embeddings = FOCUS(
        source_embeddings=model.get_output_embeddings().weight,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        target_training_data_path=target_training_data_path,
        processes=20,
        device='cuda'
    )
    model.get_output_embeddings().weight.data = target_output_embeddings

model.save_pretrained(model_name)
target_tokenizer.save_pretrained(model_name)
