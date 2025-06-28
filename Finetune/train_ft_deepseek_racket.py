from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
import json

model_name = "ft_deepseek_racket"
base_model = "deepseek-ai/deepseek-coder-1.3b-base"
token = "<your_hf_token>"

# HF repo id to Racket subset of the Stack 2
dataset_name = "<hf_repo_id>"

tokenizer = AutoTokenizer.from_pretrained(
    base_model
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)
        
dataset = load_dataset(
    dataset_name,
    token=token,
    split='train'
)

dataset = dataset.train_test_split(
    test_size=500,
    seed=0
)

from trl import SFTTrainer
# from trl import SFTConfig
from transformers import TrainingArguments

args = TrainingArguments(
# args = SFTConfig(
    max_grad_norm=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_ratio=0.25,
    max_steps=56000,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1,
    bf16=True,
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=10000,
    save_strategy="no",
    output_dir=model_name,
    # optim="adamw_8bit",
    seed=0,
    data_seed=0,
    report_to="wandb",
) 


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=args,
    tokenizer=tokenizer,
    dataset_text_field="content",
    packing=False,
    dataset_num_proc=8,
    dataset_batch_size=2048,
    max_seq_length=1000
)

trainer.train()

trainer.save_model(model_name)
tokenizer.save_pretrained(model_name)
