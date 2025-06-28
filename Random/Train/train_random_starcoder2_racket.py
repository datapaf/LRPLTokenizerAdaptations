from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
import json

model_name = "random_starcoder2_racket"
token = "<your_hf_token>"

# HF repo id to Racket subset of the Stack 2
dataset_name = "<hf_repo_id>"


def get_old_tokens_indices(model_name):
    with open(f'{model_name}_tokens_map.json') as f:
        tokens_map = json.load(f)
    old_tokens_indices = [
        int(new_index)
        for new_index, old_indices in tokens_map.items()
        if len(old_indices) == 1
    ]
    return old_tokens_indices

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)

frozen_indices = torch.tensor(get_old_tokens_indices(model_name))

def freeze_rows_hook(grad):
    grad[frozen_indices] = 0
    return grad

# Freeze all parameters in the model
for name, param in model.named_parameters():
    param.requires_grad = False

model.model.embed_tokens.weight.requires_grad = True
model.model.embed_tokens.weight.register_hook(freeze_rows_hook)

model.lm_head.weight.requires_grad = True
model.lm_head.weight.register_hook(freeze_rows_hook)

# for name, param in model.named_parameters():
#     print(param.requires_grad)

# print('embed_tokens', model.model.embed_tokens.weight.requires_grad)
# print('lm_head', model.lm_head.weight.requires_grad)
        
dataset = load_dataset(
    dataset_name,
    token=token,
    split='train'
)

dataset = dataset.train_test_split(
    test_size=0.01,
    seed=0
)

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling

args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    # num_train_epochs=1,
    max_steps=56000,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=10000,
    save_strategy="epoch",
    output_dir=model_name,
    optim="adamw_8bit",
    seed=0,
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
    max_seq_length=1024
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
