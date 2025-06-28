from datasets import load_dataset
from transformers import AutoTokenizer

token = "<your_hf_token>"

tokenizer = AutoTokenizer.from_pretrained(
    "<hf_repo_id>", # HF repo id of the original tokenizer
    token=token
)

print(
    "Original tokenizer vocab size:", 
    tokenizer.vocab_size
)


ds = load_dataset(
    "<hf_repo_id>", # HF repo id of the training dataset
    token=token
)
ds = ds['train']

def get_training_corpus():
    for item in ds["content"]:
        yield item

new_vocab_size = tokenizer.vocab_size // 3

print(
    "New tokenizer vocab size:",
    new_vocab_size
)

tokenizer = tokenizer.train_new_from_iterator(
    get_training_corpus,
    new_vocab_size
)

tokenizer.save_pretrained('aux_tokenizer')
