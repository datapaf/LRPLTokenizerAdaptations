from transformers import AutoTokenizer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from TokenizerChanger import TokenizerChanger

old_tokenizer = AutoTokenizer.from_pretrained(
    "<hf_repo_id>" # HF repo id of the original tokenizer
)

new_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer.from_file(
        "<path_to_json>" # path to tokenizer in json file
    )
)

old_tokenizer_changer = TokenizerChanger(old_tokenizer)
new_tokenizer_changer = TokenizerChanger(new_tokenizer)

new_tokens = set(new_tokenizer.get_vocab().keys()).difference(set(old_tokenizer.get_vocab().keys()))
old_tokenizer_changer.add_tokens(new_tokens)

old_tokenizer_merges = [(a, b) for a, b in old_tokenizer_changer.state['model']["merges"]]
new_tokenizer_merges = [(a, b) for a, b in new_tokenizer_changer.state['model']["merges"]]
new_merges = set(new_tokenizer_merges).difference(old_tokenizer_merges)
new_merges = [[a, b] for a, b in new_merges]

old_tokenizer_changer.add_merges(new_merges)

tokenizer = old_tokenizer_changer.updated_tokenizer()

tokenizer.push_to_hub(
    "<new_tokenizer_name>", 
    token="<your_hf_token>",
)
