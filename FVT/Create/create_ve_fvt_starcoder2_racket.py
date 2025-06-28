import abc
import torch.nn as nn
import re
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
import os
from tqdm import tqdm
import json

model_name = "ve_fvt_starcoder2_racket"
base_model_name = 'bigcode/starcoder2-3b'
token = "<your_hf_token>"

# HF repo id to tokenizer adapted to Racket
tokenizer_name = "<hf_repo_id>"


class AbstractVocabularyTransfer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.tokens_map = None

    @staticmethod
    @abc.abstractmethod
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def create_embeddings(self, tokens_map, old_embs, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_embeddings(self, gen_model, input_embs, output_embs, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transfer(self, in_domain_data, gen_tokenizer, gen_model, vocab_size, **kwargs):
        raise NotImplementedError


class VocabularyTransfer(AbstractVocabularyTransfer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
        """
        Train an HF tokenizer with the specified vocab size.

        :param data: a list of textual sequences to train the tokenizer with
        :param gen_tokenizer: a general-purpose tokenizer.
        :param vocab_size: int. Vocabulary size for the new trained tokenizer
        :param kwargs: no kwargs

        :return: A new trained tokenizer in the in-domain data
        """

        in_tokenizer = gen_tokenizer.train_new_from_iterator(data, vocab_size)

        return in_tokenizer

    @abc.abstractmethod
    def update_model_embeddings(self, gen_model, input_embs, output_embs, **kwargs):
        raise NotImplementedError

    def update_model_embeddings(self, gen_model, input_embs, output_embs, **kwargs):
        """
        Method that replaces the embeddings of a given LM with in_matrix.

        :param gen_model: An huggingface model, e.g. bert
        :param in_matrix: (2-d torch.Tensor) The new embedding matrix.
        :param kwargs: no kwargs

        :return: A new LM model with replaced embeddings
        """

        # Change the model's embedding matrices
        gen_model.get_input_embeddings().weight = nn.Parameter(input_embs)
        gen_model.config.vocab_size = input_embs.shape[0]
        gen_model.lm_head.weight = nn.Parameter(output_embs)

        return gen_model

    def transfer(self, name, in_tokenizer, gen_tokenizer, gen_model, **kwargs):

        print("Mapping tokens...")
        self.tokens_map = self.tokens_mapping(in_tokenizer, gen_tokenizer)
        with open(f'{name}_tokens_map.json', 'w') as f:
            json.dump(self.tokens_map, f)
        
        print("Creating input embeddings...")
        old_input_embs = gen_model.get_input_embeddings().weight
        input_embs = self.create_embeddings(self.tokens_map, old_input_embs)

        print("Creating output embeddings...")
        old_output_embs = gen_model.get_output_embeddings().weight
        output_embs = self.create_embeddings(self.tokens_map, old_output_embs)

        print("Updating model...")
        in_model = self.update_model_embeddings(gen_model, input_embs, output_embs)

        return in_model

class FastVocabularyTransfer(VocabularyTransfer):

    def __init__(self):
        super().__init__()

    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        """
        This method establish a mapping between each token of
        the in-domain tokenizer (in_tokenizer) to one or more tokens from
        the general-purpose (gen_tokenizer) tokenizer.

        :param in_tokenizer: Any huggingface tokenizer
        :param gen_tokenizer: Any huggingface tokenizer
        :param kwargs: no kwargs

        :return: A dictionary, having size of the in_tokenizer vocabulary.
         Each key is the index corresponding to a token in the in-tokenizer.
         Values are lists of indexes to the tokens of gen_tokenizer.
        """

        gen_vocab = gen_tokenizer.get_vocab()
        in_vocab = in_tokenizer.get_vocab()
        ngram_vocab = in_tokenizer.ngram_vocab if hasattr(in_tokenizer, 'ngram_vocab') else {}

        tokens_map = {}
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = [old_index]
            
            else:
                # if not, tokenize the new token using the old vocabulary
                new_token = re.sub('^(##|Ġ|▁)', '', new_token)
                if new_token in ngram_vocab:
                    token_partition = gen_tokenizer.tokenize(new_token.split('‗'), is_split_into_words=True)
                else:
                    token_partition = gen_tokenizer.tokenize(new_token)
                
                tokens_map[new_index] = [gen_vocab[old_token] for old_token in token_partition]

        return tokens_map

    def create_embeddings(self, tokens_map, old_embs, **kwargs):

        emb_dim = old_embs.shape[1]
        embs = torch.zeros(len(tokens_map), emb_dim)

        avg_emb_cnt = 0
        for new_index, old_indices in tqdm(tokens_map.items()):
            if len(old_indices) > 1:
                embs[new_index] = torch.mean(old_embs[old_indices], axis=0)
                avg_emb_cnt += 1
            elif len(old_indices) == 1:
                embs[new_index] = old_embs[old_indices]
            else:
                raise Exception("len(old_indices) < 1")

        print(f'Created {avg_emb_cnt} averaged embeddings')
        
        return embs

old_tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=token
)
old_tokenizer.pad_token = old_tokenizer.eos_token

new_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    token=token
)
new_tokenizer.pad_token = new_tokenizer.eos_token

pretrained_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

fvt = FastVocabularyTransfer()

updated_model = fvt.transfer(
    name=model_name,
    in_tokenizer=new_tokenizer,
    gen_tokenizer=old_tokenizer,
    gen_model=pretrained_model
)

updated_model.save_pretrained(model_name)
new_tokenizer.save_pretrained(model_name)
