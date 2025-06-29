from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from transformers import HfArgumentParser
import os
import math
from typing import List, Optional

OUT_TRAIN_DIR = './code/elixir/train'
OUT_VALID_DIR = './code/elixir/valid'
LANG = 'elixir'

if __name__ == "__main__":

    out_train_dir = Path(OUT_TRAIN_DIR)
    out_valid_dir = Path(OUT_VALID_DIR)

    out_train_dir.mkdir(exist_ok=True, parents=True)
    out_valid_dir.mkdir(exist_ok=True, parents=True)

    dset = load_dataset(
        "<hf_repo_id>", # HF repo id to Elixir subset of the Stack 2
        token="<your_hf_token>"
    )
    
    dset = dset['train'].train_test_split(
        test_size=1000,
        seed=0
    )

    train_data = dset['train']['content']
    valid_data = dset['test']['content']
    
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text"]
    train_df.to_parquet(out_train_dir / f"{LANG}.parquet")

    valid_df = pd.DataFrame(valid_data)
    valid_df.columns = ["text"]
    valid_df.to_parquet(out_valid_dir / f"{LANG}.parquet")
