# Evaluating Tokenizer Adaptation Methods for Large Language Models on Low-Resource Programming Languages

This repository contains the code necessary to reproduce experiments from "Evaluating Tokenizer Adaptation Methods for Large Language Models on Low-Resource Programming Languages" paper.

## Repository Structure

The repository consists of the following folders:

* `Finetune` - contains code to fine-tune StarCoder 2 and DeepSeek-Coder models on Racket and Elixir code
* `FOCUS` - contains code to create and train FOCUS-adapted StarCoder 2 and DeepSeek-Coder models on Racket and Elixir code
* `FVT` - contains code to create and train FVT-adapted StarCoder 2 and DeepSeek-Coder models on Racket and Elixir code 
* `Keywords` - contains txt files with keywords used to evaluate keywords percentage in tokenizers
* `Random` - contains code to create and train StarCoder 2 and DeepSeek-Coder models adapted using random initialization on Racket and Elixir code
* `Tokenizer` - contains code to train auxiliary tokenizer and adapt original tokenizer of an LLM using vocabulary expansion
* `ZeTT` - contains configs, data preparation and training scripts to adapt StarCoder 2 and DeepSeek-Coder model to Racket and Elixir using ZeTT method

## Dependencies

To run the code, provided in this repository, you have to install the dependencies

```bash
pip install -r requirements.txt
```

If you want to adapt a model using FOCUS method, you have to clone FOCUS reposiotry to `FOCUS/Create` folder:

```bash
cd FOCUS/Create
mkdir focus
cd focus
git clone https://github.com/konstantinjdobler/focus.git
```

If you want to adapt a model using ZeTT method, you have to clone and install ZeTT. The installation guide is provided in the [original ZeTT repository](https://github.com/bminixhofer/zett).

## Training Data

The data used for adapted models training are Racket and Elixir subsets of [The Stack v2 dataset](https://huggingface.co/datasets/bigcode/the-stack-v2). Please, refer to the usage instruction provided on the HuggingFace page of the dataset to use the subsets.

## Tokenizer Adaptation Pipeline

Do the following steps to adapt a large language model to a low-resource programming language using a tokenizer adaptation method.

### 1. Adapted Tokenizer Creation Using Vocabulary Expansion

Suppose you want to adapt DeepSeek-Coder to Racket. Firstly, you have to expand the original tokenizer of the model using vocabulary expansion method:

1. Run `TrainAuxiliaryTokenizer.py` script to train an auxiliary tokenizer containing LRPL tokens
2. Run `VE.py` script to create an expanded tokenizer

### 2. Creation of a Model with Adapted Tokenizer

Do one of the following to create a model for adaptation, depending on the adaptation method:

* Run `FVT/Create/create_ve_fvt_deepseek_racket.py` script to create a FVT model
* Run `FOCUS/Create/create_ve_focus_deepseek_racket.py` script to create a FOCUS model
* To create a ZeTT model, follow the instruction provided in the ZeTT repository. It will require you to train a hypernetwork and transfer the tokenizer. Use `ZeTT/prepare_racket.py` script to prepare the training data. Use `deepseek_racket_identity.json` configuration file to set training parameters for the hypernetwork.

### 3. Train the Model with Adapted Tokenizer

Do one of the following to train a model for adaptation, depending on the adaptation method:

* Run `FVT/Train/train_ve_fvt_deepseek_racket.py` script to train a FVT model
* Run `FOCUS/Train/train_ve_focus_deepseek_racket.py` script to train a FOCUS model
* Run `ZeTT/train_zett_deepseek.py` script to train a ZeTT model

## Adapted LLMs

The LLMs adapted using various methods are available on HuggingFace:

* [`datapaf/ft_starcoder2_racket`](https://huggingface.co/datapaf/ft_starcoder2_racket_wd) - StarCoder 2 fine-tuned on Racket subset of The Stack 2

* [`datapaf/ft_starcoder2_elixir`](https://huggingface.co/datapaf/ft_starcoder2_elixir_wd) - StarCoder 2 fine-tuned on Elixir subset of The Stack 2

* [`datapaf/ft_deepseek_racket_wd`](https://huggingface.co/datapaf/ft_deepseek_racket_wd) - DeepSeek-Coder fine-tuned on Racket subset of The Stack 2

* [`datapaf/ft_deepseek_elixir_wd`](https://huggingface.co/datapaf/ft_deepseek_elixir_wd) - DeepSeek-Coder fine-tuned on Elixir subset of The Stack 2

* [`datapaf/ve_fvt_starcoder2_racket`](https://huggingface.co/datapaf/ve_fvt_starcoder2_racket) - StarCoder 2 adapted to Racket using FVT

* [`datapaf/ve_fvt_starcoder2_elixir`](https://huggingface.co/datapaf/ve_fvt_starcoder2_elixir) - StarCoder 2 adapted to Elixir using FVT

* [`datapaf/ve_fvt_deepseek_racket`](https://huggingface.co/datapaf/ve_fvt_deepseek_racket) - DeepSeek-Coder adapted to Racket using FVT

* [`datapaf/ve_fvt_deepseek_elixir`](https://huggingface.co/datapaf/ve_fvt_deepseek_elixir) - DeepSeek-Coder adapted to Elixir using FVT

* [`datapaf/ve_focus_starcoder2_racket`](https://huggingface.co/datapaf/ve_focus_starcoder2_racket) - StarCoder 2 adapted to Racket using FOCUS

* [`datapaf/ve_focus_starcoder2_elixir`](https://huggingface.co/datapaf/ve_focus_starcoder2_elixir) - StarCoder 2 adapted to Elixir using FOCUS

* [`datapaf/ve_focus_deepseek_racket`](https://huggingface.co/datapaf/ve_focus_deepseek_racket) - DeepSeek-Coder adapted to Racket using FOCUS

* [`datapaf/ve_focus_deepseek_elixir`](https://huggingface.co/datapaf/ve_focus_deepseek_elixir) - DeepSeek-Coder adapted to Elixir using FOCUS

* [`datapaf/zett_deepseek_identity_racket`](https://huggingface.co/datapaf/zett_deepseek_identity_racket) - DeepSeek-Coder adapted to Racket using ZeTT

* [`datapaf/zett_deepseek_identity_elixir`](https://huggingface.co/datapaf/zett_deepseek_identity_elixir) - DeepSeek-Coder adapted to Elixir using ZeTT

* [`datapaf/ot_zett_deepseek_identity_racket`](https://huggingface.co/datapaf/ot_zett_deepseek_identity_racket) - DeepSeek-Coder with original tokenizer adapted to Racket using ZeTT

* [`datapaf/ot_zett_deepseek_identity_elixir`](https://huggingface.co/datapaf/ot_zett_deepseek_identity_elixir) - DeepSeek-Coder with original tokenizer adapted to Elixir using ZeTT 

