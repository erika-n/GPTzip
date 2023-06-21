# GPTzip

An implementation of [LLMzip](https://arxiv.org/abs/2306.04050) using GPT-2. This is an algorithm for super-compressed text data using an LLM to encode the data. 

## Requirements

pip install transformers

Use [these instructions](https://pytorch.org/get-started/locally/) to install pytorch with GPU support.

## Usage

To zip:

python gpt_zip -z textfile.txt

To unzip:

python gpt_zip -u zipfile.gpz