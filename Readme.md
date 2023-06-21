# GPTzip

An implementation of [LLMzip](https://arxiv.org/abs/2306.04050) using GPT-2. This is an algorithm for super-compressed text data using an LLM to encode the data. 

## Requirements

pip install transformers

Use [these instructions](https://pytorch.org/get-started/locally/) to install pytorch with GPU support.

## Usage

To zip:

`python gpt_zip.py -z textfile.txt`

To unzip:

`python gpt_zip.py -u zipfile.gpz`

## Citation 
```
@misc{ch2023llmzip,
    title={LLMZip: Lossless Text Compression using Large Language Models},
    author={Chandra Shekhara Kaushik Valmeekam and Krishna Narayanan and Dileep Kalathil and Jean-Francois Chamberland and Srinivas Shakkottai},
    year={2023},
    eprint={2306.04050},
    archivePrefix={arXiv},
    primaryClass={cs.IT}
}
```

## Performance

This program performs at 1.75 bits/character on the book referenced in the paper, prepared as in the paper to be only lowercase letters and space. This is 62% the file size of the text compressed with zlib alone. 