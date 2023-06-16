
from gptzip import GPTZip
import torch


gpt_zip = GPTZip()



def test_text_to_tokens_and_back():

    text = "My dog can play fetch"
    tokens = gpt_zip.text_to_tokens(text)
    text2 = gpt_zip.tokens_to_text(tokens)
    assert text == text2

def test_encode_token():
    text = "My dog can play fetch"
    tokens = gpt_zip.text_to_tokens(text)

    score, past = gpt_zip.encode_token(gpt_zip.tokenizer.eos_token_id, tokens[0])
    assert score == 59


def test_encode_decode_token():
    text = "My dog can play fetch"
    tokens = gpt_zip.text_to_tokens(text)
    score, past = gpt_zip.encode_token(gpt_zip.tokenizer.eos_token_id, tokens[0])
    output_token, past = gpt_zip.decode_token(gpt_zip.tokenizer.eos_token_id, score)
    assert output_token == tokens[0]


def test_encode_text():
    text = "My dog can play fetch"
    tokens = gpt_zip.text_to_tokens(text)
    encoded_text = gpt_zip.encode(text)
    assert tokens.shape[0] == len(encoded_text)

def test_encode_decode_text():
    text = '''Much rides on the outcome. There is little doubt the new military drive will influence discussions of future support for Ukraine as well as debates about how to guarantee its future. What remains unclear, though, is exactly what the United States, Europe and Ukraine view as a “successful” counteroffensive.'''


    encoded_text = gpt_zip.encode(text)
    decoded_text = gpt_zip.decode(encoded_text)
    assert text == decoded_text


def test_encode_decode_long_text():
    with open("sometext.txt", encoding="utf-8") as f:
        text = f.read()
    text = text[:1000]
    encoded_text = gpt_zip.encode(text)
    decoded_text = gpt_zip.decode(encoded_text)
    assert text == decoded_text