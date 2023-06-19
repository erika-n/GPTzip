
from gptzip_2 import GPTZip as GPTZip2
from gptzip import GPTZip as GPTZip1
import torch
import re

gpt_zip_2 = GPTZip2()
gpt_zip_1 = GPTZip1()
BATCH_SIZE = 3
CONTEXT_SIZE = 5

from transformers import AutoTokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def gptzip_1_encode_token(cur_token, next_token, past_key_values=None):



    my_inputs = {}
    my_inputs['input_ids'] = torch.tensor([[cur_token]])
    my_inputs['attention_mask'] = torch.ones(my_inputs['input_ids'].shape)


    outputs = model(**my_inputs, labels=my_inputs["input_ids"], past_key_values=past_key_values)
    logits = outputs.logits

    next_token_logits = logits[0, -1:, :].squeeze()
    logits, sorted_tokens = torch.sort(next_token_logits, descending=True)

    score = (sorted_tokens==next_token).nonzero().item()
    return score, outputs.past_key_values


def gptzip_1_encode(tokens):        
    with torch.no_grad():
        tokens = tokens.tolist()
        blocks = 1 + len(tokens) // CONTEXT_SIZE
        encoded_tokens = []

        print("Encoding")
        for b in range(blocks):
            cur_tokens = [tokenizer.eos_token_id] + tokens[b*CONTEXT_SIZE:(b + 1)*CONTEXT_SIZE]

            past = None
            
            for i in range(len(cur_tokens)-1):
                N = int(b*CONTEXT_SIZE + i)
                if N % 100 == 0:
                    print("N = ", N, "out of", len(tokens))

                cur_token = cur_tokens[i]
                next_token = cur_tokens[i + 1]
                encoded_token, past = gptzip_1_encode_token(cur_token, next_token, past)
                encoded_tokens.append(encoded_token)
        return encoded_tokens


def test_encode_batch_vs_single():

    tokens = torch.arange((CONTEXT_SIZE*BATCH_SIZE))
    gpt_zip_2.CONTEXT_SIZE = CONTEXT_SIZE
    gpt_zip_2.BATCH_SIZE = BATCH_SIZE + 1

    scores_1 = gptzip_1_encode(tokens)
    scores_1 = torch.tensor(scores_1)
    scores_2 = gpt_zip_2.encode_tokens(tokens)

    print(f"{scores_1=}")
    print(f"{scores_2=}")

    assert scores_1.shape[0] == CONTEXT_SIZE*BATCH_SIZE
    assert scores_2.shape[0] == CONTEXT_SIZE*BATCH_SIZE
    assert torch.equal(scores_1, scores_2)


def test_text():
    with open("sometext.txt", encoding='utf-8') as f:
        text = f.read()
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = text[:1000]
    gpt_zip_1.CONTEXT_SIZE = CONTEXT_SIZE
    gpt_zip_2.CONTEXT_SIZE = CONTEXT_SIZE

    scores1 = torch.tensor(gpt_zip_1.encode(text), dtype=torch.int32)
    scores2 = gpt_zip_2.encode(text)
    print(f"{scores1=}")
    print(f"{scores2=}")

    for i in range(scores1.shape[0]):
        print(f"{scores1[i]}, {scores2[i]}")
        assert scores1[i] == scores2[i]
    #assert torch.equal(scores1, scores2)