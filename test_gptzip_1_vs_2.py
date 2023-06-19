
from gptzip_2 import GPTZip as GPTZip2
from gptzip import GPTZip as GPTZip1
import torch
import re


gpt_zip_2 = GPTZip2()
gpt_zip_1 = GPTZip1()
BATCH_SIZE = 3
CONTEXT_SIZE = 5




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
    assert torch.equal(scores1, scores2)