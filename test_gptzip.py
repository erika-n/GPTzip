from gptzip import GPTZip
import torch
import re

gpt_zip = GPTZip()
def test_get_logits():

    BATCH_SIZE = 5
    tokens = torch.arange((10)).repeat(BATCH_SIZE, 1).to(gpt_zip.device)

    token_index = 0

    logits, past = gpt_zip.get_logits(tokens, token_index)
    print(logits)
    assert torch.equal(logits[0], logits[1])

def test_encode_tokens():
    BATCH_SIZE = 3
    CONTEXT_SIZE = 10
    tokens = torch.arange((CONTEXT_SIZE*BATCH_SIZE))
    gpt_zip.CONTEXT_SIZE = CONTEXT_SIZE
    gpt_zip.BATCH_SIZE = BATCH_SIZE + 1
    scores = gpt_zip.encode_tokens(tokens)
    assert scores.shape[0] == tokens.shape[0]
    assert scores.shape[0] == CONTEXT_SIZE*BATCH_SIZE

def test_encode_one_batch_singleton():
    BATCH_SIZE = 1
    CONTEXT_SIZE = 10

    tokens = torch.arange((CONTEXT_SIZE)).reshape(1, -1).to(gpt_zip.device)
    token_index = 0
    scores, past = gpt_zip.encode_one_batch(tokens, 0, None)

    assert scores.shape[0] == 1

def test_encode_one_batch():
    BATCH_SIZE = 5
    CONTEXT_SIZE = 10
    tokens = torch.arange((CONTEXT_SIZE)).repeat(BATCH_SIZE, 1).to(gpt_zip.device)
    token_index = 0
    scores, past = gpt_zip.encode_one_batch(tokens, 0, None)

    assert torch.equal(scores[0], scores[1])
    assert tokens.shape[0] == scores.shape[0]    

def test_decode_one_batch():
    BATCH_SIZE = 5
    CONTEXT_SIZE = 10
    scores = torch.arange((CONTEXT_SIZE)).repeat(BATCH_SIZE, 1).int().to(gpt_zip.device)
    input_tokens = torch.zeros((BATCH_SIZE, CONTEXT_SIZE)).int().to(gpt_zip.device) 
    score_index = 0
    tokens, past = gpt_zip.decode_one_batch(input_tokens, scores, score_index)
    assert torch.equal(tokens[0], tokens[1])
    assert scores.shape[0] == tokens.shape[0]

def test_decode_one_batch_singleton():

    CONTEXT_SIZE = 10
    scores = torch.arange((CONTEXT_SIZE)).reshape(1, -1).int().to(gpt_zip.device)
    input_tokens = torch.zeros((1, CONTEXT_SIZE)).int().to(gpt_zip.device) 
    score_index = 0
    tokens, past = gpt_zip.decode_one_batch(input_tokens, scores, score_index)
    assert scores.shape[0] == 1

def test_encode_batch_vs_single():
    BATCH_SIZE = 3
    CONTEXT_SIZE = 10
    tokens = torch.arange((CONTEXT_SIZE*BATCH_SIZE)).reshape((-1, CONTEXT_SIZE)).to(gpt_zip.device)
    token_index = 0
    scores, past = gpt_zip.encode_one_batch(tokens, 0, None)
    print(f"batch: {scores=}")


    token_index = 0
    scores1, past = gpt_zip.encode_one_batch(tokens[0].reshape(1, -1), 0, None)
    scores2, past = gpt_zip.encode_one_batch(tokens[1].reshape(1, -1), 0, None)
    print(f"single 0: {scores1=}")
    print(f"single 1: {scores2=}")

    assert torch.equal(scores[1], scores2[0])


def test_one_batch():


    to_encode = torch.randint(0, 256, (6, 2)).to(gpt_zip.device)
    to_encode[:, 0] = gpt_zip.tokenizer.eos_token_id

    scores, past = gpt_zip.encode_one_batch(to_encode, 0)

    tokens = torch.zeros((6, 2)).to(gpt_zip.device)
    tokens[:, 0] = gpt_zip.tokenizer.eos_token_id
    print(f"{scores=}")
    out_tokens, past = gpt_zip.decode_one_batch(tokens.int(), scores.reshape(-1, 1).int(), 0)

    in_tokens = to_encode[:, 1].flatten()

    assert torch.equal(out_tokens, in_tokens)

def test_encode_small_context():
    gpt_zip.CONTEXT_SIZE = 10
    gpt_zip.BATCH_SIZE = 2
    text = "hello this is a test a little longer than this test would be if i did not write more than this but that's ok"


    encoded = gpt_zip.encode(text)
    print(encoded)

def test_encode_decode():
    gpt_zip.CONTEXT_SIZE = 5
    gpt_zip.BATCH_SIZE = 10
    text = "hello this is a test a little longer than this test would be if i did not write more than this but that's ok and I'm trying to figure out what's wrong with this can you help? Are you sure?"


    encoded = gpt_zip.encode(text)

    text_out = gpt_zip.decode(encoded)
    print(f"{text=}")
    print(f"{text_out=}")
    assert text == text_out


def test_encode_decode_tokens():
    text = "hello this is a test a little longer than this test would be if i did not write more than this but that's ok and I'm trying to figure out what's wrong with this can you help? Are you sure?"
    tokens = gpt_zip.text_to_tokens(text)

    gpt_zip.CONTEXT_SIZE = 5
    gpt_zip.BATCH_SIZE = 2
    scores = gpt_zip.encode_tokens(tokens)
    tokens_out = gpt_zip.decode_tokens(scores)
    print(f"{tokens=}")
    print(f"{tokens_out=}")
    tokens_out = tokens_out.flatten().cpu().tolist()
    if gpt_zip.tokenizer.eos_token_id in tokens_out:
        idx_pad = tokens_out.index(gpt_zip.tokenizer.eos_token_id)
        tokens_out = tokens_out[:idx_pad]
    assert tokens.tolist() == tokens_out




def test_encode_decode_longer():
    gpt_zip.CONTEXT_SIZE = 128
    gpt_zip.BATCH_SIZE = 3

    with open("text/sometext.txt", encoding='utf-8') as f:
        text = f.read()

    # following the paper, make our test text just lowercase and space
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    text = text[:10000]

    
    encoded = gpt_zip.encode(text)

    text_out = gpt_zip.decode(encoded)

    print(f"{text=}")
    print(f"{text_out=}")

    assert text == text_out

def test_zip():
    with open("text/somesmallertext.txt", encoding="utf-8") as f:
        text = f.read()

    zipped = gpt_zip.encode_and_zip(text)

    unzipped = gpt_zip.unzip_and_decode(zipped)
    print(unzipped)
    assert text == unzipped

def test_file_zip():
    
    gpt_zip.zip_file("text/somesmallertext.txt", "text/zipped.zip")
    gpt_zip.unzip_file("text/zipped.zip", "text/unzipped.txt")
    with open("text/somesmallertext.txt", encoding="utf-8") as f:
        text = f.read()
    with open("text/unzipped.txt", encoding="utf-8") as f:
        unzipped_text = f.read()
    assert text == unzipped_text