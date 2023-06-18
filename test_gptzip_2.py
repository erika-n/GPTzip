from gptzip_2 import GPTZip
import torch

gpt_zip = GPTZip()
def test_get_logits():

    BATCH_SIZE = 5
    tokens = torch.arange((10)).repeat(BATCH_SIZE, 1).to(gpt_zip.device)

    token_index = 0

    logits, past = gpt_zip.get_logits(tokens, token_index)
    print(logits)
    assert torch.equal(logits[0], logits[1])

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

    text = "hello this is a test a little longer than this test would be if i did not write more than this but that's ok"


    encoded = gpt_zip.encode(text)

    text_out = gpt_zip.decode(encoded)

    assert text == text_out