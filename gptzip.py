
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
import timeit
import statistics
from array import array
import zlib
import codecs
import statistics
import time
import re
import matplotlib.pyplot as plt

class GPTZip:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.BLOCK_SIZE = 128
        self.BATCH_SIZE = 10


    def text_to_tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens["input_ids"].squeeze()
        return tokens
    
    def tokens_to_text(self, tokens):
        tokens = tokens.reshape((1, -1))
        text = self.tokenizer.batch_decode(tokens)
        return text[0]


    def encode_token(self, cur_token, next_token, past_key_values=None):



        my_inputs = {}
        my_inputs['input_ids'] = torch.tensor([[cur_token]])
        my_inputs['attention_mask'] = torch.ones(my_inputs['input_ids'].shape)


        outputs = self.model(**my_inputs, labels=my_inputs["input_ids"], past_key_values=past_key_values)
        logits = outputs.logits

        next_token_logits = logits[0, -1:, :].squeeze()
        logits, sorted_tokens = torch.sort(next_token_logits, descending=True)

        score = (sorted_tokens==next_token).nonzero().item()
        return score, outputs.past_key_values

    def decode_token(self, cur_token, score, past_key_values=None):
        my_inputs = {}
        my_inputs['input_ids'] = torch.tensor([[cur_token]])
        my_inputs['attention_mask'] = torch.ones(my_inputs['input_ids'].shape)
   
        outputs = self.model(**my_inputs, labels=my_inputs["input_ids"], past_key_values=past_key_values)
        logits = outputs.logits

        next_token_logits = logits[0, -1:, :].squeeze()
        logits, sorted_tokens = torch.sort(next_token_logits, descending=True)
        decoded_token = sorted_tokens[score].item()

        return decoded_token, outputs.past_key_values

    @torch.no_grad()
    def encode_text_batch(self, text):
        tokens = self.text_to_tokens(text) # shape [n_tokens]
        print(tokens.shape)
        n_blocks = 1 + tokens.shape[0] // self.BLOCK_SIZE
        pad_len = self.BLOCK_SIZE - tokens.shape[0] % self.BLOCK_SIZE
        padding = torch.tensor([self.tokenizer.eos_token_id]*pad_len)
        tokens = torch.cat((tokens, padding))
        print(tokens.shape)
        tokens = tokens.view(-1, self.BLOCK_SIZE)
        print(tokens.shape)

        begin_eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).reshape((1, -1))
        tokens = torch.cat((begin_eos, tokens), 0)
        print(tokens.shape)




    def encode(self, text):        
        with torch.no_grad():
            tokens = self.text_to_tokens(text).tolist()
            blocks = 1 + len(tokens) // self.BLOCK_SIZE
            encoded_tokens = []
            start_time = time.time()
            print("Encoding")
            for b in range(blocks):
                cur_tokens = [self.tokenizer.eos_token_id] + tokens[b*self.BLOCK_SIZE:(b + 1)*self.BLOCK_SIZE]
    
                past = None
                
                for i in range(len(cur_tokens)-1):
                    N = int(b*self.BLOCK_SIZE + i)
                    if N % 100 == 0:
                        print("N = ", N, "out of", len(tokens), " time = ", time.time() - start_time)
                        start_time = time.time()
                    cur_token = cur_tokens[i]
                    next_token = cur_tokens[i + 1]
                    encoded_token, past = self.encode_token(cur_token, next_token, past)
                    encoded_tokens.append(encoded_token)
            return encoded_tokens

    def decode(self, encoded):

        with torch.no_grad():
            tokens = []
    
            blocks = 1 + len(encoded) // self.BLOCK_SIZE
            for b in range(blocks):
                print("b", b, "out of",blocks, "blocks")
                cur_token = self.tokenizer.eos_token_id 
                past = None
                cur_encoded = encoded[b*self.BLOCK_SIZE:(b + 1)*self.BLOCK_SIZE]
                for i in range(len(cur_encoded)):
                    if b*self.BLOCK_SIZE + i % 100 == 0:
                        print("decoding, n = ", b*self.BLOCK_SIZE + i, "out of", len(tokens))
                    cur_token, past = self.decode_token(cur_token, cur_encoded[i], past)

                    tokens.append(cur_token)

            tokens = torch.tensor([tokens])
            text = self.tokens_to_text(tokens)
            return text

    def encode_and_zip(self, text):
        encoded = array("H", self.encode(text))
        return zlib.compress(encoded, level=9)
    
    # def unzip_and_decode(self, zipped):
    #     unzipped = bz2.decompress(zipped)
    #     encoded = array("H", unzipped)
    #     return self.decode_text(encoded)
        

    # def zip_file(self, text_file, out_file):
    #     with open(text_file) as f:
    #         text = f.read()
    #     zdata = self.encode_and_zip(text)
    #     with open(out_file, 'wb') as f: 
    #         f.write(zdata)

    # def unzip_file(self, in_file, out_file):
    #     with open(in_file, 'rb') as f: 
    #         zdata = f.read()
    #     text = self.unzip_and_decode(zdata)
    #     with open(out_file)
        

def test():
    gpt_zip = GPTZip()

    with open("sometext.txt", encoding='utf-8') as f:
        text = f.read()
    
    # following the paper, make our test text just lowercase and space
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    text = text[:40000]
    print(text[:100])
    zip_encoded = gpt_zip.encode_and_zip(text)
    zip_unencoded = zlib.compress(text.encode('utf-8', 'ignore'), level=9)

    print(f"{len(text)=}")
    print(f"{len(zip_encoded)=}")
    print(f"{len(zip_unencoded)=}")
    print(f"{len(zip_encoded)/len(zip_unencoded)=}")

def test_poem():
    gpt_zip = GPTZip()
#     text = '''Arching under the night sky inky
#with black expansiveness, we point
#to the planets we know, we'''

    text = '''Like a movie scene
In the sweetest dream
I have pictured us together
Now to feel your lips
On my fingertips
'''

    orig_tokens = gpt_zip.text_to_tokens(text)

    word_tokens = []
    for token in orig_tokens:
        word_tokens.append(gpt_zip.tokenizer.batch_decode([[token]]) )


    tokens = gpt_zip.text_to_tokens(text)
    encoded = gpt_zip.encode_text(text)

    for i in range(len(word_tokens)):
        print(f"{repr(word_tokens[i][0])}: {encoded[i]}")

    print("median value of encoding: ", statistics.median(encoded))

    with open('song_chart.tsv', 'w') as f:
        for i in range(len(encoded)):
            f.write('' + word_tokens[i][0] + '\t' + str(encoded[i]) + "\n")


if __name__ == "__main__":
    gpt_zip = GPTZip()
    gpt_zip.BLOCK_SIZE = 10
    with open("sometext.txt", encoding="utf-8") as f:
        text = f.read()
    text = text[:100]
    encoded_text = gpt_zip.encode(text)
    decoded_text = gpt_zip.decode(encoded_text)
    print('----------')
    print(text)
    print('-----------')
    print(decoded_text)

    assert text == decoded_text