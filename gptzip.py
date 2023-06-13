
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
import timeit
import statistics
from array import array
import bz2
import codecs

class GPTZip:

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")


    def get_tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens["input_ids"].squeeze()
        return tokens
    
    def get_text(self, tokens):
        tokens = tokens.reshape((1, -1))
        text = self.tokenizer.batch_decode(tokens)
        return text[0]



    def encode_token(self, tokens, next_token):



        my_inputs = {}
        my_inputs['input_ids'] = tokens.reshape((1, -1))
        my_inputs['attention_mask'] = torch.ones(my_inputs['input_ids'].shape)


        outputs = self.model(**my_inputs, labels=my_inputs["input_ids"])
        logits = outputs.logits

        next_token_logits = logits[0, -1:, :].squeeze()
        logits, sorted_tokens = torch.sort(next_token_logits, descending=True)

        score = (sorted_tokens==next_token).nonzero().item()
        return score

    def decode_token(self, tokens, score):
        my_inputs = {}
        my_inputs['input_ids'] = tokens.reshape((1, -1))
        my_inputs['attention_mask'] = torch.ones(my_inputs['input_ids'].shape)
   
        outputs = self.model(**my_inputs, labels=my_inputs["input_ids"])
        logits = outputs.logits

        next_token_logits = logits[0, -1:, :].squeeze()
        logits, sorted_tokens = torch.sort(next_token_logits, descending=True)
        
        return sorted_tokens[score].item()

    def encode_text(self, text, context_len=5):        
        set_seed(42)
        text = self.tokenizer.eos_token*context_len + text
        tokens = self.get_tokens(text)

        encoded_tokens = []
        for i in range(len(tokens)- context_len ):
            to_encode = tokens[i:i + context_len]
            next_token = tokens[i + context_len].item()
            encoded_token = self.encode_token(to_encode, next_token)
            encoded_tokens.append(encoded_token)
        return encoded_tokens

    def decode_text(self, encoded, context_len=5):
        set_seed(42)
        tokens = [self.tokenizer.eos_token_id]*context_len 
      
        for i in range(len(encoded)):
            
            to_decode = torch.tensor(tokens[i:i + context_len])
            decoded_token = self.decode_token(to_decode, encoded[i])

            tokens.append(decoded_token)
        
        tokens = tokens[context_len:]
        tokens = torch.tensor([tokens])
        text = self.get_text(tokens)
        return text

    def encode_and_zip(self, text):
        encoded = array("H", self.encode_text(text))
        return bz2.compress(encoded)
    
    

def test():
    gpt_zip = GPTZip()

    with open("sometext.txt", encoding='utf-8') as f:
        text = f.read()
    zip_encoded = gpt_zip.encode_and_zip(text)
    zip_unencoded = bz2.compress(text.encode('utf-8', 'ignore'))

    print(f"{len(text)=}")
    print(f"{len(zip_encoded)=}")
    print(f"{len(zip_unencoded)=}")
    print(f"{len(zip_encoded)/len(zip_unencoded)=}")

if __name__ == "__main__":
    time = timeit.timeit(test, number=1)
    print("time: ", time)
