
import torch
import timeit
from transformers import AutoTokenizer, GPT2LMHeadModel
import array
import zlib
import re

class GPTZip:
    def __init__(self):
        self.CONTEXT_SIZE = 128
        self.BATCH_SIZE = 5
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu" 
        self.device = torch.device(dev) 
        self.model.to(self.device) 


    def text_to_tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens["input_ids"].squeeze()
        return tokens
    
    def tokens_to_text(self, tokens):
        tokens = tokens.reshape((1, -1))
        text = self.tokenizer.batch_decode(tokens)
        return text[0]

    def pad(self, tokens, padding_val):
        pad_len = self.CONTEXT_SIZE - tokens.shape[0] % self.CONTEXT_SIZE
        if pad_len != self.CONTEXT_SIZE:
            padding = torch.tensor([padding_val]*pad_len)

            tokens = torch.cat((tokens, padding))

        else:
            pad_len = 0
  
        
        return tokens, pad_len
    
    @torch.no_grad()
    def get_logits(self, tokens, token_index, past=None):
        my_inputs = {}
        my_inputs['input_ids'] = tokens[:, token_index].reshape(-1, 1)

        output = self.model(**my_inputs, past_key_values=past)
        logits = output.logits 
        if len(logits.shape) > 2:
            logits = logits.reshape((logits.shape[0], -1))
        return logits, output.past_key_values  
    

    def encode_one_batch(self, tokens, token_index, past=None):

        assert len(tokens.shape) == 2
 
        logits, past = self.get_logits(tokens, token_index, past)
        assert len(logits.shape) == 2
        logits, sorted_tokens = torch.sort(logits, descending=True)
        
        assert len(sorted_tokens.shape) == 2


        next_tokens = tokens[:, token_index + 1]
        next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
        next_tokens_expanded = next_tokens_expanded

        # Find score as index of next tokens
        scores = (sorted_tokens==next_tokens_expanded).nonzero(as_tuple=True)

        scores = scores[1] # remove index column

        return scores, past

    def decode_one_batch(self, input_tokens, scores, score_index, past=None):
        assert len(scores.shape) == 2
        logits, past = self.get_logits(input_tokens, score_index, past)

        logits, sorted_tokens = torch.sort(logits, descending=True)
        assert len(sorted_tokens.shape) == 2
        # the scores give the indexes of the decoded tokens
        indexes = scores[:, score_index].flatten()
        decoded_tokens = sorted_tokens[torch.arange(indexes.shape[0]), indexes]

        return decoded_tokens.int(), past


    def encode(self, text):
        tokens = self.text_to_tokens(text)
        return self.encode_tokens(tokens)

    def encode_tokens(self, tokens):

        tokens, pad_len = self.pad(tokens, self.tokenizer.eos_token_id)
        tokens = tokens.view(-1, self.CONTEXT_SIZE)

        output_scores = torch.zeros(tokens.shape)


        # Add eos to the start of each block (to give it somewhere to start)
        eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
        tokens = torch.cat((eos, tokens), 1)

        tokens = tokens.to(self.device)

        batches = tokens.shape[0]//self.BATCH_SIZE
        if tokens.shape[0] % self.BATCH_SIZE != 0:
            batches += 1

        # score each batch
        print("Encoding")
        for i in range(batches):
            cur_tokens = tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE]
            cur_output_scores = torch.zeros((cur_tokens.shape[0], cur_tokens.shape[1]-1))
            past = None
            print(i, "out of", batches)
            
            for j in range(cur_tokens.shape[1]-1):

                cur_output_scores[:, j], past = self.encode_one_batch(cur_tokens, j, past)
            output_scores[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] = cur_output_scores

        output_scores = output_scores.flatten().int()
        if pad_len > 0:
            output_scores = output_scores[:-pad_len]
        return output_scores
    
    def decode(self, scores):
        output_tokens = self.decode_tokens(scores)
        text = self.tokenizer.batch_decode(output_tokens)
        text = "".join(text)
        return text
    
    def decode_tokens(self, scores):

        scores, pad_len = self.pad(scores, self.tokenizer.eos_token_id)

        scores = scores.view(-1, self.CONTEXT_SIZE) # all rows, CONTEXT_SIZE

        output_tokens = torch.zeros(scores.shape, dtype=int)


        # Add eos to the start of each block (to give it somewhere to start)
        eos = torch.tensor([self.tokenizer.eos_token_id]*output_tokens.shape[0]).unsqueeze(1)
        output_tokens = torch.cat((eos, output_tokens), 1) # all rows, CONTEXT_SIZE + 1

        output_tokens = output_tokens.to(self.device)

        batches = scores.shape[0]//self.BATCH_SIZE
        if scores.shape[0] % self.BATCH_SIZE != 0:
            batches += 1

        # score each batch
        print("Decoding")
        for i in range(batches):
            print(i, "out of", batches)
            cur_scores = scores[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] # BATCH_SIZE, CONTEXT_SIZE

            cur_output_tokens = output_tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] # BATCH_SIZE, CONTEXT_SIZE
            cur_output_tokens = cur_output_tokens.to(self.device)
            past = None
            for j in range(scores.shape[1]):
 
                cur_output_tokens[:, j+1], past = self.decode_one_batch(cur_output_tokens, cur_scores, j, past) # BATCH_SIZE

            output_tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] = cur_output_tokens
        
        if pad_len != 0:
            output_tokens[-1, -pad_len:] = self.tokenizer.eos_token_id

        output_tokens = output_tokens[:, 1:].int()



        return output_tokens



    def encode_and_zip(self, text):
        encoded = self.encode(text)
        encoded = array.array("H", encoded)
        return zlib.compress(encoded, level=9)
    




