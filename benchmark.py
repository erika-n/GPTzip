from gptzip import GPTZip as GPTZip1

from gptzip_2 import GPTZip as GPtZip2
import re
import zlib
import array

def benchmark2(gpt_zip):
    with open("sometext.txt", encoding='utf-8') as f:
        text = f.read()

    # following the paper, make our test text just lowercase and space
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    # text = text[:10000]

    gpt_zip.CONTEXT_SIZE = 1024
    gpt_zip.BATCH_SIZE = 10
    
    encoded = gpt_zip.encode(text)
    print(encoded[:150])
    zip_encoded = zlib.compress(array.array("H", encoded), level=9)
    zip_unencoded = zlib.compress(text.encode('utf-8', 'ignore'), level=9)
    print(f"{len(text)=}")
    print(f"{len(zip_encoded)=}")
    print(f"{len(zip_unencoded)=}")
    print(f"{len(zip_encoded)/len(zip_unencoded)=}")

def benchmark(gpt_zip):
    text = '''Arching under the night sky inky
with black expansiveness, we point
to the planets we know, we

pin quick wishes on stars. From earth,
we read the sky as if it is an unerring book
of the universe, expert and evident.
'''
    gpt_zip.CONTEXT_SIZE = 5
    
    encoded = gpt_zip.encode(text)
    print(f"{encoded[:11]}")
    decoded = gpt_zip.decode(encoded)
    #print("decoded", decoded)
    assert text == decoded
    # orig_tokens = gpt_zip.text_to_tokens(text)

    # word_tokens = []
    # for token in orig_tokens:
    #     word_tokens.append(gpt_zip.tokenizer.batch_decode([[token]]) )

    # for i in range(len(word_tokens)):
    #     print(f"{repr(word_tokens[i][0])}\t {encoded[i]}")

    assert text == decoded


if __name__ == "__main__":
    gpt_zip_1 = GPTZip1()
    gpt_zip_2 = GPtZip2()
    # print("GPTZip 1:")
    # benchmark2(gpt_zip_1)
    print("GPTZip 2")
    benchmark2(gpt_zip_2)

