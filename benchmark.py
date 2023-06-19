from gptzip import GPTZip as GPTZip1

from gptzip_2 import GPTZip as GPtZip2
import re
import zlib
import array

def benchmark(gpt_zip):
    with open("sometext.txt", encoding='utf-8') as f:
        text = f.read()

    # following the paper, make our test text just lowercase and space
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    gpt_zip.CONTEXT_SIZE = 1024
    gpt_zip.BATCH_SIZE = 25
    
    encoded = gpt_zip.encode(text)
    print(encoded[:150])
    zip_encoded = zlib.compress(array.array("H", encoded), level=9)
    zip_unencoded = zlib.compress(text.encode('utf-8', 'ignore'), level=9)
    print(f"{len(text)=}")
    print(f"{len(zip_encoded)=}")
    print(f"{len(zip_unencoded)=}")
    print(f"{len(zip_encoded)/len(zip_unencoded)=}")
    bpc = len(zip_encoded)*8/len(text)
    print(f"encoded bits per character: {bpc}")



if __name__ == "__main__":
    gpt_zip_1 = GPTZip1()
    gpt_zip_2 = GPtZip2()
    # print("GPTZip 1:")
    # benchmark2(gpt_zip_1)
    print("GPTZip 2")
    benchmark(gpt_zip_2)

