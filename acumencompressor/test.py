from main import AcumenCompressor
import string
import random

test_corpus = '''
    lololololololololo
'''

test_corpus = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(100000))

if __name__ == '__main__':
    ac = AcumenCompressor(vocab_size = 1024, verbose = True)

    compressed = ac.compress(test_corpus.encode('utf-8'))
    # print(compressed)

    uncompressed = ac.uncompress(compressed).decode('utf-8', errors = 'replace')
    # print(uncompressed)

    assert test_corpus == uncompressed
    assert len(compressed) < len(test_corpus.encode('utf-8')), "No compression performed!!!"

    print("Size before compression:", len(test_corpus.encode('utf-8')))
    print("Size after compression:", len(compressed))

    print("Compression ratio:", round((1 - len(compressed) / len(test_corpus.encode('utf-8'))) * 100, 2), '%')
    