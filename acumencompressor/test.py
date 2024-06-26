from main import AcumenCompressor
import string
import random
import gzip
import numpy as np
import time


def test_sanity():
    test_corpus = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(1000))
    ac = AcumenCompressor(vocab_size = 300, verbose = True)

    start_time = time.time()
    compressed = ac.compress(test_corpus.encode('utf-8'))
    end_time = time.time()
    uncompressed = ac.uncompress(compressed).decode('utf-8', errors = 'replace')

    assert test_corpus == uncompressed
    assert len(compressed) < len(test_corpus.encode('utf-8')), "No compression performed!!!"

    gzip_start_time = time.time()
    compressed_gzip = gzip.compress(test_corpus.encode('utf-8'))
    gzip_end_time = time.time()

    print("Size before compression:", len(test_corpus.encode('utf-8')))
    print("Size after compression:", len(compressed))
    print("Size after gzip compression:", len(compressed_gzip))

    print("Time taken for AcumenCompressor:", end_time - start_time)
    print("Time taken for GZip:", gzip_end_time - gzip_start_time)

    print("Compression ratio:", round((1 - len(compressed) / len(test_corpus.encode('utf-8'))) * 100, 2), '%')
    print("GZip Compression ratio:", round((1 - len(compressed_gzip) / len(test_corpus.encode('utf-8'))) * 100, 2), '%')


def test_ordering():
    corpora = [
        ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(1000))
        for _ in range(100)
    ]

    sizes_gzip = []
    sizes_ours = []
    sizes_ours_orderless = []
    for _ in range(10):
        # 100 permutations
        random.shuffle(corpora)
        target_permutation = "".join(corpora)

        # gzip
        compressed_gzip = gzip.compress(target_permutation.encode('utf-8'))
        sizes_gzip.append(len(compressed_gzip))

        # # AcumenCompressor
        ac = AcumenCompressor(vocab_size = 300, verbose = True)
        compressed_ac = ac.compress(target_permutation.encode('utf-8'))
        sizes_ours.append(len(compressed_ac))

        # AcumenCompressor order agnostic
        ac = AcumenCompressor(vocab_size = 300, verbose = True)
        compressed_ac = ac.compress_list([c.encode('utf-8') for c in corpora])
        sizes_ours_orderless.append(len(b"".join(compressed_ac)))

        # print(ac.merges)

    print('gzip Sizes:', sizes_gzip, np.mean(sizes_gzip), np.std(sizes_gzip))
    print('AC Sizes:', sizes_ours, np.mean(sizes_ours), np.std(sizes_ours))
    print('AC (orderless) Sizes:', sizes_ours_orderless, np.mean(sizes_ours_orderless), np.std(sizes_ours_orderless))

if __name__ == '__main__':
    test_sanity()
    test_ordering()