<h1 align="center"><span style="font-weight:normal"> <img src="https://github.com/cosmaadrian/acumen-template/blob/master/assets/icon.png" alt="drawing" style="width:30px;"/> Acumen 🗜️ Compressor 🗜️</h1>

Coded with love and coffee ☕ by [Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en). But I need more coffee!

<a href="https://www.buymeacoffee.com/cosmadrian" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

# Description
**`AcumenCompressor`** is designed as a simple data compression tool, using [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding), combined with [Huffman codes](https://en.wikipedia.org/wiki/Huffman_coding). Compared to other compression algorithms like LZ77 / LZSS, which use a sliding-window approach to compression (i.e. gzip), `BPE+HC` is **deterministic** and should be agnostic to the order of target elements when compressing multiple files.

Moreover, the combination of `BPE+HC` obtains a better compression ratio compared to `gzip` at the expense of computation time.

## But why?
When compressing a dataset, the order of samples in the dataset is random and using a tool like `gzip` will give slightly differently-sized compressed values depending on the order of examples. When conducting reproducible experiments, I found it important to have a deterministic .

For example, in the paper [gzip Predicts Data-dependent Scaling Laws](https://arxiv.org/abs/2405.16684), the author used gzip to compress the dataset, but did not take into account the order of elements. While this is negligeable, using **`AcumenCompressor`** will give the same compressed value for a dataset.

## But how?
This project makes heavy use of a modified version of [karpathy/minbpe](https://github.com/karpathy/minbpe) to compute the byte-pairs and uses [soxofaan/dahuffman](https://github.com/soxofaan/dahuffman) for arithmetic coding.

# Installation

Install the pypi package via pip:

```bash
pip install -U acumencompressor
```

Alternatively, install directly via git:
```bash
pip install -U git+https://github.com/cosmaadrian/acumen-compressor
```

# Usage

### Compressing a String

```python
import acumencompressor as ac

import string
import random
import gzip

# Make a test string.

test_corpus = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(1000))
compressor = ac.AcumenCompressor(vocab_size = 512, verbose = True)

compressed = compressor.compress(test_corpus.encode('utf-8'))
uncompressed = compressor.uncompress(compressed).decode('utf-8', errors = 'replace')

assert test_corpus == uncompressed
assert len(compressed) < len(test_corpus.encode('utf-8')), "No compression performed!!!"

compressed_gzip = gzip.compress(test_corpus.encode('utf-8'))

print("Size before compression:", len(test_corpus.encode('utf-8')))
# 1000

print("Size after compression:", len(compressed))
# 485

print("Size after gzip compression:", len(compressed_gzip))
# 695

print("Compression ratio:", round((1 - len(compressed) / len(test_corpus.encode('utf-8'))) * 100, 2), '%')
# 51.5 %

print("GZip Compression ratio:", round((1 - len(compressed_gzip) / len(test_corpus.encode('utf-8'))) * 100, 2), '%')
# 30.5 %
```

### Compressing a list of strings

Use the function `AcumenCompressor.compress_list()` to compress a list of strings regardless of order.

```python
import acumencompressor as ac

import string
import random
import gzip

corpora = [
    ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(1000))
    for _ in range(100)
]

sizes_gzip = []
sizes_ours = []
sizes_ours_orderless = []
for _ in range(10):
    random.shuffle(corpora)
    target_permutation = "".join(corpora)

    # gzip
    compressed_gzip = gzip.compress(target_permutation.encode('utf-8'))
    sizes_gzip.append(len(compressed_gzip))

    # AcumenCompressor
    ac = AcumenCompressor(vocab_size = 300, verbose = True)
    compressed_ac = ac.compress(target_permutation.encode('utf-8'))
    sizes_ours.append(len(compressed_ac))

    # AcumenCompressor order agnostic
    ac = AcumenCompressor(vocab_size = 300, verbose = True)
    compressed_ac = ac.compress_list([c.encode('utf-8') for c in corpora])
    sizes_ours_orderless.append(len(b"".join(compressed_ac)))

print('gzip Sizes:', sizes_gzip, np.mean(sizes_gzip), np.std(sizes_gzip))
print('AC Sizes:', sizes_ours, np.mean(sizes_ours), np.std(sizes_ours))
print('AC (orderless) Sizes:', sizes_ours_orderless, np.mean(sizes_ours_orderless), np.std(sizes_ours_orderless))

```

# License
This repository uses [MIT License](LICENSE).
