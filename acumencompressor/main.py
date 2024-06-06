# Heavily based on https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py

import unicodedata
import dahuffman
import tqdm

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class BPETokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose = False):
        vocab_size = max(vocab_size, 256) # cannot have vocab size smaller than number of 8-bit bytes

        num_merges = vocab_size - 256

        text_bytes = text
        ids = list(text_bytes) 

        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        
        if verbose:
            range_fn = tqdm.trange
        else:
            range_fn = range

        for i in range_fn(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        return merges, vocab

    def decode(self, ids, merges, vocab):
        text_bytes = b"".join(vocab[idx] for idx in ids)
        return text_bytes

    def encode(self, target, merges, vocab): # should be bytes, returns list of numbers
        text_bytes = target
        ids = list(text_bytes) 
        
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: merges.get(p, float("inf")))
            
            if pair not in merges:
                break

            idx = merges[pair]
            ids = merge(ids, pair, idx)

        return ids

class AcumenCompressor():
    def __init__(self, vocab_size, use_huffman = True, verbose = False):
        super().__init__()
        self.bpe = BPETokenizer()
        self.huffman = dahuffman.HuffmanCodec
        
        self.vocab_size = vocab_size
        self.verbose = verbose

    def compress(self, target): # should be utf-8 bytes, returns bytes
        merges, vocab = self.bpe.train(target, vocab_size = self.vocab_size, verbose = self.verbose)
        ids = self.bpe.encode(target, merges, vocab)
        codec = self.huffman.from_data(ids)

        self.codec = codec
        self.merges = merges
        self.vocab = vocab

        compressed = codec.encode(ids)
        return compressed

    def uncompress(self, compressed_text):
        ids = self.codec.decode(compressed_text)
        uncompressed = self.bpe.decode(ids, merges = self.merges, vocab = self.vocab)
        return uncompressed

   