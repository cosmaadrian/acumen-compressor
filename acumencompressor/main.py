# Heavily based on https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py

import unicodedata
import dahuffman
import tqdm
from collections import Counter, defaultdict

def get_stats(ids, counts = None):
    counts = dict(Counter(zip(ids, ids[1:])))
    return counts

def dsum(dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

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

class BPETokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        self.merges = {} # (int, int) -> int

    def train_on_lists(self, list_of_texts, vocab_size, verbose):
        vocab_size = max(vocab_size, 256) # cannot have vocab size smaller than number of 8-bit bytes
        num_merges = vocab_size - 256

        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        
        if verbose:
            range_fn = tqdm.trange
        else:
            range_fn = range

        # secret sauce?
        ids = sorted([list(text_bytes) for text_bytes in list_of_texts])

        for i in range_fn(num_merges):
            stats = dsum([
                get_stats(sub_ids)
                for sub_ids in ids
            ])
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            ids = [
                merge(sub_ids, pair, idx)
                for sub_ids in ids
            ]
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        return merges, vocab

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

    def decode(self, ids, vocab):
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

    def compress_list(self, list_of_targets):
        merges, vocab = self.bpe.train_on_lists(list_of_targets, vocab_size = self.vocab_size, verbose = self.verbose)

        ids = [
            self.bpe.encode(t, merges, vocab)
            for t in list_of_targets
        ]

        codec = self.huffman.from_data(sorted(sum(ids, [])))

        self.codec = codec
        self.merges = merges
        self.vocab = vocab

        compressed = [
            codec.encode(sub_ids)
            for sub_ids in ids
        ]

        return compressed

    def uncompress_list(self, list_of_targets):
        ids = [
            self.codec.decode(compressed_ids)
            for compressed_ids in list_of_targets
        ]
        uncompressed = [
            self.bpe.decode(sub_ids, vocab = self.vocab)
            for sub_ids in ids
        ]

        return uncompressed

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
        uncompressed = self.bpe.decode(ids, vocab = self.vocab)
        return uncompressed

   