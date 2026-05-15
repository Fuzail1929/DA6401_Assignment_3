"""
dataset.py — Multi30k Dataset Loading and Tokenization
DA6401 Assignment 3
"""

import torch
from torch.utils.data import Dataset
from collections import Counter

import spacy
from datasets import load_dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


class Vocab:
    """Simple vocabulary class."""
    def __init__(self, stoi, itos):
        self.stoi = stoi  # str -> int
        self.itos = itos  # int -> str

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx):
        return self.itos[idx]

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None,
                 src_data=None, tgt_data=None):
        self.split = split

        # Load spacy models
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.spacy_en = spacy.load("en_core_web_sm")

        # Load dataset
        dataset = load_dataset("bentrevett/multi30k")
        self.raw = dataset[split]

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        if src_data is None or tgt_data is None:
            self.src_data, self.tgt_data = self.process_data()
        else:
            self.src_data = src_data
            self.tgt_data = tgt_data

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq=2):
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.raw:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        def build(counter, min_freq):
            itos = SPECIAL_TOKENS + [
                w for w, c in counter.items() if c >= min_freq
            ]
            stoi = {w: i for i, w in enumerate(itos)}
            return Vocab(stoi, itos)

        src_vocab = build(src_counter, min_freq)
        tgt_vocab = build(tgt_counter, min_freq)
        return src_vocab, tgt_vocab

    def process_data(self):
        src_data, tgt_data = [], []
        for example in self.raw:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])
            src_ids = [SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(tgt_tokens) + [EOS_IDX]
            src_data.append(src_ids)
            tgt_data.append(tgt_ids)
        return src_data, tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])


def collate_fn(batch, pad_idx=PAD_IDX):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )
    return src_padded, tgt_padded


def build_datasets(min_freq=2):
    """
    Build train/val/test datasets sharing the same vocabulary.
    Returns: train_ds, val_ds, test_ds, src_vocab, tgt_vocab
    """
    train_ds = Multi30kDataset(split='train')
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    val_ds  = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_ds = Multi30kDataset(split='test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    return train_ds, val_ds, test_ds, src_vocab, tgt_vocab