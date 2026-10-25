#!/usr/bin/env python3
"""
Step 1 of the pipeline: turn raw text into tokens the model can train on.

Takes a text file from the texts/ folder, splits it into tokens with the
GPT-2 tokenizer (tiktoken "gpt2"), and writes two files:

    data/<name>/train.bin   tokens the model learns from
    data/<name>/val.bin     held-out tokens, only used to measure the loss

To use your own text, put a .txt file in texts/, set TEXT_FILE below and run:
    python prepare_data.py                      # texts/my_book.txt -> data/my_book/

File format (.bin): a header of 256 int32 values (magic 20240520, version 1,
number of tokens), followed by the tokens as uint16. GPT-2 has 50,257 tokens,
so every token id fits in 16 bits.

Adapted from dev/data/tinyshakespeare.py and dev/data/data_common.py in
Andrej Karpathy's llm.c (https://github.com/karpathy/llm.c), MIT license.
"""

import os

import numpy as np
import tiktoken

# ============================== settings ======================================

TEXT_FILE = "tinyshakespeare.txt"   # the .txt file in the texts/ folder to prepare
VAL_FRACTION = 0.1                  # fraction of the tokens held out for validation

# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTS_DIR = os.path.join(SCRIPT_DIR, "texts")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MAGIC, VERSION = 20240520, 1

def dataset_name(text_file):
    """The dataset folder name is the file name without .txt: 'my_book.txt' -> 'my_book'"""
    return os.path.splitext(os.path.basename(text_file))[0]

def write_datafile(filename, toks):
    """Save tokens as a .bin file: 256 int32 header, then uint16 tokens"""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(text):
    """
    Split the text into "documents" at blank lines and tokenize each one.
    An <|endoftext|> token goes before every document, so the model learns
    where one piece of text ends and the next begins.
    """
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"] # 50256
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        # keep the blank line at the end of each section, except the last
        tokens.extend(enc.encode_ordinary(s + "\n\n" if i != len(sections) - 1 else s))
    return enc, tokens

def prepare(text_file, val_fraction=VAL_FRACTION):
    """Tokenize texts/<text_file> into data/<name>/{train,val}.bin. Returns the dataset name."""
    name = dataset_name(text_file)
    out_dir = os.path.join(DATA_DIR, name)
    if os.path.exists(os.path.join(out_dir, "train.bin")) and os.path.exists(os.path.join(out_dir, "val.bin")):
        print(f"data/{name}/ already exists, pre-processing is skipped and the existing tokens are used.")
        print(f"(if you changed {text_file}, delete data/{name}/ to tokenize it again)")
        return name

    path = os.path.join(TEXTS_DIR, text_file)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found, put your .txt file in the texts/ folder")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"read {len(text):,} characters from {path}")

    enc, tokens = tokenize(text)
    print(f"{len(tokens):,} tokens ({len(text) / len(tokens):.1f} characters per token on average)")
    sample = tokens[1:21]
    print("first tokens:", [enc.decode([t]) for t in sample])
    print("as ids:      ", sample)

    # the last val_fraction of the tokens is held out for validation
    os.makedirs(out_dir, exist_ok=True)
    n_val = int(len(tokens) * val_fraction)
    write_datafile(os.path.join(out_dir, "train.bin"), tokens[:len(tokens) - n_val])
    write_datafile(os.path.join(out_dir, "val.bin"), tokens[len(tokens) - n_val:])
    return name

if __name__ == "__main__":
    prepare(TEXT_FILE)
