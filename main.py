#!/usr/bin/env python3
"""
Main script: the whole pipeline in three sections.
Edit the settings, then run the whole file, or run one section at a time with
"Run Cell" in VS Code (the "# %%" lines mark the sections, like MATLAB's "%%").

    1) Pre-processing   texts/<TEXT_FILE>  ->  data/<name>/train.bin, val.bin
    2) Train            GPT-2 on those tokens  ->  gpt2_model/
    3) Test             load gpt2_model/ and generate text
"""

# %% Settings

# the text to train on: a .txt file in the texts/ folder
TEXT_FILE = "tinyshakespeare.txt"

# "gpt2" = fine-tune OpenAI's pretrained GPT-2 (readable text after a few dozen steps)
# "tiny" = small model from random weights (try NUM_ITERATIONS = 300, LEARNING_RATE = 1e-3)
MODEL = "gpt2"

NUM_ITERATIONS = 50             # optimizer steps (= weight updates)
LEARNING_RATE = 3e-4            # peak learning rate
BATCH_SIZE = 8                  # sequences per pass (B)
SEQUENCE_LENGTH = 128           # tokens per sequence (T)
TOTAL_BATCH_SIZE = 1024         # tokens per optimizer step (multiple of B*T)
VAL_EVERY = 10                  # print val loss every N steps

PROMPTS = ["To be or"]
MAX_NEW_TOKENS = 60             # tokens to generate per prompt
TEMPERATURE = 0.8               # <1 = safer, more predictable; >1 = more random

from scripts.prepare_data import prepare
from scripts.train_gpt2 import train
from scripts.test_gpt2_model import test

# %% 1) Pre-processing
print(f"=== STARTED PRE-PROCESSING ({TEXT_FILE}) ===")
name = prepare(TEXT_FILE)

# %% 2) Train
print(f"\n=== STARTED TRAINING ({MODEL}, {NUM_ITERATIONS} steps) ===")
train(
    data=name,
    model=MODEL,
    num_iterations=NUM_ITERATIONS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    total_batch_size=TOTAL_BATCH_SIZE,
    val_every=VAL_EVERY,
)

# %% 3) Test
print("\n=== STARTED TEST ===")
test(prompts=PROMPTS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
