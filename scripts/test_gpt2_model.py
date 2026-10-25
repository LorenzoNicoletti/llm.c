#!/usr/bin/env python3
"""
Step 3 of the pipeline: load the trained model and generate text.

The model is loaded with huggingface transformers from gpt2_model/
(written by train_gpt2.py at the end of training).
Normally called from main.py; to run on its own, set the prompts below and run it.
"""

import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================== settings ======================================

PROMPTS = ["To be or"]
MAX_NEW_TOKENS = 100     # tokens to generate per prompt
TEMPERATURE = 0.8       # <1 = safer, more predictable; >1 = more random
TOP_P = 0.95            # only sample from the most likely tokens that together make up top_p probability

# ==============================================================================

# written by train_gpt2.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the repo folder, one up from scripts/
MODEL_DIR = os.path.join(ROOT_DIR, "gpt2_model")

def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P):
    """Generate a continuation of prompt"""
    print(f"Prompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("=" * 60)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("=" * 60)
    print()

def test(prompts=PROMPTS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, model_dir=MODEL_DIR):
    """Load the saved model from model_dir and continue each prompt"""
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise SystemExit(f"no model found in {model_dir}, train one first (run main.py)")

    print(f"Loading model from {model_dir}...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.eval()
    print()

    for prompt in prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p)

if __name__ == "__main__":
    test()
