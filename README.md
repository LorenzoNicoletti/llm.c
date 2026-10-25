# Train GPT-2 on any text

A small, readable project for understanding how a model like GPT-2 works and how it is trained.
It trains on any text file (TinyShakespeare by default) in three steps: **pre-processing → training → test**.

Trimmed down from Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) (MIT license, see
[LICENSE](LICENSE)). The GPT-2 model, weight loading, training loop and token file format are his work.
This version keeps only the PyTorch training path, running on the CPU, and adds explanatory comments,
training on any text file, a main script and a test script.

## How GPT-2 works, in short

1. **Tokens.** Text is cut into pieces called tokens (whole words, parts of words or punctuation), and each
   token is a number between 0 and 50,256. `" king"` → 5822. Common words are one token; rare words are split.
2. **Embeddings.** Each token number picks a learned vector (768 numbers in GPT-2 small), and a second
   vector encodes its position in the text. The two are added together.
3. **Transformer blocks** (12 in GPT-2 small). Each block has:
   - **attention**: every token looks back at the earlier tokens and pulls in what is relevant
     ("who is *he*?" → looks at the name mentioned before);
   - an **MLP**: a small neural network applied to each token on its own.
4. **Next-token scores.** The final vectors are turned into a score for every one of the 50,257 tokens:
   the model's guess for which token comes **next**.
5. **Training.** Show it real text, measure how much probability it gave to the actual next token
   (the **loss**), use backpropagation to work out how each weight should change, and nudge all weights
   a little (the **AdamW** optimizer). Repeat.
6. **Generation.** Pick a next token from the scores, append it, and repeat.

All of this is in [scripts/train_gpt2.py](scripts/train_gpt2.py); the comments there walk through each part.

## Setup

Requires Python 3.10+. [setup_env.py](setup_env.py) creates a virtual environment and downloads the packages needed for running this example (see [requirements.txt](requirements.txt)).

Remember to activate the environment before running anything.

## Quick start

Open [main.py](main.py), check the settings at the top, and run it. With the default settings it
tokenizes TinyShakespeare, fine-tunes GPT-2 for 50 steps, saves the model to `gpt2_model/`, and
generates text from 4 prompts.

`main.py` has three sections and each section calls one script in [scripts/](scripts/); each script can also run on its own, using the
settings at its top or its defaults.

### 1) Pre-processing — [prepare_data.py](scripts/prepare_data.py)

Put your text in the [texts/](texts/) folder (TinyShakespeare, from Karpathy's
[char-rnn](https://github.com/karpathy/char-rnn), is already there) and set
`TEXT_FILE = "my_book.txt"` in `main.py`. The tokens are written to `data/my_book/`.

It tokenizes the text and writes `train.bin` (90%) and `val.bin` (10%, never trained on; used to check the
model generalizes). Blank lines separate "documents", each one starting with an `<|endoftext|>` token.
Any plain UTF-8 text works. More text is better: TinyShakespeare is ~340k tokens.

If `data/<name>/` already exists, pre-processing is skipped (with a message). If you edit the text file, delete `data/<name>/` so it gets tokenized again.

### 2) Train — [train_gpt2.py](scripts/train_gpt2.py)

`MODEL` picks the starting point:

| `MODEL` | Starts from | Size | Good for |
|---|---|---|---|
| `"gpt2"` (default) | OpenAI's pretrained weights | 124M params | readable text after a few dozen steps; it already knows English and learns your style |
| `"tiny"` | random weights | ~16M params | seeing learning from zero on a CPU: babble → words → style (use ~300 steps, learning rate 1e-3) |
| `"d12"` | random weights | 124M params | the full GPT-2 size from scratch (very slow on a CPU, needs lots of text) |

What the output means:

```
step   10/50 | train loss 3.41 | norm 2.10 | lr 3.00e-04 | (2100 ms | 487 tok/s)
val loss 3.52
```

- **train loss**: how wrong the model is on the batch it learned from (lower is better).
  A random model starts at about 10.8 (= ln 50257).
- **val loss**: the same on held-out text. If train loss keeps falling but val loss rises, the model
  is memorizing instead of learning (overfitting).
- **norm**: size of the gradient (clipped to 1.0). **lr**: current learning rate.

Each step (iteration) is one weight update, computed from `TOTAL_BATCH_SIZE` tokens. If that is more than
`BATCH_SIZE × SEQUENCE_LENGTH`, several passes are added up first (gradient accumulation).
`train()` has a few more options (warmup, learning rate decay, weight decay); see its definition. Training always runs on the CPU.

Every run starts from the chosen starting point and **overwrites** `gpt2_model/`.

### 3) Test — [scripts/test_gpt2_model.py](scripts/test_gpt2_model.py)

Loads `gpt2_model/` with huggingface `transformers` and continues each of the `PROMPTS`.
`TEMPERATURE` below 1 is more predictable, above 1 more random.

## Further learning

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video building
  this model and training loop step by step.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): his series from the basics
  (gradients, backpropagation) up to GPT.
