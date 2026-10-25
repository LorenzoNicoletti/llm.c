# Train GPT-2 on any text

A small, readable project for understanding how a model like GPT-2 works and how it is trained.
It trains on any text file (TinyShakespeare by default) in three steps: **prepare → train → generate**.

Trimmed down from Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). Only the
PyTorch training path is kept; the C/CUDA engine, multi-GPU code and evals were removed.

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

All of this is in [train_gpt2.py](train_gpt2.py); the comments there walk through each part.

## Setup

Requires Python 3.10+. From this folder:

```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (macOS/Linux)
python -m pip install -r requirements.txt
```

In VS Code, select `venv` as the interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*) so the
Run button uses it.

| Package | Why it's needed |
|---|---|
| `torch` | neural network library: tensors, backpropagation, optimizer |
| `numpy` | reads and writes the token `.bin` files |
| `tiktoken` | GPT-2 tokenizer (text → tokens) |
| `transformers` | downloads pretrained GPT-2, saves and loads the trained model |

## Quick start

Open [main.py](main.py), check the settings at the top, and run it:

```
python main.py
```

With the default settings it tokenizes TinyShakespeare, fine-tunes GPT-2 for 50 steps, saves the model
to `gpt2_model/`, and generates text from 4 prompts.

`main.py` has three sections marked with `# %%`. In VS Code, each one gets a **Run Cell** button, like
*Run Section* in MATLAB. Variables stay in memory between cells, so you can train once and re-run only
the Test section with new prompts. (The first time, VS Code asks to install `ipykernel` into the venv.)

## The three sections

### 1) Pre-processing — [prepare_data.py](prepare_data.py)

Put your text in the [texts/](texts/) folder (TinyShakespeare is already there) and set
`TEXT_FILE = "my_book.txt"` in `main.py`. The tokens are written to `data/my_book/`.

It tokenizes the text and writes `train.bin` (90%) and `val.bin` (10%, never trained on; used to check the
model generalizes). Blank lines separate "documents", each one starting with an `<|endoftext|>` token.
Any plain UTF-8 text works. More text is better: TinyShakespeare is ~340k tokens.

If `data/<name>/` already exists, pre-processing is skipped (with a message) and the existing tokens are
used. If you edit the text file, delete `data/<name>/` so it gets tokenized again.

### 2) Train — [train_gpt2.py](train_gpt2.py)

`MODEL` picks the starting point:

| `MODEL` | Starts from | Size | Good for |
|---|---|---|---|
| `"gpt2"` (default) | OpenAI's pretrained weights | 124M params | readable text after a few dozen steps; it already knows English and learns your style |
| `"tiny"` | random weights | ~16M params | seeing learning from zero on a CPU: babble → words → style (use ~300 steps, learning rate 1e-3) |
| `"d12"` | random weights | 124M params | the full GPT-2 size from scratch (needs a GPU and lots of text) |

What the output means:

```
step   10/50 | train loss 3.41 | norm 2.10 | lr 3.00e-04 | (2100 ms | 487 tok/s)
val loss 3.52
```

- **train loss**: how wrong the model is on the batch it just learned from (lower is better).
  A random model starts at about 10.8 (= ln 50257).
- **val loss**: the same on held-out text. If train loss keeps falling but val loss rises, the model
  is memorizing instead of learning (overfitting).
- **norm**: size of the gradient (clipped to 1.0). **lr**: current learning rate.

Each step (iteration) is one weight update, computed from `TOTAL_BATCH_SIZE` tokens. If that is more than
`BATCH_SIZE × SEQUENCE_LENGTH`, several passes are added up first (gradient accumulation).
`train()` has a few more options (warmup, learning rate decay, weight decay, device); see its definition.

Every run starts from the chosen starting point and **overwrites** `gpt2_model/`.

### 3) Test — [test_gpt2_model.py](test_gpt2_model.py)

Loads `gpt2_model/` with huggingface `transformers` and continues each of the `PROMPTS`.
`TEMPERATURE` below 1 is more predictable, above 1 more random.

Each of the three files can also run on its own (`python prepare_data.py`, `python train_gpt2.py`,
`python test_gpt2_model.py`), using the settings at its top or its defaults.

## Files

| Path | What it is |
|---|---|
| `main.py` | the main script: settings, then pre-processing, training and test |
| `prepare_data.py` | `texts/<name>.txt` → `data/<name>/train.bin`, `val.bin` |
| `train_gpt2.py` | the GPT-2 model, data loader and training loop |
| `test_gpt2_model.py` | load the trained model and generate text |
| `texts/` | the text files to train on (TinyShakespeare included) |
| `data/` | generated token files (git-ignored) |
| `gpt2_model/` | trained model: `model.safetensors` (weights), `config.json` (shape), tokenizer files (git-ignored) |

## Credits

This project is a fork of [karpathy/llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy,
used under the MIT license (see [LICENSE](LICENSE)). The original work is his:

- the GPT-2 model, weight loading and training loop in `train_gpt2.py` come from llm.c's
  `train_gpt2.py`, itself a version of his [nanoGPT](https://github.com/karpathy/nanoGPT);
- the tokenization and `.bin` format in `prepare_data.py` come from llm.c's `dev/data/` scripts;
- the TinyShakespeare text (`texts/tinyshakespeare.txt`) is from his [char-rnn](https://github.com/karpathy/char-rnn) project.

This fork removes the C/CUDA code and multi-GPU support, adds explanatory comments, data preparation
for any text, export to huggingface format and a test script, all for learning purposes.

To learn this properly, use his material: the video
[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), the
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series, and the original
[llm.c](https://github.com/karpathy/llm.c) repo.
