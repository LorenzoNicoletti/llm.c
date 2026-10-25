#!/usr/bin/env python3
"""
Step 2 of the pipeline: the GPT-2 model and the code that trains it.

GPT-2 in one paragraph: text is split into tokens (integers). Each token gets an
embedding vector, plus a vector for its position. These vectors pass through a
stack of transformer blocks; in each block, attention lets every token look at
the tokens before it, and an MLP processes each token on its own. At the end,
the model outputs a score (logit) for every word in the vocabulary: its guess
for the *next* token. Training means showing it real text and nudging all the
weights so the correct next token gets a higher probability (lower loss).

Normally called from main.py: train(data="tinyshakespeare", model="gpt2", ...).
All settings and their defaults are the arguments of train() below.

Adapted from train_gpt2.py in Andrej Karpathy's llm.c (https://github.com/karpathy/llm.c),
MIT license. That file is in turn based on his nanoGPT and on the OpenAI and huggingface
GPT-2 implementations.
"""

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the repo folder, one up from scripts/
DATA_DIR = os.path.join(ROOT_DIR, "data")

# -----------------------------------------------------------------------------
# The GPT-2 model

class CausalSelfAttention(nn.Module):
    """
    Lets each token gather information from the tokens before it.
    Each token emits a query ("what am I looking for"), a key ("what do I contain")
    and a value ("what I pass on"). query·key decides how much attention each
    earlier token gets; the output is the weighted sum of their values.
    "Causal" = a token may never look at future tokens (that would be cheating,
    since predicting the future is the task).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # lower-triangular mask: position t can only see positions <= t
        # (a fixed helper, not a weight: persistent=False keeps it out of the saved weights)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention scores: how well each query matches each key, scaled to keep softmax well-behaved
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf')) # hide the future
        att = F.softmax(att, dim=-1) # each row now sums to 1
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(y)

class MLP(nn.Module):
    """Per-token feed-forward network: expand to 4x width, non-linearity, project back"""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        # the GELU variant used by OpenAI's GPT-2: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        self.gelu    = nn.GELU(approximate="tanh")
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    """
    One transformer block. The "x = x + ..." residual connections mean each block
    only adds a correction to the running representation, which keeps deep stacks trainable.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # tokens talk to each other
        x = x + self.mlp(self.ln_2(x))  # each token thinks on its own
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max context length (tokens the model can look back over)
    vocab_size: int = 50257 # number of distinct tokens
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # attention heads per block
    n_embd: int = 768 # size of each token's vector

# model sizes for training from random weights
SCRATCH_CONFIGS = {
    "tiny": GPTConfig(block_size=256, n_layer=4, n_head=4, n_embd=256), # ~16M params, CPU-friendly
    "d12":  GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768), # GPT-2 small size, 124M params
}

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.SKIP_INIT = True # don't init this one, we will tie weights
        # weight tying: the same matrix maps tokens -> vectors (input) and vectors -> token scores (output)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'RESIDUAL_SCALE_INIT') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long) # shape (t)

        # each token's vector = what the token is + where it is
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # training: a score for every vocabulary token at every position, and the loss.
            # cross-entropy = -log(probability the model gave to the correct next token)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @classmethod
    def from_pretrained(cls):
        """Loads OpenAI's pretrained GPT-2 (124M) weights from huggingface"""
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: gpt2")

        # a randomly initialized model of the same shape (the GPTConfig defaults are GPT-2 124M)
        model = GPT(GPTConfig())
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_hf) == len(sd), f"mismatched keys: {len(sd_hf)} != {len(sd)}"
        for k in sd_hf:
            with torch.no_grad():
                if any(k.endswith(w) for w in transposed):
                    # special treatment for the Conv1D weights we need to transpose
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    sd[k].copy_(sd_hf[k].t())
                else:
                    # vanilla copy over the other parameters
                    assert sd_hf[k].shape == sd[k].shape
                    sd[k].copy_(sd_hf[k])
        return model

    def save_pretrained_hf(self, save_dir):
        """Export weights as a huggingface/transformers GPT2LMHeadModel (inverse of from_pretrained)"""
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        config = GPT2Config(
            vocab_size=self.config.vocab_size,
            n_positions=self.config.block_size,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
        )
        model_hf = GPT2LMHeadModel(config)
        sd_hf = model_hf.state_dict()
        sd = self.state_dict()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd:
            with torch.no_grad():
                if any(k.endswith(w) for w in transposed):
                    # our Linear weights go back to the Conv1D layout
                    assert sd[k].shape[::-1] == sd_hf[k].shape
                    sd_hf[k].copy_(sd[k].t())
                else:
                    assert sd[k].shape == sd_hf[k].shape
                    sd_hf[k].copy_(sd[k])
        os.makedirs(save_dir, exist_ok=True)
        model_hf.save_pretrained(save_dir)
        # the tokenizer is unchanged (tiktoken "gpt2" == GPT2Tokenizer), save it alongside for convenience
        GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(save_dir)
        print(f"saved huggingface model and tokenizer to {save_dir}")

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in self.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # AdamW: gradient descent with per-parameter adaptive step sizes and momentum
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# Data loading: reads the .bin files written by prepare_data.py

def load_tokens(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        if header[0] != 20240520:
            raise SystemExit(f"{filename} is not a token file, re-run prepare_data.py")
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DataLoader:
    """
    Walks through the token file in chunks of B*T+1 tokens.
    x is the chunk without its last token, y is the chunk shifted by one:
    at every position the target is simply the token that comes next.
    """

    def __init__(self, filename, B, T):
        self.B = B
        self.T = T
        self.tokens = load_tokens(filename)
        if len(self.tokens) < B * T + 1:
            raise SystemExit(f"{filename} has only {len(self.tokens):,} tokens, need at least B*T+1 = {B*T+1:,}. "
                             "Use more text or a smaller batch_size / sequence_length.")
        print(f"DataLoader: {len(self.tokens):,} tokens in {filename}")
        self.reset()

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance, and start over from the beginning once we run out of data (a new "epoch")
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.reset()
        return x, y

# -----------------------------------------------------------------------------
# Training

def train(
    data="tinyshakespeare",     # dataset name: the folder under data/ with train.bin and val.bin
    model="gpt2",               # "gpt2" = fine-tune pretrained | "tiny", "d12" = from random weights
    save_dir=os.path.join(ROOT_DIR, "gpt2_model"), # where to save the trained model ("" to skip)
    batch_size=8,               # sequences per forward/backward pass (B)
    sequence_length=128,        # tokens per sequence (T)
    total_batch_size=1024,      # tokens per optimizer step; must be a multiple of B*T
    num_iterations=50,          # number of optimizer steps
    learning_rate=3e-4,         # peak learning rate
    warmup_iters=0,             # steps of linear learning rate warmup
    learning_rate_decay_frac=1.0, # final lr as a fraction of the peak (1.0 = constant)
    weight_decay=0.0,           # AdamW weight decay
    grad_clip=1.0,              # maximum gradient norm
    val_every=10,               # evaluate val loss every N steps (0 = only at the end)
    val_max_steps=20,           # how many val batches to average
    sample_every=0,             # print a generated sample every N steps (0 = only at the end)
):
    """Train GPT-2 (on the CPU) on a dataset prepared with prepare_data.py, then save it to save_dir"""
    B, T = batch_size, sequence_length
    assert model in {"gpt2", *SCRATCH_CONFIGS}, f"unknown model {model}"

    # gradient accumulation: if total_batch_size is bigger than one pass (B*T),
    # run several passes and add up their gradients before each optimizer step
    assert total_batch_size % (B * T) == 0, "total_batch_size must be a multiple of batch_size * sequence_length"
    grad_accum_steps = total_batch_size // (B * T)

    # rng / reproducibility
    torch.manual_seed(42)

    enc = tiktoken.get_encoding("gpt2")

    # the model: either OpenAI's pretrained GPT-2, or random weights
    if model == "gpt2":
        gpt = GPT.from_pretrained()
    else:
        gpt = GPT(SCRATCH_CONFIGS[model])
        print(f"training from scratch: {model} {SCRATCH_CONFIGS[model]}")
    assert T <= gpt.config.block_size, f"sequence_length can be at most {gpt.config.block_size} for this model"
    print(f"model parameters: {sum(p.numel() for p in gpt.parameters()):,}")

    # the data
    data_dir = os.path.join(DATA_DIR, data)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"dataset folder {data_dir} not found, run prepare_data.py first")
    train_loader = DataLoader(os.path.join(data_dir, "train.bin"), B, T)
    val_loader = DataLoader(os.path.join(data_dir, "val.bin"), B, T)

    optimizer = gpt.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(0.9, 0.95))

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = learning_rate * learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it+1) / warmup_iters
        # 2) after that, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (learning_rate - min_lr)

    # we loop for step <= num_iterations so the last step only evaluates and samples
    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = (step == num_iterations)

        # once in a while measure the loss on text the model never trains on.
        # if train loss keeps falling but val loss rises, the model is memorizing (overfitting)
        if (val_every > 0 and step % val_every == 0) or last_step:
            gpt.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_max_steps):
                    x, y = val_loader.next_batch()
                    _, loss = gpt(x, y)
                    val_loss += loss.item()
                val_loss /= val_max_steps
            print(f"val loss {val_loss:.4f}")

        # once in a while generate some text, starting from <|endoftext|>
        if (sample_every > 0 and step % sample_every == 0) or last_step:
            gpt.eval()
            xg = torch.tensor([[enc.eot_token]], dtype=torch.long)
            yg = gpt.generate(xg, max_new_tokens=64, temperature=1.0, top_k=40)
            print('---------------')
            print(enc.decode(yg[0].tolist()))
            print('---------------')

        if last_step:
            break

        # --------------- one training step -----------------
        gpt.train()
        optimizer.zero_grad()
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            # forward pass: predict the next token at every position and measure how wrong we are
            _, loss = gpt(x, y)
            # gradients add up over micro steps, so scale to get the mean instead of the sum
            loss = loss / grad_accum_steps
            lossf += loss.item()
            # backward pass: compute how each weight should change to lower the loss
            loss.backward()
        # cap the gradient size, so one unlucky batch can't wreck the weights
        norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # update the weights
        optimizer.step()
        # ---------------------------------------------------

        t1 = time.time()
        tokens_per_second = total_batch_size / (t1-t0)
        print(f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")

    if save_dir:
        gpt.save_pretrained_hf(save_dir)
    return gpt

if __name__ == "__main__":
    train() # with the defaults above; normally called from main.py
