"""LoRA adapter injection for a small pre-trained causal LM (milestones 3-5).

    h = W0 x + (alpha / r) * B A x      W0 frozen, A (r, in) random, B (out, r) zero-initialised

Because B starts at zero, a freshly injected adapter leaves the model's output unchanged;
training then only learns the low-rank difference.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

DEFAULT_MODEL = "distilgpt2"
DEFAULT_TARGETS = ("c_attn", "c_proj")  # GPT-2 attention (and MLP output) projections


# --- Milestone 3: Load a small pre-trained LLM and implement LoRA adapter injection.
class LoRALinear(nn.Module):
    """A frozen base layer (nn.Linear or GPT-2 Conv1D) plus a trainable rank-r update."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("rank r must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        if isinstance(base, Conv1D):
            in_f, out_f = base.weight.shape       # Conv1D stores (in, out)
        elif isinstance(base, nn.Linear):
            in_f, out_f = base.in_features, base.out_features
        else:
            raise TypeError(f"unsupported layer {type(base).__name__}")
        if r > min(in_f, out_f):
            raise ValueError(f"rank {r} exceeds min(in, out) = {min(in_f, out_f)}")
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def load_base_model(name: str = DEFAULT_MODEL):
    """Download (first run only) and return (model, tokenizer) for a small pre-trained LM."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return AutoModelForCausalLM.from_pretrained(name), tokenizer


def tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    """A small randomly initialised GPT-2 (no download); used by the tests."""
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(cfg).eval()


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                targets: tuple[str, ...] = DEFAULT_TARGETS) -> nn.Module:
    """Freeze every base weight, then wrap each target layer with a LoRALinear (in place)."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, (Conv1D, nn.Linear)):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no layers named {targets} found to adapt")
    return model


def lora_layers(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_adapter_enabled(model: nn.Module, enabled: bool) -> None:
    for layer in lora_layers(model):
        layer.enabled = enabled


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


# --- Milestone 4: Fine-tune the LoRA adapter on the sports commentary corpus.
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"


def _batch(tokenizer, sentences: list[str], max_len: int = 64):
    """Tokenise sentences + EOS; pad positions are masked out of the loss with -100."""
    enc = [tokenizer(s, truncation=True, max_length=max_len - 1)["input_ids"] + [tokenizer.eos_token_id]
           for s in sentences]
    width = max(map(len, enc))
    ids = torch.full((len(enc), width), tokenizer.pad_token_id)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for i, seq in enumerate(enc):
        ids[i, :len(seq)], mask[i, :len(seq)] = torch.tensor(seq), 1
    labels = ids.masked_fill(mask == 0, -100)
    return ids, mask, labels


@torch.no_grad()
def lm_perplexity(model, tokenizer, sentences: list[str]) -> float:
    """Per-token perplexity (GPT-2 tokens, so not comparable to the word-level n-gram number)."""
    model.eval()
    ids, mask, labels = _batch(tokenizer, sentences)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                       ignore_index=-100)
    return math.exp(loss.item())


def train_adapter(model, tokenizer, train: list[str], val: list[str] | None = None, epochs: int = 10,
                  lr: float = 2e-3, batch_size: int = 8, seed: int = 0) -> list[dict]:
    """Train only the LoRA parameters; returns one history row per epoch."""
    if not lora_layers(model):
        raise ValueError("no LoRA layers found: call inject_lora first")
    params = [p for p in model.parameters() if p.requires_grad]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    history, best = [], (float("inf"), None)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        losses = []
        for i in range(0, len(order), batch_size):
            ids, mask, labels = _batch(tokenizer, [train[j] for j in order[i:i + batch_size]])
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
        if val:
            row["val_ppl"] = lm_perplexity(model, tokenizer, val)
            if row["val_ppl"] < best[0]:   # tiny corpus: remember the best epoch, it overfits fast
                best = (row["val_ppl"], {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k})
        history.append(row)
    if best[1]:
        model.load_state_dict(best[1], strict=False)
    return history


def save_adapter(model, style: str, directory=ADAPTER_DIR):
    """Save just the adapter tensors (a couple of MB), not the base model."""
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    path = directory / f"{style}.pt"
    torch.save(state, path)
    return path


def load_adapter(model, style: str, directory=ADAPTER_DIR) -> None:
    state = torch.load(directory / f"{style}.pt", map_location="cpu")
    missing = model.load_state_dict(state, strict=False).unexpected_keys
    if missing:
        raise ValueError(f"adapter does not match this model: {missing[:3]}")


def finetune_style(style: str, epochs: int = 10, r: int = 8, model_name: str = DEFAULT_MODEL):
    """End to end: load base, inject, train on the style's training split, save the adapter."""
    from app.core import load_sentences, split_corpus
    model, tok = load_base_model(model_name)
    train, val = split_corpus(load_sentences(style))
    base_ppl = lm_perplexity(model, tok, val)
    inject_lora(model, r=r)
    history = train_adapter(model, tok, train, val, epochs=epochs)
    path = save_adapter(model, style)
    best = min(history, key=lambda row: row["val_ppl"])
    return {"style": style, "base_val_ppl": base_ppl, "history": history, "adapter": str(path),
            "best_epoch": best["epoch"], "best_val_ppl": best["val_ppl"]}


# --- Milestone 5: Generate text from the LoRA-adapted LLM.
@torch.no_grad()
def generate_text(model, tokenizer, prompt: str = "", max_new_tokens: int = 40, temperature: float = 1.0,
                  top_k: int = 40, seed: int | None = None) -> str:
    """Sample a continuation of `prompt`; stops at end-of-sequence. Same seed -> same text."""
    model.eval()
    gen = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
    ids = tokenizer(prompt)["input_ids"] if prompt else []
    prompt_len = len(ids)
    ids = ids or [tokenizer.eos_token_id]           # an empty prompt starts from the EOS token
    max_ctx = getattr(model.config, "n_positions", 1024)
    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-max_ctx:]])
        logits = model(input_ids=window).logits[0, -1]
        if temperature <= 0:                        # greedy
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if 0 < top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        if nxt == tokenizer.eos_token_id:
            break
        ids.append(nxt)
    start = 0 if prompt_len else 1
    return tokenizer.decode(ids[start:]).strip()


def load_styled_model(style: str, model_name: str = DEFAULT_MODEL, r: int = 8):
    """Base model with the style's saved adapter loaded: returns (model, tokenizer)."""
    model, tok = load_base_model(model_name)
    inject_lora(model, r=r)
    load_adapter(model, style)
    return model.eval(), tok
