"""LoRA adapter injection for a small pre-trained causal LM (milestones 3-5).

    h = W0 x + (alpha / r) * B A x      W0 frozen, A (r, in) random, B (out, r) zero-initialised

Because B starts at zero, a freshly injected adapter leaves the model's output unchanged;
training then only learns the low-rank difference.
"""
from __future__ import annotations

import math

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
