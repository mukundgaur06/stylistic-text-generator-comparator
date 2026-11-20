"""Milestone 6: a qualitative, side-by-side comparison of the n-gram model and the LoRA-adapted LLM."""
from __future__ import annotations

import shutil
import textwrap
from typing import Callable

Generator = Callable[[str, int], str]  # (prompt, seed) -> text


def render_side_by_side(left_title: str, left: list[str], right_title: str, right: list[str],
                        width: int | None = None) -> str:
    """Two equal-width columns; row i pairs left[i] with right[i], each wrapped to its column."""
    width = width or shutil.get_terminal_size((100, 20)).columns
    col = max(20, (width - 3) // 2)
    rule = "-" * col + "-+-" + "-" * col
    lines = [f"{left_title:<{col}} | {right_title}", rule]
    for i, (a, b) in enumerate(zip(left, right), 1):
        lines.append(f"#{i}".ljust(col) + " |")
        wa, wb = textwrap.wrap(a, col) or [""], textwrap.wrap(b, col) or [""]
        for j in range(max(len(wa), len(wb))):
            lines.append(f"{wa[j] if j < len(wa) else '':<{col}} | {wb[j] if j < len(wb) else ''}")
        lines.append(rule)
    return "\n".join(lines)


def build_comparison(ngram: Generator, llm: Generator, prompt: str, samples: int = 3,
                     seed: int = 0) -> tuple[list[str], list[str]]:
    """Generate `samples` continuations from each model; sample i uses seed + i for both."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    seeds = [seed + i for i in range(samples)]
    return [ngram(prompt, s) for s in seeds], [llm(prompt, s) for s in seeds]


def run_comparison(style: str, prompt: str = "", samples: int = 3, seed: int = 0, max_tokens: int = 40,
                   temperature: float = 0.8, order: int = 3, width: int | None = None) -> str:
    """Load both models for `style` and return the printable comparison."""
    from app.core import load_sentences, split_corpus, train_ngram
    from app.lora import generate_text, lm_perplexity, load_styled_model

    ngram, val_words = train_ngram(style, order)
    model, tok = load_styled_model(style)
    _, val_text = split_corpus(load_sentences(style))
    left, right = build_comparison(
        lambda p, s: ngram.generate(p, max_tokens, temperature, s),
        lambda p, s: generate_text(model, tok, p, max_tokens, temperature, seed=s),
        prompt, samples, seed)
    header = [
        f"Style: {style} | prompt: {prompt!r} | temperature {temperature} | max {max_tokens} tokens | seed {seed}",
        f"Held-out perplexity: n-gram {ngram.perplexity(val_words):.1f} per word, "
        f"LoRA {lm_perplexity(model, tok, val_text):.1f} per GPT-2 token (different units: not comparable)",
        "",
    ]
    table = render_side_by_side(f"N-gram ({order}-gram, trained from scratch)", left,
                                "LoRA (distilgpt2 + adapter)", right, width)
    return "\n".join(header) + table
