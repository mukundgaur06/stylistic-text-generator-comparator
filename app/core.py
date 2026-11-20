"""Corpus preparation and the n-gram model (milestones 1-2). The LoRA side lives in app/lora.py."""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?]")


# --- Milestone 1: Prepare a small corpus of sports commentary text.
def clean_text(text: str) -> str:
    """Normalise quotes and dashes, drop stray symbols, collapse whitespace."""
    text = text.replace("’", "'").replace("‘", "'").replace("—", " ")
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(sentence: str) -> list[str]:
    """Lowercase word tokens; . , ! ? stay as their own tokens."""
    return _TOKEN.findall(sentence.lower())


def prepare_corpus(text: str) -> list[list[str]]:
    sentences = _SENTENCE_END.split(clean_text(text))
    return [tokens for tokens in map(tokenize, sentences) if tokens]


def prepare_sentences(text: str) -> list[str]:
    """Cleaned sentences with original casing (for the LLM); same order as prepare_corpus."""
    return [s for s in _SENTENCE_END.split(clean_text(text)) if tokenize(s)]


def load_sentences(style: str, directory: Path | str = CORPUS_DIR) -> list[str]:
    path = Path(directory) / f"{style}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no corpus for style {style!r} at {path}")
    return prepare_sentences(path.read_text(encoding="utf-8"))


def load_corpus(style: str, directory: Path | str = CORPUS_DIR) -> list[list[str]]:
    """Read corpus/<style>.txt and return one token list per sentence."""
    path = Path(directory) / f"{style}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no corpus for style {style!r} at {path}")
    return prepare_corpus(path.read_text(encoding="utf-8"))


def available_styles(directory: Path | str = CORPUS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(directory).glob("*.txt"))


def split_corpus(sentences: list[list[str]], held_out: float = 0.2, seed: int = 0):
    """Deterministic (train, validation) split."""
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    cut = max(1, int(len(order) * held_out))
    val = [sentences[i] for i in order[:cut]]
    train = [sentences[i] for i in order[cut:]]
    return train, val


# --- Milestone 2: Implement the n-gram model and generate text from it.
class NGramModel:
    """Word n-gram model. Probabilities use add-k smoothing; sampling backs off to a
    shorter context when the full one was never seen, so generation never gets stuck."""

    def __init__(self, order: int = 3, k: float = 0.1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order, self.k = order, k
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def _context(self, words) -> tuple:
        return tuple(words)[len(words) - (self.order - 1):] if self.order > 1 else ()

    def fit(self, sentences: list[list[str]]) -> "NGramModel":
        if not sentences:
            raise ValueError("cannot train on an empty corpus")
        for sent in sentences:
            padded = [BOS] * (self.order - 1) + sent + [EOS]
            self.vocab.update(sent)
            for i in range(self.order - 1, len(padded)):
                self.counts[self._context(padded[:i])][padded[i]] += 1
        self.vocab.update((EOS, UNK))
        return self

    def prob(self, context, word: str) -> float:
        """Add-k smoothed P(word | context); sums to 1 over vocab."""
        followers = self.counts.get(self._context(context), Counter())
        return (followers[word] + self.k) / (sum(followers.values()) + self.k * len(self.vocab))

    def perplexity(self, sentences: list[list[str]]) -> float:
        log_sum, n = 0.0, 0
        for sent in sentences:
            padded = [BOS] * (self.order - 1) + [w if w in self.vocab else UNK for w in sent] + [EOS]
            for i in range(self.order - 1, len(padded)):
                log_sum += math.log(self.prob(padded[:i], padded[i]))
                n += 1
        return math.exp(-log_sum / n)

    def _next(self, context: tuple, rng: random.Random, temperature: float) -> str:
        ctx = self._context(context)
        while ctx not in self.counts and ctx:
            ctx = ctx[1:]  # back off to a shorter context
        followers = self.counts[ctx]
        weights = [c ** (1.0 / max(temperature, 1e-3)) for c in followers.values()]
        return rng.choices(list(followers), weights)[0]

    def generate(self, prompt: str = "", max_tokens: int = 40, temperature: float = 1.0,
                 seed: int | None = None) -> str:
        rng = random.Random(seed)
        out = tokenize(prompt)
        context = tuple([BOS] * (self.order - 1) + out)
        for _ in range(max_tokens):
            word = self._next(context, rng, temperature)
            if word == EOS:
                break
            out.append(word)
            context += (word,)
        return detokenize(out)


def detokenize(tokens: list[str]) -> str:
    text = re.sub(r"\s+([.,!?])", r"\1", " ".join(tokens))
    return text[:1].upper() + text[1:]


def train_ngram(style: str, order: int = 3, directory: Path | str = CORPUS_DIR):
    """Train on the style's training split; return (model, validation sentences)."""
    train, val = split_corpus(load_corpus(style, directory))
    return NGramModel(order).fit(train), val
