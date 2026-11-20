"""Tests for milestones 1-2 (corpus and n-gram). LoRA tests come with milestones 3-5."""
import pytest

from app.core import (NGramModel, available_styles, clean_text, load_corpus,
                      prepare_corpus, split_corpus, tokenize, train_ngram)


# Milestone 1: corpus preparation
def test_milestone_1_cleaning_and_tokenizing():
    assert clean_text("Four!!  — what a   shot #1") == "Four!! what a shot 1"
    assert tokenize("He's out, LBW!") == ["he's", "out", ",", "lbw", "!"]


def test_milestone_1_corpus_files_load_per_style():
    assert {"cricket", "football"} <= set(available_styles())
    for style in ("cricket", "football"):
        sents = load_corpus(style)
        assert len(sents) >= 20 and all(sents)
    with pytest.raises(FileNotFoundError):
        load_corpus("curling")


def test_milestone_1_split_is_deterministic():
    sents = prepare_corpus("One a. Two b. Three c. Four d. Five e.")
    a, b = split_corpus(sents, 0.4, seed=1), split_corpus(sents, 0.4, seed=1)
    assert a == b and len(a[1]) == 2 and len(a[0]) == 3


# Milestone 2: n-gram model
def test_milestone_2_probabilities_sum_to_one():
    m = NGramModel(2).fit(prepare_corpus("a b. a c. a b."))
    assert m.prob(("a",), "b") > m.prob(("a",), "c") > 0
    assert sum(m.prob(("a",), w) for w in m.vocab) == pytest.approx(1.0)


def test_milestone_2_generation_is_seeded_and_uses_the_prompt():
    model, _ = train_ngram("cricket", order=3)
    a = model.generate("The bowler", max_tokens=15, seed=7)
    assert a == model.generate("The bowler", max_tokens=15, seed=7)
    assert a.startswith("The bowler") and len(a.split()) > 2


def test_milestone_2_perplexity_is_finite_and_style_sensitive():
    cricket, val = train_ngram("cricket")
    football, _ = train_ngram("football")
    assert 1 < cricket.perplexity(val) < float("inf")
    assert cricket.perplexity(val) < football.perplexity(val)  # in-style text is likelier


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        NGramModel().fit([])
