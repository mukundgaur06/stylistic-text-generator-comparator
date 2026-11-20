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


# Milestone 3: LoRA adapter injection (tiny random GPT-2, no download)
def test_milestone_3_injection_changes_nothing_at_first_and_trains_only_the_adapter():
    import torch
    from app.lora import inject_lora, lora_layers, parameter_counts, tiny_gpt2
    model = tiny_gpt2()
    ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        before = model(ids).logits
    inject_lora(model, r=4)
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-6)          # B starts at zero
    trainable, total = parameter_counts(model)
    assert 0 < trainable < total * 0.2
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)
    assert len(lora_layers(model)) == 2 * 3                  # c_attn, attn c_proj, mlp c_proj per block


def test_milestone_3_adapter_can_be_switched_off_and_bad_input_is_rejected():
    import torch
    from app.lora import inject_lora, lora_layers, set_adapter_enabled, tiny_gpt2
    model = inject_lora(tiny_gpt2(), r=4)
    for layer in lora_layers(model):
        torch.nn.init.normal_(layer.lora_B, std=0.5)
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        on = model(ids).logits
        set_adapter_enabled(model, False)
        off = model(ids).logits
    assert not torch.allclose(on, off)
    with pytest.raises(ValueError):
        inject_lora(tiny_gpt2(), targets=("nope",))
    with pytest.raises(ValueError):
        inject_lora(tiny_gpt2(), r=10_000)
