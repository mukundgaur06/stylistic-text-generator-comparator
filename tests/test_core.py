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


# Milestone 4: fine-tuning the adapter (tiny model + fake tokenizer, no download)
class _FakeTokenizer:
    eos_token_id = pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=64):
        return {"input_ids": [1 + (ord(c) % 90) for c in text][:max_length]}


def test_milestone_4_training_updates_only_the_adapter_and_lowers_loss(tmp_path):
    import torch
    from app.lora import (inject_lora, load_adapter, lm_perplexity, save_adapter,
                          tiny_gpt2, train_adapter)
    model, tok = inject_lora(tiny_gpt2(), r=4), _FakeTokenizer()
    base_before = {n: p.clone() for n, p in model.named_parameters() if "lora_" not in n}
    data = ["what a shot", "he is out", "a lovely drive", "caught behind"] * 2
    ppl0 = lm_perplexity(model, tok, data)
    hist = train_adapter(model, tok, data, data, epochs=15, lr=1e-2)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert lm_perplexity(model, tok, data) < ppl0
    # the weights kept are the best epoch's, not necessarily the last one's
    assert lm_perplexity(model, tok, data) == pytest.approx(min(h["val_ppl"] for h in hist), rel=1e-4)
    assert all(torch.equal(p, base_before[n]) for n, p in model.named_parameters() if n in base_before)
    path = save_adapter(model, "demo", tmp_path)
    fresh = inject_lora(tiny_gpt2(), r=4)
    load_adapter(fresh, "demo", tmp_path)
    ids = torch.randint(0, 100, (1, 6))
    with torch.no_grad():
        assert torch.allclose(model.eval()(ids).logits, fresh(ids).logits, atol=1e-5)
    assert path.stat().st_size < 1_000_000


def test_milestone_4_training_requires_an_injected_adapter():
    from app.lora import tiny_gpt2, train_adapter
    with pytest.raises(ValueError):
        train_adapter(tiny_gpt2(), _FakeTokenizer(), ["a b"])


# Milestone 5: generation from the adapted model (tiny model + fake tokenizer)
class _CharTokenizer(_FakeTokenizer):
    def __call__(self, text, **kw):
        return {"input_ids": [1 + (ord(c) % 90) for c in text]}

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_milestone_5_generation_is_seeded_and_respects_length_and_prompt():
    from app.lora import generate_text, inject_lora, tiny_gpt2
    model, tok = inject_lora(tiny_gpt2(), r=4), _CharTokenizer()
    a = generate_text(model, tok, "abc", max_new_tokens=12, seed=5)
    assert a == generate_text(model, tok, "abc", max_new_tokens=12, seed=5)
    assert a != generate_text(model, tok, "abc", max_new_tokens=12, seed=6)
    assert a.startswith(tok.decode(tok("abc")["input_ids"])) and len(a) <= 3 + 12
    assert len(generate_text(model, tok, "", max_new_tokens=8, seed=1)) <= 8


def test_milestone_5_greedy_is_deterministic_and_adapter_changes_output():
    import torch
    from app.lora import generate_text, inject_lora, lora_layers, set_adapter_enabled, tiny_gpt2
    model, tok = inject_lora(tiny_gpt2(), r=4), _CharTokenizer()
    g1 = generate_text(model, tok, "abc", 10, temperature=0)
    assert g1 == generate_text(model, tok, "abc", 10, temperature=0, seed=99)
    for layer in lora_layers(model):
        torch.nn.init.normal_(layer.lora_B, std=0.5)
    assert generate_text(model, tok, "abc", 10, temperature=0) != g1
    set_adapter_enabled(model, False)
    assert generate_text(model, tok, "abc", 10, temperature=0) == g1   # off = base model again


# Milestone 6: side-by-side comparison
def test_milestone_6_columns_are_aligned_and_wrapped():
    from app.compare import render_side_by_side
    out = render_side_by_side("LEFT", ["a long long long sentence that must wrap around"], "RIGHT", ["short"], width=43)
    lines = out.splitlines()
    assert lines[0].startswith("LEFT") and lines[0].endswith("RIGHT")
    assert len({min(i for i in (line.find("|"), line.find("+")) if i >= 0) for line in lines}) == 1  # divider never moves
    assert all(len(line) <= 43 for line in lines) and any("short" in line for line in lines)


def test_milestone_6_samples_pair_up_with_the_same_seeds():
    from app.compare import build_comparison
    left, right = build_comparison(lambda p, s: f"ngram {p} {s}", lambda p, s: f"llm {p} {s}", "hi", 3, seed=10)
    assert left == ["ngram hi 10", "ngram hi 11", "ngram hi 12"]
    assert right == ["llm hi 10", "llm hi 11", "llm hi 12"]
    with pytest.raises(ValueError):
        build_comparison(lambda p, s: "", lambda p, s: "", "x", 0)


def test_milestone_6_compare_command_prints_both_models(monkeypatch, capsys):
    import app.compare
    from app.cli import main
    monkeypatch.setattr(app.compare, "run_comparison", lambda *a, **k: "N-gram | LoRA")
    main(["compare", "--style", "cricket", "--prompt", "The bowler"])
    assert "N-gram | LoRA" in capsys.readouterr().out
