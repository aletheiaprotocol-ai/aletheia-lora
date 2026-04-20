"""
Tests for the aletheia-lora package.

These tests validate the core API (select_layers, aletheia_lora_config,
apply_aletheia_lora) without requiring GPU or large model downloads.
"""

import pytest
from aletheia_lora.selection import select_layers
from aletheia_lora.config import aletheia_lora_config, apply_aletheia_lora


# ── select_layers ────────────────────────────────────────

class TestSelectLayers:
    """Tests for the layer selection function."""

    def test_top50_even(self):
        scores = {0: 1.0, 1: 4.0, 2: 2.0, 3: 3.0}
        result = select_layers(scores, top_pct=50)
        assert result == [1, 3]

    def test_top50_odd(self):
        scores = {0: 5.0, 1: 1.0, 2: 3.0, 3: 2.0, 4: 4.0}
        result = select_layers(scores, top_pct=50)
        # 50% of 5 = 2.5 → int → 2 layers: indices 0 (5.0) and 4 (4.0)
        assert result == [0, 4]

    def test_top100_returns_all(self):
        scores = {i: float(i) for i in range(10)}
        result = select_layers(scores, top_pct=100)
        assert result == list(range(10))

    def test_top1_returns_at_least_one(self):
        scores = {0: 0.1, 1: 0.9, 2: 0.5}
        result = select_layers(scores, top_pct=1)
        assert result == [1]

    def test_returns_sorted(self):
        scores = {5: 10.0, 0: 9.0, 3: 8.0, 1: 7.0}
        result = select_layers(scores, top_pct=50)
        assert result == sorted(result)

    def test_single_layer(self):
        scores = {0: 42.0}
        result = select_layers(scores, top_pct=50)
        assert result == [0]

    def test_32_layers_top50(self):
        """Simulate a 32-layer model — should get exactly 16 layers."""
        scores = {i: float(i) for i in range(32)}
        result = select_layers(scores, top_pct=50)
        assert len(result) == 16
        # Should pick the top-scoring layers (indices 16–31)
        assert result == list(range(16, 32))

    def test_identical_scores(self):
        """When all scores are equal, should still return top_pct% layers."""
        scores = {i: 1.0 for i in range(10)}
        result = select_layers(scores, top_pct=50)
        assert len(result) == 5

    def test_zero_scores(self):
        """Layers with zero gradient should be deprioritized."""
        scores = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
        result = select_layers(scores, top_pct=25)
        assert 2 in result


# ── aletheia_lora_config ─────────────────────────────────

class TestAletheiaLoraConfig:
    """Tests for LoRA config generation."""

    def test_basic_config(self):
        cfg = aletheia_lora_config([0, 2, 4])
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.layers_to_transform == [0, 2, 4]
        assert cfg.task_type == "CAUSAL_LM"
        assert cfg.bias == "none"

    def test_custom_rank(self):
        cfg = aletheia_lora_config([1, 3], r=8, lora_alpha=16)
        assert cfg.r == 8
        assert cfg.lora_alpha == 16

    def test_asymmetric_rank_pattern(self):
        cfg = aletheia_lora_config([1, 3], attention_r=16, mlp_r=64)
        assert cfg.r == 16
        assert cfg.rank_pattern["q_proj"] == 16
        assert cfg.rank_pattern["k_proj"] == 16
        assert cfg.rank_pattern["v_proj"] == 16
        assert cfg.rank_pattern["o_proj"] == 16
        assert cfg.rank_pattern["gate_proj"] == 64
        assert cfg.rank_pattern["up_proj"] == 64
        assert cfg.rank_pattern["down_proj"] == 64

    def test_asymmetric_alpha_pattern(self):
        cfg = aletheia_lora_config([1, 3], attention_alpha=32, mlp_alpha=128)
        assert cfg.alpha_pattern["q_proj"] == 32
        assert cfg.alpha_pattern["o_proj"] == 32
        assert cfg.alpha_pattern["gate_proj"] == 128
        assert cfg.alpha_pattern["down_proj"] == 128

    def test_custom_target_modules(self):
        modules = ["q_proj", "v_proj"]
        cfg = aletheia_lora_config([0], target_modules=modules)
        assert set(cfg.target_modules) == set(modules)

    def test_default_target_modules(self):
        cfg = aletheia_lora_config([0])
        expected = {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}
        assert set(cfg.target_modules) == expected

    def test_dropout(self):
        cfg = aletheia_lora_config([0, 1], lora_dropout=0.1)
        assert cfg.lora_dropout == 0.1

    def test_layers_to_transform_matches_input(self):
        layers = [3, 7, 11, 15]
        cfg = aletheia_lora_config(layers)
        assert cfg.layers_to_transform == layers


# ── apply_aletheia_lora ──────────────────────────────────

class TestApplyAletheiaLora:
    """Tests for the convenience wrapper."""

    def test_returns_same_as_config_builder(self):
        layers = [0, 5, 10]
        cfg1 = aletheia_lora_config(layers, r=8)
        cfg2 = apply_aletheia_lora(None, layers, r=8)
        assert cfg1.r == cfg2.r
        assert cfg1.layers_to_transform == cfg2.layers_to_transform
        assert cfg1.lora_alpha == cfg2.lora_alpha

    def test_forwards_asymmetric_args(self):
        cfg = apply_aletheia_lora(None, [0, 5], attention_r=8, mlp_r=32)
        assert cfg.rank_pattern["q_proj"] == 8
        assert cfg.rank_pattern["up_proj"] == 32


# ── gradient_probe (import-only, no GPU) ─────────────────

class TestProbeImport:
    """Verify the probe module is importable without GPU."""

    def test_import(self):
        from aletheia_lora.probe import gradient_probe
        assert callable(gradient_probe)

    def test_extract_layer_idx(self):
        from aletheia_lora.probe import _extract_layer_idx
        assert _extract_layer_idx("model.layers.5.self_attn.q_proj.weight", 32) == 5
        assert _extract_layer_idx("transformer.h.12.mlp.weight", 24) == 12
        assert _extract_layer_idx("model.embed_tokens.weight", 32) is None
        assert _extract_layer_idx("model.layers.99.weight", 32) is None


# ── __init__ exports ─────────────────────────────────────

class TestExports:
    """Verify package exports."""

    def test_version(self):
        import aletheia_lora
        assert aletheia_lora.__version__ == "0.1.1"

    def test_all_exports(self):
        import aletheia_lora
        for name in ["gradient_probe", "select_layers",
                      "apply_aletheia_lora", "aletheia_lora_config"]:
            assert hasattr(aletheia_lora, name)
