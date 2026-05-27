from __future__ import annotations

from archs.parcae.configuration_parcae import ParcaeConfig


PARCAE_370M_RAW = {
    "name": "parcae-medium-370m",
    "hf_config": {"org": "SandyResearch", "name": "parcae-medium-370m"},
    "block_size": 2048,
    "n_embd": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "vocab_size": 32768,
    "padding_multiple": 64,
    "padded_vocab_size": 32768,
    "rope_settings": {"use_rope": True, "rope_condense_ratio": 1, "rope_base": 50000},
    "attn_impl": "flash",
    "norm_eps": 1e-5,
    "bias": False,
    "qk_bias": False,
    "init_strategy": "scaled-zero",
    "init_orthogonal": True,
    "use_fused_head": "pytorch",
    "qk_norm": True,
    "injection_type": "diagonal",
    "n_layers_in_recurrent_block": 4,
    "n_layers_in_prelude": 4,
    "n_layers_in_coda": 4,
    "state_init": "like-init",
    "recurrent_embedding_dimension": 1024,
    "recurrent_intermediation_embedding_dimension": 4096,
    "prelude_norm": True,
    "sampling_scheme": "poisson-truncated-full",
    "mean_recurrence": 8,
    "mean_backprop_depth": 4,
    "recurrent_iteration_method": "per-sequence",
    "tie_embeddings": True,
    "model_class_name": "Parcae",
}


PARCAE_1_3B_RAW = {
    "name": "parcae-xlarge-1_3b",
    "hf_config": {"org": "SandyResearch", "name": "parcae-xlarge-1_3b"},
    "block_size": 2048,
    "n_embd": 1536,
    "intermediate_size": 6144,
    "num_attention_heads": 12,
    "num_key_value_heads": 12,
    "vocab_size": 32768,
    "padding_multiple": 64,
    "padded_vocab_size": 32768,
    "rope_settings": {"use_rope": True, "rope_condense_ratio": 1, "rope_base": 50000},
    "attn_impl": "flash",
    "norm_eps": 1e-5,
    "bias": False,
    "qk_bias": False,
    "init_strategy": "scaled-zero",
    "init_orthogonal": True,
    "use_fused_head": "pytorch",
    "qk_norm": True,
    "injection_type": "diagonal",
    "n_layers_in_recurrent_block": 8,
    "n_layers_in_prelude": 8,
    "n_layers_in_coda": 8,
    "state_init": "like-init",
    "recurrent_embedding_dimension": 1536,
    "recurrent_intermediation_embedding_dimension": 6144,
    "prelude_norm": True,
    "sampling_scheme": "poisson-truncated-full",
    "mean_recurrence": 8,
    "mean_backprop_depth": 4,
    "recurrent_iteration_method": "per-sequence",
    "tie_embeddings": True,
    "model_class_name": "Parcae",
}


def test_parcae_config_supports_370m_shape():
    cfg = ParcaeConfig.from_parcae_config_dict(PARCAE_370M_RAW)

    assert cfg.name == "parcae-medium-370m"
    assert cfg.hidden_size == 1024
    assert cfg.intermediate_size == 4096
    assert cfg.num_attention_heads == 8
    assert cfg.num_hidden_layers == 12
    assert cfg.n_layers_in_prelude == 4
    assert cfg.n_layers_in_recurrent_block == 4
    assert cfg.n_layers_in_coda == 4
    assert cfg.recurrent_embedding_dimension == 1024
    assert cfg.recurrent_intermediation_embedding_dimension == 4096
    assert cfg.padded_vocab_size == 32768
    assert cfg.tie_embeddings is True


def test_parcae_config_keeps_1_3b_shape():
    cfg = ParcaeConfig.from_parcae_config_dict(PARCAE_1_3B_RAW)

    assert cfg.name == "parcae-xlarge-1_3b"
    assert cfg.hidden_size == 1536
    assert cfg.intermediate_size == 6144
    assert cfg.num_attention_heads == 12
    assert cfg.num_hidden_layers == 24
    assert cfg.n_layers_in_prelude == 8
    assert cfg.n_layers_in_recurrent_block == 8
    assert cfg.n_layers_in_coda == 8
    assert cfg.recurrent_embedding_dimension == 1536
    assert cfg.recurrent_intermediation_embedding_dimension == 6144
    assert cfg.padded_vocab_size == 32768
    assert cfg.tie_embeddings is True
