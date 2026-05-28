from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from archs.parcae.configuration_parcae import ParcaeConfig
from archs.parcae.modeling_parcae import ParcaeForCausalLM


os.environ.setdefault("TEUTONIC_CHAIN_OVERRIDE", "chain.parcae.toml")

from eval.torch_runner import _load_raw_local_parcae_model


def test_raw_local_parcae_repo_loads_on_cpu(tmp_path: Path):
    repo = tmp_path / "raw-parcae"
    repo.mkdir()

    raw_cfg = {
        "name": "parcae-test-tiny",
        "hf_config": {"org": "local", "name": "parcae-test-tiny"},
        "block_size": 16,
        "n_embd": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 32,
        "padding_multiple": 8,
        "padded_vocab_size": 32,
        "rope_settings": {"use_rope": True, "rope_condense_ratio": 1, "rope_base": 50000},
        "attn_impl": "sdpa",
        "norm_eps": 1e-5,
        "bias": False,
        "qk_bias": False,
        "init_strategy": "scaled-zero",
        "init_orthogonal": True,
        "use_fused_head": "pytorch",
        "qk_norm": True,
        "injection_type": "diagonal",
        "n_layers_in_recurrent_block": 1,
        "n_layers_in_prelude": 1,
        "n_layers_in_coda": 1,
        "state_init": "like-init",
        "recurrent_embedding_dimension": 32,
        "recurrent_intermediation_embedding_dimension": 64,
        "prelude_norm": True,
        "sampling_scheme": "poisson-truncated-full",
        "mean_recurrence": 2,
        "mean_backprop_depth": 1,
        "recurrent_iteration_method": "per-sequence",
        "tie_embeddings": True,
        "model_class_name": "Parcae",
    }
    (repo / "config.json").write_text(json.dumps(raw_cfg, indent=2))

    source_model = ParcaeForCausalLM(ParcaeConfig.from_parcae_config_dict(raw_cfg))
    raw_state = {name: tensor.detach().clone() for name, tensor in source_model.model.backbone.state_dict().items()}
    save_file(raw_state, str(repo / "model.safetensors"))

    model = _load_raw_local_parcae_model(str(repo), "cpu")
    model.eval()

    sample = torch.randint(0, 32, (1, 4), dtype=torch.long)
    with torch.no_grad():
        logits = model(sample).logits

    assert logits.shape == (1, 4, 32)
