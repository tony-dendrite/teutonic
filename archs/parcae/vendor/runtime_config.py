from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Literal, Optional, Type

import torch
from torch.utils.checkpoint import checkpoint

from .modules.basic import Linear, Relu2
from .parcae_init import ParcaeInit
from .utils import find_multiple


@dataclass
class RoPESettings:
    use_rope: bool = True
    rope_condense_ratio: int = 1
    rope_base: int = 50_000


@dataclass
class ParcaeRuntimeConfig:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    block_size: int = 2048
    n_embd: int = 1536
    intermediate_size: Optional[int] = None
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 32768
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    rope_settings: RoPESettings | dict = field(default_factory=RoPESettings)
    use_abacus: bool = False
    randomize_positions_from: Optional[int] = None
    block_class_name: str = "TransformerPreNormBlock"
    norm_class_name: str = "RMSNorm"
    attn_impl: Literal["flash", "sdpa", "debug-skip"] = "flash"
    norm_eps: float = 1e-5
    mlp_class_name: str = "BaseMLP"
    nonlin_name: str = "ReLU2"
    bias: bool = False
    qk_bias: bool = False
    init_strategy: str = "scaled-zero"
    init_orthogonal: bool = True
    skip_initialization: bool = False
    mup_model_scaling_factor: int = 1
    use_fused_head: Literal["hhe", "cce", "full-triton", "pytorch"] = "pytorch"
    debias_attention: bool = False
    center_attention: bool = False
    clip_qkv: Optional[float] = None
    qk_norm: bool = True
    logit_softcap: Optional[float] = None
    activation_checkpoint_impl: str = "per-iteration"
    simple_ops: bool = False
    strategy: str = "single"
    injection_type: Literal["diagonal", "linear", "add"] = "diagonal"
    n_layers_in_recurrent_block: int = 8
    n_layers_in_prelude: int = 8
    n_layers_in_coda: int = 8
    state_init: str = "like-init"
    recurrent_embedding_dimension: int = 1536
    recurrent_intermediation_embedding_dimension: int = 6144
    recurrent_num_attention_heads: Optional[int] = None
    prelude_norm: bool = True
    sampling_scheme: str = "poisson-truncated-full"
    mean_recurrence: int = 8
    mean_backprop_depth: int = 4
    lockstep_n: bool = False
    lockstep_k: bool = False
    curriculum_target: Literal["forward", "backward", "both"] = "forward"
    recurrent_iteration_method: Literal["per-batch", "per-sequence", "per-token"] = "per-sequence"
    tie_embeddings: bool = True
    model_class_name: Literal["Parcae"] = "Parcae"
    _is_recurrent_block_config: bool = field(default=False, repr=False)

    def __post_init__(self):
        if isinstance(self.rope_settings, dict):
            self.rope_settings = RoPESettings(**self.rope_settings)

        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        assert self.n_embd % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

        self.head_size = self.n_embd // self.num_attention_heads
        self.n_head = self.num_attention_heads
        self.n_query_groups = self.num_key_value_heads
        self.n_layer = self.n_layers_in_recurrent_block * self.mean_backprop_depth

        effective_expected_depth = (
            self.n_layers_in_prelude
            + self.n_layers_in_coda
            + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.init = ParcaeInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            effective_expected_depth,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

        self._recurrent_block_config = None
        if not self._is_recurrent_block_config:
            self._recurrent_block_config = self._build_recurrent_block_config()

    def _build_recurrent_block_config(self):
        recurrent_num_heads = self.recurrent_num_attention_heads or self.num_attention_heads
        assert self.recurrent_embedding_dimension % recurrent_num_heads == 0
        recurrent_config = replace(
            self,
            n_embd=self.recurrent_embedding_dimension,
            intermediate_size=self.recurrent_intermediation_embedding_dimension,
            num_attention_heads=recurrent_num_heads,
            num_key_value_heads=recurrent_num_heads,
            _is_recurrent_block_config=True,
        )
        return recurrent_config

    @property
    def recurrent_block_config(self):
        return self._recurrent_block_config or self

    @property
    def MLP(self) -> Type[torch.nn.Module]:
        from .modules import mlp

        return getattr(mlp, self.mlp_class_name)

    @property
    def Linear(self) -> Type[torch.nn.Module]:
        return Linear

    @property
    def Block(self) -> Type[torch.nn.Module]:
        from .modules import blocks

        return getattr(blocks, self.block_class_name)

    @property
    def Nonlin(self) -> Type[torch.nn.Module]:
        if self.nonlin_name == "ReLU2":
            return Relu2
        return getattr(torch.nn, self.nonlin_name)

    @property
    def Norm(self):
        from .modules import norms

        return getattr(norms, self.norm_class_name)

    @property
    def checkpoint(self) -> Callable:
        return partial(checkpoint, use_reentrant=False, preserve_rng_state=False)
