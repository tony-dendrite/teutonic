"""Hugging Face config wrapper for Parcae."""

from __future__ import annotations

from transformers import PretrainedConfig


class ParcaeConfig(PretrainedConfig):
    model_type = "parcae"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 1536,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        intermediate_size: int = 6144,
        head_dim: int | None = None,
        max_position_embeddings: int = 2048,
        tie_word_embeddings: bool = True,
        rope_theta: float = 50_000.0,
        rope_condense_ratio: int = 1,
        use_rope: bool = True,
        name: str = "parcae-xlarge-1_3b",
        hf_config: dict | None = None,
        padding_multiple: int = 64,
        padded_vocab_size: int | None = None,
        bias: bool = False,
        qk_bias: bool = False,
        norm_eps: float = 1e-5,
        block_class_name: str = "TransformerPreNormBlock",
        norm_class_name: str = "RMSNorm",
        mlp_class_name: str = "BaseMLP",
        nonlin_name: str = "ReLU2",
        init_strategy: str = "scaled-zero",
        init_orthogonal: bool = True,
        skip_initialization: bool = False,
        mup_model_scaling_factor: int = 1,
        use_fused_head: str = "pytorch",
        qk_norm: bool = True,
        logit_softcap: float | None = None,
        injection_type: str = "diagonal",
        n_layers_in_prelude: int = 8,
        n_layers_in_recurrent_block: int = 8,
        n_layers_in_coda: int = 8,
        mean_recurrence: int = 8,
        mean_backprop_depth: int = 4,
        recurrent_embedding_dimension: int = 1536,
        recurrent_intermediation_embedding_dimension: int = 6144,
        recurrent_num_attention_heads: int | None = None,
        prelude_norm: bool = True,
        state_init: str = "like-init",
        recurrent_iteration_method: str = "per-sequence",
        sampling_scheme: str = "poisson-truncated-full",
        activation_checkpoint_impl: str = "per-iteration",
        randomize_positions_from: int | None = None,
        attn_impl: str = "flash",
        clip_qkv: float | None = None,
        debias_attention: bool = False,
        center_attention: bool = False,
        model_class_name: str = "Parcae",
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_embd = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_position_embeddings
        self.block_size = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.tie_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_settings = {
            "use_rope": use_rope,
            "rope_condense_ratio": rope_condense_ratio,
            "rope_base": rope_theta,
        }
        self.name = name
        self.hf_config = hf_config or {}
        self.padding_multiple = padding_multiple
        self.padded_vocab_size = padded_vocab_size
        self.bias = bias
        self.qk_bias = qk_bias
        self.norm_eps = norm_eps
        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name
        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.skip_initialization = skip_initialization
        self.mup_model_scaling_factor = mup_model_scaling_factor
        self.use_fused_head = use_fused_head
        self.qk_norm = qk_norm
        self.logit_softcap = logit_softcap
        self.injection_type = injection_type
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.n_layers_in_coda = n_layers_in_coda
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth
        self.recurrent_embedding_dimension = recurrent_embedding_dimension
        self.recurrent_intermediation_embedding_dimension = recurrent_intermediation_embedding_dimension
        self.recurrent_num_attention_heads = recurrent_num_attention_heads
        self.prelude_norm = prelude_norm
        self.state_init = state_init
        self.recurrent_iteration_method = recurrent_iteration_method
        self.sampling_scheme = sampling_scheme
        self.activation_checkpoint_impl = activation_checkpoint_impl
        self.randomize_positions_from = randomize_positions_from
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.debias_attention = debias_attention
        self.center_attention = center_attention
        self.model_class_name = model_class_name
        self.architectures = kwargs.pop("architectures", ["ParcaeForCausalLM"])
        self.use_cache = kwargs.pop("use_cache", False)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_parcae_config_dict(cls, config_dict: dict):
        data = dict(config_dict)
        data.pop("_class_name", None)
        data.pop("init", None)
        data.pop("_recurrent_block_config", None)
        rope = data.get("rope_settings") or {}
        num_hidden_layers = (
            data.get("n_layers_in_prelude", 0)
            + data.get("n_layers_in_recurrent_block", 0)
            + data.get("n_layers_in_coda", 0)
        )
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["n_embd"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data.get("num_key_value_heads", data["num_attention_heads"]),
            intermediate_size=data["intermediate_size"],
            max_position_embeddings=data["block_size"],
            tie_word_embeddings=data.get("tie_embeddings", data.get("tie_word_embeddings", False)),
            rope_theta=rope.get("rope_base", data.get("rope_theta", 50_000.0)),
            rope_condense_ratio=rope.get("rope_condense_ratio", 1),
            use_rope=rope.get("use_rope", True),
            name=data.get("name", "parcae"),
            hf_config=data.get("hf_config", {}),
            padding_multiple=data.get("padding_multiple", 64),
            padded_vocab_size=data.get("padded_vocab_size"),
            bias=data.get("bias", False),
            qk_bias=data.get("qk_bias", False),
            norm_eps=data.get("norm_eps", 1e-5),
            block_class_name=data.get("block_class_name", "TransformerPreNormBlock"),
            norm_class_name=data.get("norm_class_name", "RMSNorm"),
            mlp_class_name=data.get("mlp_class_name", "BaseMLP"),
            nonlin_name=data.get("nonlin_name", "ReLU2"),
            init_strategy=data.get("init_strategy", "scaled-zero"),
            init_orthogonal=data.get("init_orthogonal", True),
            skip_initialization=data.get("skip_initialization", False),
            mup_model_scaling_factor=data.get("mup_model_scaling_factor", 1),
            use_fused_head=data.get("use_fused_head", "pytorch"),
            qk_norm=data.get("qk_norm", True),
            logit_softcap=data.get("logit_softcap"),
            injection_type=data.get("injection_type", "diagonal"),
            n_layers_in_prelude=data.get("n_layers_in_prelude", 0),
            n_layers_in_recurrent_block=data.get("n_layers_in_recurrent_block", 0),
            n_layers_in_coda=data.get("n_layers_in_coda", 0),
            mean_recurrence=data.get("mean_recurrence", 1),
            mean_backprop_depth=data.get("mean_backprop_depth", 1),
            recurrent_embedding_dimension=data.get("recurrent_embedding_dimension", data["n_embd"]),
            recurrent_intermediation_embedding_dimension=data.get(
                "recurrent_intermediation_embedding_dimension", data["intermediate_size"]
            ),
            recurrent_num_attention_heads=data.get("recurrent_num_attention_heads"),
            prelude_norm=data.get("prelude_norm", False),
            state_init=data.get("state_init", "like-init"),
            recurrent_iteration_method=data.get("recurrent_iteration_method", "per-sequence"),
            sampling_scheme=data.get("sampling_scheme", "poisson-truncated-full"),
            activation_checkpoint_impl=data.get("activation_checkpoint_impl", "per-iteration"),
            randomize_positions_from=data.get("randomize_positions_from"),
            attn_impl=data.get("attn_impl", "flash"),
            clip_qkv=data.get("clip_qkv"),
            debias_attention=data.get("debias_attention", False),
            center_attention=data.get("center_attention", False),
            model_class_name=data.get("model_class_name", "Parcae"),
            pad_token_id=data.get("pad_token_id"),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
        )

    def to_runtime_config(self):
        from .vendor.runtime_config import ParcaeRuntimeConfig

        return ParcaeRuntimeConfig(
            name=self.name,
            hf_config=self.hf_config,
            block_size=self.block_size,
            n_embd=self.n_embd,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            padding_multiple=self.padding_multiple,
            padded_vocab_size=self.padded_vocab_size,
            rope_settings=self.rope_settings,
            randomize_positions_from=self.randomize_positions_from,
            block_class_name=self.block_class_name,
            norm_class_name=self.norm_class_name,
            attn_impl=self.attn_impl,
            norm_eps=self.norm_eps,
            mlp_class_name=self.mlp_class_name,
            nonlin_name=self.nonlin_name,
            bias=self.bias,
            qk_bias=self.qk_bias,
            init_strategy=self.init_strategy,
            init_orthogonal=self.init_orthogonal,
            skip_initialization=self.skip_initialization,
            mup_model_scaling_factor=self.mup_model_scaling_factor,
            use_fused_head=self.use_fused_head,
            debias_attention=self.debias_attention,
            center_attention=self.center_attention,
            clip_qkv=self.clip_qkv,
            qk_norm=self.qk_norm,
            logit_softcap=self.logit_softcap,
            activation_checkpoint_impl=self.activation_checkpoint_impl,
            injection_type=self.injection_type,
            n_layers_in_recurrent_block=self.n_layers_in_recurrent_block,
            n_layers_in_prelude=self.n_layers_in_prelude,
            n_layers_in_coda=self.n_layers_in_coda,
            state_init=self.state_init,
            recurrent_embedding_dimension=self.recurrent_embedding_dimension,
            recurrent_intermediation_embedding_dimension=self.recurrent_intermediation_embedding_dimension,
            recurrent_num_attention_heads=self.recurrent_num_attention_heads,
            prelude_norm=self.prelude_norm,
            sampling_scheme=self.sampling_scheme,
            mean_recurrence=self.mean_recurrence,
            mean_backprop_depth=self.mean_backprop_depth,
            recurrent_iteration_method=self.recurrent_iteration_method,
            tie_embeddings=self.tie_embeddings,
            model_class_name=self.model_class_name,
        )


__all__ = ["ParcaeConfig"]
