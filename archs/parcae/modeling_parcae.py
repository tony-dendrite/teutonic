"""Hugging Face model wrapper for vendored Parcae inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_parcae import ParcaeConfig
from .vendor.parcae_core import Parcae as VendoredParcae


ParcaeModelOutputWithPast = BaseModelOutputWithPast
ParcaeCausalLMOutputWithPast = CausalLMOutputWithPast


class ParcaePreTrainedModel(PreTrainedModel):
    config_class = ParcaeConfig
    base_model_prefix = "model"
    _no_split_modules = ["TransformerPreNormBlock"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        return


class ParcaeModel(ParcaePreTrainedModel):
    def __init__(self, config: ParcaeConfig):
        super().__init__(config)
        self.runtime_config = config.to_runtime_config()
        self.backbone = VendoredParcae(self.runtime_config)

    def get_input_embeddings(self):
        return self.backbone.transformer.wte

    def set_input_embeddings(self, value):
        self.backbone.transformer.wte = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        num_steps_pair: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        hidden_states = self.backbone.forward_hidden(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_steps_pair=num_steps_pair,
        )
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class ParcaeForCausalLM(ParcaePreTrainedModel):
    _tied_weights_keys = [r"lm_head\.weight$"]
    all_tied_weights_keys = {}

    def __init__(self, config: ParcaeConfig):
        super().__init__(config)
        self.model = ParcaeModel(config)

    @property
    def lm_head(self):
        return self.model.backbone.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.backbone.lm_head = new_embeddings

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.model.backbone.gradient_checkpointing = enable

    def reset_state(self) -> None:
        return

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        num_steps_pair: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_steps_pair=num_steps_pair,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float() * self.model.runtime_config.init.logit_scale
        if self.config.logit_softcap is not None:
            softcap = self.config.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=None, past_key_values=None)


__all__ = [
    "ParcaeModelOutputWithPast",
    "ParcaeCausalLMOutputWithPast",
    "ParcaePreTrainedModel",
    "ParcaeModel",
    "ParcaeForCausalLM",
]
