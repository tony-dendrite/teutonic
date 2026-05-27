"""Vendored Parcae architecture for Teutonic."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_parcae import ParcaeConfig
from .modeling_parcae import (
    ParcaeCausalLMOutputWithPast,
    ParcaeForCausalLM,
    ParcaeModel,
    ParcaeModelOutputWithPast,
    ParcaePreTrainedModel,
)


def _register():
    try:
        AutoConfig.register("parcae", ParcaeConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(ParcaeConfig, ParcaeModel)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(ParcaeConfig, ParcaeForCausalLM)
    except ValueError:
        pass


_register()


__all__ = [
    "ParcaeConfig",
    "ParcaePreTrainedModel",
    "ParcaeModel",
    "ParcaeForCausalLM",
    "ParcaeModelOutputWithPast",
    "ParcaeCausalLMOutputWithPast",
]
