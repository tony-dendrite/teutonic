from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import pytest

from archs.parcae.configuration_parcae import ParcaeConfig


if "bittensor" not in sys.modules:
    bittensor_stub = types.ModuleType("bittensor")
    bittensor_stub.Wallet = object
    bittensor_stub.Subtensor = object
    sys.modules["bittensor"] = bittensor_stub

os.environ.setdefault("TEUTONIC_CHAIN_OVERRIDE", "chain.parcae.toml")

import validator


PARCAE_EXTRA_LOCK_KEYS = (
    "injection_type",
    "n_layers_in_prelude",
    "n_layers_in_recurrent_block",
    "n_layers_in_coda",
    "mean_recurrence",
    "mean_backprop_depth",
    "recurrent_embedding_dimension",
    "recurrent_intermediation_embedding_dimension",
    "prelude_norm",
    "qk_norm",
    "state_init",
    "recurrent_iteration_method",
    "sampling_scheme",
)

KING_REPO = "teutonic/teutonic-parcae-1_3b-genesis"
KING_DIGEST = "hf:" + ("1" * 40)
CHALLENGER_REPO = "alice/Teutonic-Parcae-1.3B-test"
CHALLENGER_DIGEST = "hf:" + ("2" * 40)


def _write_repo(
    root: Path,
    *,
    config_overrides: dict[str, object] | None = None,
    include_safetensors: bool = True,
    extra_files: dict[str, str] | None = None,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    config = ParcaeConfig(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ).to_dict()
    config["architectures"] = ["ParcaeForCausalLM"]
    config.update(config_overrides or {})
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    if include_safetensors:
        (root / "model.safetensors").write_bytes(b"stub-safetensors")
    for rel_path, text in (extra_files or {}).items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return config


@pytest.fixture()
def validator_harness(tmp_path, monkeypatch):
    king_dir = tmp_path / "king"
    challenger_dir = tmp_path / "challenger"
    king_cfg = _write_repo(king_dir)
    _write_repo(challenger_dir)

    repo_map = {
        KING_REPO: king_dir,
        CHALLENGER_REPO: challenger_dir,
    }

    def fake_get_king_config(_repo: str, _digest: str = ""):
        return king_cfg

    def fake_materialize_model(ref, local_dir=None, max_workers=None, *, config_only=False):
        return str(repo_map[ref.repo])

    def fake_list_remote_files(ref):
        base = repo_map[ref.repo]
        return sorted(
            str(path.relative_to(base)).replace("\\", "/")
            for path in base.rglob("*")
            if path.is_file()
        )

    def fake_snapshot_size(snapshot, files=None):
        base = Path(snapshot)
        wanted = files or [
            str(path.relative_to(base)).replace("\\", "/")
            for path in base.rglob("*")
            if path.is_file()
        ]
        return sum((base / rel_path).stat().st_size for rel_path in wanted if (base / rel_path).exists())

    monkeypatch.setattr(validator, "get_king_config", fake_get_king_config)
    monkeypatch.setattr(validator, "materialize_model", fake_materialize_model)
    monkeypatch.setattr(validator, "list_remote_files", fake_list_remote_files)
    monkeypatch.setattr(validator, "snapshot_size", fake_snapshot_size)
    monkeypatch.setattr(validator.chain_config, "EXTRA_LOCK_KEYS", PARCAE_EXTRA_LOCK_KEYS)
    return challenger_dir


def _validate() -> str | None:
    return validator.validate_challenger_config(
        CHALLENGER_REPO,
        CHALLENGER_DIGEST,
        king_repo=KING_REPO,
        king_digest=KING_DIGEST,
    )


def test_valid_parcae_challenger_passes(validator_harness):
    assert _validate() is None


def test_recurrence_mismatch_fails(validator_harness):
    cfg_path = validator_harness / "config.json"
    config = json.loads(cfg_path.read_text())
    config["mean_recurrence"] = int(config["mean_recurrence"]) + 1
    cfg_path.write_text(json.dumps(config, indent=2))

    rejection = _validate()

    assert rejection is not None
    assert rejection.startswith("mean_recurrence mismatch:")


def test_auto_map_fails(validator_harness):
    cfg_path = validator_harness / "config.json"
    config = json.loads(cfg_path.read_text())
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_parcae.ParcaeForCausalLM"}
    cfg_path.write_text(json.dumps(config, indent=2))

    rejection = _validate()

    assert rejection == "auto_map present in config.json (custom modeling code is not allowed)"


def test_python_file_fails(validator_harness):
    (validator_harness / "modeling_parcae.py").write_text("print('nope')\n")

    rejection = _validate()

    assert rejection is not None
    assert rejection.startswith("repo ships *.py files")


def test_missing_safetensors_fails(validator_harness):
    (validator_harness / "model.safetensors").unlink()

    rejection = _validate()

    assert rejection == "no .safetensors files in repo"
