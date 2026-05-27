"""Convert a published Parcae checkpoint into a Teutonic-safe seed repo."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chain_config  # noqa: E402

from .configuration_parcae import ParcaeConfig
from .modeling_parcae import ParcaeForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("parcae-seed")

DEFAULT_SOURCE_REPO = "SandyResearch/parcae-1.3b"
DEFAULT_TOKENIZER_REPO = "SandyResearch/parcae-tokenizer"
DEFAULT_TARGET_REPO = "teutonic/teutonic-parcae-1_3b-genesis"
DEFAULT_REPO_BACKEND = "hippius"

if chain_config.ARCH_MODULE == "archs.parcae":
    DEFAULT_TARGET_REPO = chain_config.SEED_REPO
    DEFAULT_TOKENIZER_REPO = chain_config.SEED_TOKENIZER_REPO or DEFAULT_TOKENIZER_REPO
    DEFAULT_REPO_BACKEND = getattr(chain_config, "SEED_REPO_BACKEND", DEFAULT_REPO_BACKEND)


def _strip_auto_map(out_dir: Path):
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.pop("auto_map", None) is not None:
        cfg_path.write_text(json.dumps(cfg, indent=2))


def _copy_tokenizer_files(tokenizer_repo: str, out_dir: Path) -> dict[str, object]:
    tok_dir = Path(out_dir.parent / f"{out_dir.name}-tokenizer")
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    snapshot_download(
        tokenizer_repo,
        local_dir=str(tok_dir),
        allow_patterns=["tokenizer*", "special_tokens*", "token_bytes.pt"],
        token=os.environ.get("HF_TOKEN"),
    )
    for path in tok_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, out_dir / path.name)
    meta = _infer_token_metadata(out_dir / "tokenizer.json", out_dir / "tokenizer_config.json")
    _write_tokenizer_sidecars(out_dir, meta)
    return meta


def _infer_token_metadata(tokenizer_json: Path, tokenizer_config_json: Path) -> dict[str, object]:
    token_data = json.loads(tokenizer_json.read_text())
    added = {item["content"]: item["id"] for item in token_data.get("added_tokens", [])}
    tok_cfg = json.loads(tokenizer_config_json.read_text()) if tokenizer_config_json.exists() else {}

    def unwrap(value):
        if isinstance(value, dict):
            return value.get("content")
        return value

    def resolve(*candidates):
        for candidate in candidates:
            token = unwrap(candidate)
            if token in added:
                return token, added[token]
        return None, None

    bos_token, bos_token_id = resolve(tok_cfg.get("bos_token"), "<|bos|>", "<bos>", "<s>")
    eos_token, eos_token_id = resolve(tok_cfg.get("eos_token"), "<|eos|>", "</s>", "<|endoftext|>")
    pad_token, pad_token_id = resolve(tok_cfg.get("pad_token"), "<|pad|>", "<pad>", "<|eos|>")
    return {
        "bos_token": bos_token,
        "bos_token_id": bos_token_id,
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "pad_token": pad_token,
        "pad_token_id": pad_token_id,
    }


def _write_tokenizer_sidecars(out_dir: Path, meta: dict[str, object]):
    special_map_path = out_dir / "special_tokens_map.json"
    tokenizer_cfg_path = out_dir / "tokenizer_config.json"

    special_map = {
        key: meta[key]
        for key in ("bos_token", "eos_token", "pad_token")
        if meta.get(key) is not None
    }
    if not special_map_path.exists():
        special_map_path.write_text(json.dumps(special_map, indent=2))

    tokenizer_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 2048,
        **special_map,
    }
    if not tokenizer_cfg_path.exists():
        tokenizer_cfg_path.write_text(json.dumps(tokenizer_cfg, indent=2))


def _finalize_config_json(out_dir: Path, config: ParcaeConfig):
    cfg_path = out_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    data["tie_word_embeddings"] = bool(config.tie_embeddings)
    data["architectures"] = ["ParcaeForCausalLM"]
    cfg_path.write_text(json.dumps(data, indent=2))


def _verify_local_export(out_dir: Path):
    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        use_safetensors=True,
        trust_remote_code=False,
    )
    sample = torch.randint(0, int(cfg.vocab_size), (1, 8), dtype=torch.long)
    with torch.no_grad():
        logits = model(sample).logits
    log.info("verify load ok: logits shape=%s", tuple(logits.shape))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=os.environ.get("TEUTONIC_SEED_SOURCE_REPO", DEFAULT_SOURCE_REPO))
    parser.add_argument("--tokenizer-repo", default=os.environ.get("TEUTONIC_SEED_TOKENIZER_OVERRIDE", DEFAULT_TOKENIZER_REPO))
    parser.add_argument("--target-repo", default=os.environ.get("TEUTONIC_SEED_REPO_OVERRIDE", DEFAULT_TARGET_REPO))
    parser.add_argument("--repo-backend", choices=("hf", "hippius"), default=os.environ.get("TEUTONIC_SEED_REPO_BACKEND_OVERRIDE", DEFAULT_REPO_BACKEND))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--workdir", default="/tmp/teutonic/parcae-seed")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--no-verify-load", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.workdir) / "export"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    log.info("loading raw config from %s", args.source_repo)
    config_path = hf_hub_download(args.source_repo, "config.json", revision=args.revision, token=os.environ.get("HF_TOKEN"))
    source_config = json.loads(Path(config_path).read_text())
    hf_config = ParcaeConfig.from_parcae_config_dict(source_config)

    log.info("copying tokenizer from %s", args.tokenizer_repo)
    token_meta = _copy_tokenizer_files(args.tokenizer_repo, out_dir)
    for key, value in token_meta.items():
        setattr(hf_config, key, value)

    log.info("building vendored HF wrapper")
    model = ParcaeForCausalLM(hf_config)

    log.info("loading raw weights from %s", args.source_repo)
    weights_path = hf_hub_download(args.source_repo, "pytorch_model.bin", revision=args.revision, token=os.environ.get("HF_TOKEN"))
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.model.backbone.load_state_dict(raw_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"raw checkpoint mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")

    log.info("saving converted seed repo to %s", out_dir)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="10GB")
    _strip_auto_map(out_dir)
    _finalize_config_json(out_dir, hf_config)

    if not args.no_verify_load:
        _verify_local_export(out_dir)

    if args.push and args.repo_backend == "hippius":
        from model_store import upload_model_folder

        ref = upload_model_folder(out_dir, args.target_repo, commit_message=f"genesis from {args.source_repo}")
        seed_repo = ref.repo
        seed_digest = ref.digest
        log.info("uploaded to %s", ref.immutable_ref)
    elif args.push:
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        api.create_repo(args.target_repo, exist_ok=True, private=not args.public, repo_type="model")
        api.upload_folder(folder_path=str(out_dir), repo_id=args.target_repo, commit_message=f"genesis from {args.source_repo}")
        info = api.model_info(args.target_repo)
        seed_repo = args.target_repo
        seed_digest = f"hf:{info.sha}"
        log.info("uploaded to %s@%s", seed_repo, seed_digest)
    else:
        seed_repo = str(out_dir)
        seed_digest = ""

    print()
    print("=" * 60)
    print(f"seed export dir               = \"{out_dir}\"")
    print(f"chain.toml [chain].seed_repo  = \"{seed_repo}\"")
    if seed_digest:
        print(f"chain.toml [seed].seed_digest = \"{seed_digest}\"")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
