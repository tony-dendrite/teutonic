"""Seed a Qwen3 dense checkpoint as the genesis king.

Downloads the chosen Qwen3 model from HuggingFace, strips any `auto_map` from
`config.json` and any `.py` files (validator rejects these), pushes the result
to Hippius, prints the OCI digest to paste into `chain.toml::[seed].seed_digest`.

Usage:
    HIPPIUS_HUB_TOKEN=... HF_TOKEN=... \
        python -m archs.qwen3.seed --hf Qwen/Qwen3-4B --hippius <namespace>/<chain.name>-genesis

The Hippius repo name should match `chain.toml::[chain].repo_pattern` so the
validator accepts it. By default, `chain_config.SEED_REPO` is what the
validator expects as the genesis pointer.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Ensure repo-root is on the path when running as `python -m archs.qwen3.seed`.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_store import upload_model_folder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed")


def _scrub(local_dir: str) -> None:
    """Remove anything that would trip the validator's submission gates:
    `auto_map` in config.json, any `*.py` modeling code.
    """
    cfg_path = Path(local_dir) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "auto_map" in cfg:
            log.warning("stripping auto_map from config.json")
            del cfg["auto_map"]
            cfg_path.write_text(json.dumps(cfg, indent=2))
    for py in Path(local_dir).rglob("*.py"):
        log.warning("removing modeling file: %s", py.name)
        py.unlink()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True, help="HuggingFace repo id, e.g. Qwen/Qwen3-4B")
    p.add_argument("--hippius", required=True, help="Target Hippius repo, e.g. unconst/<chain.name>-genesis")
    p.add_argument("--revision", default=None, help="HF revision (default: main / latest)")
    p.add_argument("--workdir", default="/tmp/teutonic/seed",
                   help="Scratch dir for download + scrub")
    args = p.parse_args()

    work = Path(args.workdir) / args.hf.replace("/", "_")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    log.info("downloading %s @ %s -> %s", args.hf, args.revision or "main", work)
    snapshot_download(
        repo_id=args.hf,
        revision=args.revision,
        local_dir=str(work),
        allow_patterns=["*.safetensors", "config.json", "model.safetensors.index.json",
                        "tokenizer*", "*.json", "*.model", "*.txt"],
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )

    _scrub(str(work))

    sizes = sum(p.stat().st_size for p in work.rglob("*.safetensors"))
    log.info("safetensors bytes: %.2f GiB", sizes / (1 << 30))

    log.info("uploading -> %s", args.hippius)
    ref = upload_model_folder(str(work), args.hippius, commit_message=f"genesis from {args.hf}")
    log.info("UPLOADED: %s", ref.immutable_ref)
    print()
    print("=" * 60)
    print(f"chain.toml [seed].seed_digest = \"{ref.digest}\"")
    print(f"chain.toml [chain].seed_repo  = \"{ref.repo}\"")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
