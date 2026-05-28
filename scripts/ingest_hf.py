#!/usr/bin/env python3
"""Download parquet files from Hugging Face, tokenize into uint32 .npy shards,
upload them to Hippius, and keep durable progress state for long PM2 runs.

This is a disk-aware v4 ingest path tuned for large corpora such as
HuggingFaceFW/fineweb-edu. Work overlaps across files:
  - some workers download parquet files
  - other workers tokenize files already downloaded
  - completed shards upload immediately

The scheduler only keeps as many files in flight as free scratch space allows.
That makes it suitable for boxes anywhere from roughly 1 TB to 4 TB scratch
without manual retuning.

Usage:
    python scripts/ingest_hf.py --dataset HuggingFaceFW/fineweb-edu
    python scripts/ingest_hf.py --dataset HuggingFaceFW/fineweb-edu --workers 24

Environment variables:
    HF_TOKEN                          Optional HF token. Needed only for gated repos.
    TEUTONIC_DS_ENDPOINT              Hippius S3 endpoint. Default: https://s3.hippius.com
    TEUTONIC_DS_BUCKET                Hippius bucket. Default: teutonic-sn3
    TEUTONIC_DS_ACCESS_KEY            Hippius access key. Required unless --dry-run.
    TEUTONIC_DS_SECRET_KEY            Hippius secret key. Required unless --dry-run.
    TEUTONIC_INGEST_TOKENIZER         Tokenizer repo. Default: Qwen/Qwen2.5-0.5B
    TEUTONIC_INGEST_DEST_PREFIX       Destination prefix. Default: dataset/v4
    TEUTONIC_INGEST_PROGRESS_DIR      Durable progress dir. Default: /var/lib/teutonic/ingest-v4
    TEUTONIC_INGEST_SCRATCH_DIR       Scratch dir for parquet + shard temp files.
                                      Default: /mnt/local-ssd/teutonic-ingest-v4 if present,
                                      otherwise /var/tmp/teutonic-ingest-v4
    TEUTONIC_INGEST_TEXT_COLUMN       Preferred text column. Default: text
    TEUTONIC_INGEST_INCLUDE_PREFIXES  Comma-separated parquet path prefixes to include.
                                      Default for FineWeb-Edu: data/
    TEUTONIC_INGEST_MIN_FREE_GB       Free-space floor to keep untouched. Default: 128
    TEUTONIC_INGEST_WORKER_DISK_GB    Scratch budget reserved per active worker.
                                      Default: 12
    TEUTONIC_INGEST_MAX_INFLIGHT      Optional hard cap on concurrent in-flight files.
    TEUTONIC_INGEST_CPU_RESERVE       CPU cores to leave unused in auto mode. Default: 2
    TEUTONIC_INGEST_AUTO_MAX_WORKERS  Upper bound for auto-picked workers. Default: 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s3_transfer import safe_upload_file

log = logging.getLogger("ingest_hf")

DEFAULT_TOKENIZER_NAME = os.environ.get(
    "TEUTONIC_INGEST_TOKENIZER", "Qwen/Qwen2.5-0.5B"
)
DEFAULT_DEST_PREFIX = os.environ.get("TEUTONIC_INGEST_DEST_PREFIX", "dataset/v4")
DEFAULT_PROGRESS_DIR = Path(
    os.environ.get("TEUTONIC_INGEST_PROGRESS_DIR", "/var/lib/teutonic/ingest-v4")
)
DEFAULT_SCRATCH_DIR = Path(
    os.environ.get(
        "TEUTONIC_INGEST_SCRATCH_DIR",
        "/mnt/local-ssd/teutonic-ingest-v4"
        if Path("/mnt/local-ssd").exists()
        else "/var/tmp/teutonic-ingest-v4",
    )
)
DEFAULT_TEXT_COLUMN = os.environ.get("TEUTONIC_INGEST_TEXT_COLUMN", "text")
DEFAULT_MIN_FREE_GB = float(os.environ.get("TEUTONIC_INGEST_MIN_FREE_GB", "128"))
DEFAULT_WORKER_DISK_GB = float(os.environ.get("TEUTONIC_INGEST_WORKER_DISK_GB", "12"))
DEFAULT_MAX_INFLIGHT = int(os.environ.get("TEUTONIC_INGEST_MAX_INFLIGHT", "0"))
DEFAULT_CPU_RESERVE = int(os.environ.get("TEUTONIC_INGEST_CPU_RESERVE", "2"))
DEFAULT_AUTO_MAX_WORKERS = int(os.environ.get("TEUTONIC_INGEST_AUTO_MAX_WORKERS", "32"))

DTYPE = np.dtype("<u4")
BYTES_PER_TOKEN = DTYPE.itemsize
GIB = 1024**3

DS_ENDPOINT = os.environ.get("TEUTONIC_DS_ENDPOINT", "https://s3.hippius.com")
DS_BUCKET = os.environ.get("TEUTONIC_DS_BUCKET", "teutonic-sn3")
DS_ACCESS_KEY = os.environ.get("TEUTONIC_DS_ACCESS_KEY", "")
DS_SECRET_KEY = os.environ.get("TEUTONIC_DS_SECRET_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_shutdown = False
_shard_counter: multiprocessing.Value | None = None
_worker_tokenizer_name = DEFAULT_TOKENIZER_NAME
_worker_dest_prefix = DEFAULT_DEST_PREFIX
_worker_scratch_root = DEFAULT_SCRATCH_DIR
_worker_text_column = DEFAULT_TEXT_COLUMN
_worker_tokenizer = None
_worker_eos_id: int | None = None


def _handle_signal(signum, frame):
    del frame
    global _shutdown
    log.warning("received signal %s, will stop after current work", signum)
    _shutdown = True


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=DS_ENDPOINT,
        aws_access_key_id=DS_ACCESS_KEY,
        aws_secret_access_key=DS_SECRET_KEY,
        region_name="decentralized",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        ),
    )


def human_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or unit == "TiB":
            if unit == "B":
                return f"{n} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def default_include_prefixes_for_dataset(dataset_repo: str) -> tuple[str, ...]:
    if "fineweb-edu" in dataset_repo.lower():
        return ("data/",)
    return ()


def parse_prefixes(raw: str | None, dataset_repo: str) -> tuple[str, ...]:
    if raw:
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    env_raw = os.environ.get("TEUTONIC_INGEST_INCLUDE_PREFIXES", "")
    if env_raw:
        return tuple(p.strip() for p in env_raw.split(",") if p.strip())
    return default_include_prefixes_for_dataset(dataset_repo)


def file_config_name(file_path: str) -> str:
    parts = file_path.split("/")
    if len(parts) >= 3 and parts[0] in {"data", "sample"}:
        return parts[1]
    return parts[0]


def get_manifest(client, dest_prefix: str) -> dict | None:
    try:
        body = client.get_object(
            Bucket=DS_BUCKET,
            Key=f"{dest_prefix}/manifest.json",
        )["Body"].read()
        return json.loads(body)
    except Exception:
        return None


def put_manifest(client, dest_prefix: str, manifest: dict):
    body = json.dumps(manifest, indent=2).encode()
    client.put_object(
        Bucket=DS_BUCKET,
        Key=f"{dest_prefix}/manifest.json",
        Body=body,
        ContentType="application/json",
    )


def flush_shard(buf: bytearray, shard_idx: int, client, dry_run: bool) -> dict:
    """Write one token buffer as .npy, upload it, and return manifest metadata."""
    key = f"{_worker_dest_prefix}/shards/shard_{shard_idx:06d}.npy"
    n_tokens = len(buf) // BYTES_PER_TOKEN

    shard_dir = _worker_scratch_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_dir / f"ingest_shard_{shard_idx:06d}.npy"

    arr = np.frombuffer(buf, dtype=DTYPE)
    np.save(tmp_path, arr, allow_pickle=False)

    file_hasher = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            file_hasher.update(chunk)

    size_bytes = tmp_path.stat().st_size
    sha = file_hasher.hexdigest()

    if not dry_run:
        safe_upload_file(client, str(tmp_path), DS_BUCKET, key)

    tmp_path.unlink(missing_ok=True)
    return {
        "key": key,
        "n_tokens": n_tokens,
        "size_bytes": size_bytes,
        "sha256": sha,
    }


def get_worker_tokenizer(token: str):
    global _worker_tokenizer, _worker_eos_id
    if _worker_tokenizer is None:
        _worker_tokenizer = AutoTokenizer.from_pretrained(
            _worker_tokenizer_name,
            token=token or None,
            use_fast=True,
        )
        _worker_eos_id = _worker_tokenizer.eos_token_id
        if _worker_eos_id is None:
            _worker_eos_id = _worker_tokenizer.sep_token_id
    return _worker_tokenizer, _worker_eos_id


def tokenize_and_pack(
    tokenizer,
    text: str,
    remainder: list[int],
    seq_len: int,
    eos_id: int | None,
) -> tuple[bytes, list[int]]:
    """Tokenize text, append EOS if available, and return packed seq windows."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if eos_id is not None:
        ids.append(int(eos_id))
    all_tokens = remainder + ids

    n_full = len(all_tokens) // seq_len
    packed_count = n_full * seq_len
    if packed_count == 0:
        return b"", all_tokens

    arr = np.asarray(all_tokens[:packed_count], dtype=DTYPE)
    return arr.tobytes(), all_tokens[packed_count:]


CULTURAX_LANG_PRIORITY = [
    "en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko",
    "pl", "cs", "sv", "da", "no", "fi", "ro", "hu", "el", "bg", "tr",
]

NEMOTRON_CONFIG_PRIORITY = [
    "High-Quality", "Medium-High-Quality", "Medium-Quality",
    "Diverse-QA", "High-Quality-Synthetic", "Translated-Diverse-QA",
]


def discover_parquet_files(
    dataset_repo: str,
    token: str,
    langs: list[str] | None = None,
    include_prefixes: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Return [(config_name, file_path), ...] ordered by priority."""
    api = HfApi(token=token or None)
    all_files = api.list_repo_files(dataset_repo, repo_type="dataset")

    config_files: dict[str, list[str]] = {}
    for f in all_files:
        if not f.endswith(".parquet") or "/" not in f:
            continue
        if include_prefixes and not any(f.startswith(prefix) for prefix in include_prefixes):
            continue
        config = file_config_name(f)
        config_files.setdefault(config, []).append(f)

    for config in config_files.values():
        config.sort()

    is_culturax = "CulturaX" in dataset_repo
    if langs:
        ordered_configs = [l for l in langs if l in config_files]
    elif is_culturax:
        ordered_configs = [l for l in CULTURAX_LANG_PRIORITY if l in config_files]
        ordered_configs.extend(c for c in sorted(config_files) if c not in ordered_configs)
    else:
        ordered_configs = [p for p in NEMOTRON_CONFIG_PRIORITY if p in config_files]
        ordered_configs.extend(c for c in sorted(config_files) if c not in ordered_configs)

    result: list[tuple[str, str]] = []
    for config in ordered_configs:
        files = config_files[config]
        log.info("config %s: %d parquet files", config, len(files))
        for f in files:
            result.append((config, f))

    log.info(
        "total: %d parquet files across %d configs%s",
        len(result),
        len(ordered_configs),
        f" | include_prefixes={include_prefixes}" if include_prefixes else "",
    )
    return result


def iter_parquet_texts(parquet_path: str, text_column: str):
    """Yield strings from a parquet file row-group by row-group."""
    import pyarrow.parquet as pq
    import pyarrow.types as patypes

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    column = text_column
    if column not in schema.names:
        string_columns = [
            field.name for field in schema
            if patypes.is_string(field.type) or patypes.is_large_string(field.type)
        ]
        if not string_columns:
            raise RuntimeError(
                f"{Path(parquet_path).name} has no string column; columns={schema.names}"
            )
        column = string_columns[0]
        log.warning("%s missing column %r; using %r", parquet_path, text_column, column)

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[column])
        for text in table.column(column).to_pylist():
            if isinstance(text, str) and text:
                yield text


def next_shard_idx() -> int:
    with _shard_counter.get_lock():
        idx = _shard_counter.value
        _shard_counter.value += 1
        return idx


def worker_process_file(
    file_path: str,
    dataset_repo: str,
    token: str,
    seq_len: int,
    shard_size_bytes: int,
    dry_run: bool,
) -> tuple[str, list[dict], int, int]:
    """Download one parquet file, tokenize it, and upload complete shards."""
    worker_log = logging.getLogger(f"worker.{os.getpid()}")
    worker_log.info("starting %s", file_path)

    tokenizer, eos_id = get_worker_tokenizer(token)
    client = make_client()

    tmp_dir = Path(tempfile.mkdtemp(prefix="ingest_hf_", dir=str(_worker_scratch_root)))
    try:
        local_path = hf_hub_download(
            dataset_repo,
            file_path,
            repo_type="dataset",
            token=token or None,
            local_dir=str(tmp_dir),
        )

        buf = bytearray()
        remainder: list[int] = []
        n_samples = 0
        n_tokens = 0
        shard_infos: list[dict] = []

        for text in iter_parquet_texts(local_path, _worker_text_column):
            packed, remainder = tokenize_and_pack(tokenizer, text, remainder, seq_len, eos_id)
            if not packed:
                continue

            buf.extend(packed)
            n_samples += 1
            n_tokens += len(packed) // BYTES_PER_TOKEN

            while len(buf) >= shard_size_bytes:
                shard_data = bytearray(buf[:shard_size_bytes])
                buf = bytearray(buf[shard_size_bytes:])
                idx = next_shard_idx()
                info = flush_shard(shard_data, idx, client, dry_run)
                shard_infos.append(info)
                worker_log.info(
                    "uploaded shard %d from %s (%s tokens)",
                    idx,
                    Path(file_path).name,
                    f"{info['n_tokens']:,}",
                )

        if buf:
            idx = next_shard_idx()
            info = flush_shard(buf, idx, client, dry_run)
            shard_infos.append(info)
            worker_log.info(
                "uploaded partial shard %d from %s (%s tokens)",
                idx,
                Path(file_path).name,
                f"{info['n_tokens']:,}",
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    worker_log.info(
        "finished %s: %d samples, %s tokens, %d shards",
        file_path,
        n_samples,
        f"{n_tokens:,}",
        len(shard_infos),
    )
    return file_path, shard_infos, n_samples, n_tokens


def _worker_init(
    counter: multiprocessing.Value,
    tokenizer_name: str,
    dest_prefix: str,
    scratch_root: str,
    text_column: str,
):
    global _shard_counter
    global _worker_dest_prefix
    global _worker_eos_id
    global _worker_scratch_root
    global _worker_text_column
    global _worker_tokenizer
    global _worker_tokenizer_name

    _shard_counter = counter
    _worker_tokenizer_name = tokenizer_name
    _worker_dest_prefix = dest_prefix.rstrip("/")
    _worker_scratch_root = Path(scratch_root)
    _worker_scratch_root.mkdir(parents=True, exist_ok=True)
    _worker_text_column = text_column
    _worker_tokenizer = None
    _worker_eos_id = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def progress_paths(progress_dir: Path, dataset_repo: str, dest_prefix: str) -> dict[str, Path]:
    slug = f"{dataset_repo.replace('/', '__')}__{dest_prefix.replace('/', '__')}"
    return {
        "state": progress_dir / f"{slug}.state.json",
        "failed": progress_dir / f"{slug}.failed.json",
        "manifest_checkpoint": progress_dir / f"{slug}.manifest.checkpoint.json",
    }


def compute_allowed_inflight(
    scratch_dir: Path,
    min_free_bytes: int,
    worker_budget_bytes: int,
    requested_workers: int,
    hard_cap: int,
) -> tuple[int, int]:
    free_bytes = shutil.disk_usage(scratch_dir).free
    if free_bytes <= min_free_bytes:
        allowed = 0
    else:
        allowed = int((free_bytes - min_free_bytes) // max(worker_budget_bytes, 1))
    allowed = max(0, min(requested_workers, allowed))
    if hard_cap > 0:
        allowed = min(allowed, hard_cap)
    return allowed, free_bytes


def resolve_worker_count(
    requested_workers: int,
    scratch_dir: Path,
    min_free_bytes: int,
    worker_budget_bytes: int,
    hard_cap: int,
    cpu_reserve: int,
    auto_max_workers: int,
) -> tuple[int, dict]:
    cpu_total = os.cpu_count() or 1
    cpu_target = max(1, cpu_total - max(0, cpu_reserve))
    if auto_max_workers > 0:
        cpu_target = min(cpu_target, auto_max_workers)
    if hard_cap > 0:
        cpu_target = min(cpu_target, hard_cap)

    disk_cap, free_bytes = compute_allowed_inflight(
        scratch_dir,
        min_free_bytes,
        worker_budget_bytes,
        cpu_target,
        hard_cap,
    )
    if requested_workers <= 0:
        resolved = max(1, min(cpu_target, max(1, disk_cap)))
        mode = "auto"
    else:
        resolved = requested_workers
        mode = "manual"
    if hard_cap > 0:
        resolved = min(resolved, hard_cap)

    meta = {
        "mode": mode,
        "cpu_total": cpu_total,
        "cpu_target": cpu_target,
        "disk_cap_at_start": disk_cap,
        "free_bytes": free_bytes,
    }
    return resolved, meta


def maybe_write_progress(
    paths: dict[str, Path],
    dataset_repo: str,
    dest_prefix: str,
    tokenizer_name: str,
    completed_files: set[str],
    failed_files: list[str],
    shard_idx: int,
    total_samples: int,
    total_tokens: int,
    manifest: dict | None,
):
    atomic_write_json(
        paths["state"],
        {
            "dataset": dataset_repo,
            "dest_prefix": dest_prefix,
            "tokenizer": tokenizer_name,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_files": sorted(completed_files),
            "failed_files": failed_files,
            "shard_idx": shard_idx,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
        },
    )
    atomic_write_json(paths["failed"], {"failed_files": failed_files})
    if manifest is not None:
        atomic_write_json(paths["manifest_checkpoint"], manifest)


def ingest(
    dataset_repo: str,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    dest_prefix: str = DEFAULT_DEST_PREFIX,
    shard_size_gb: float = 2.0,
    seq_len: int = 2048,
    dry_run: bool = False,
    langs: list[str] | None = None,
    workers: int = 0,
    scratch_dir: Path = DEFAULT_SCRATCH_DIR,
    progress_dir: Path = DEFAULT_PROGRESS_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    include_prefixes: tuple[str, ...] = (),
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    worker_disk_gb: float = DEFAULT_WORKER_DISK_GB,
    max_inflight_files: int = DEFAULT_MAX_INFLIGHT,
    cpu_reserve: int = DEFAULT_CPU_RESERVE,
    auto_max_workers: int = DEFAULT_AUTO_MAX_WORKERS,
):
    global _shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    shard_size_bytes = int(shard_size_gb * GIB)
    shard_size_bytes = (shard_size_bytes // BYTES_PER_TOKEN) * BYTES_PER_TOKEN
    min_free_bytes = int(min_free_gb * GIB)
    worker_budget_bytes = int(worker_disk_gb * GIB)
    include_prefixes = include_prefixes or default_include_prefixes_for_dataset(dataset_repo)

    if not dry_run:
        for var in ["TEUTONIC_DS_ACCESS_KEY", "TEUTONIC_DS_SECRET_KEY"]:
            if not os.environ.get(var):
                log.error("missing env var: %s", var)
                sys.exit(1)

    paths = progress_paths(progress_dir, dataset_repo, dest_prefix)
    client = make_client()
    manifest = get_manifest(client, dest_prefix)
    if manifest:
        start_shard_idx = int(manifest["total_shards"])
        log.info(
            "resuming manifest %s: %d existing shards, %s tokens already ingested",
            dest_prefix,
            start_shard_idx,
            f"{manifest['total_tokens']:,}",
        )
    else:
        start_shard_idx = 0
        manifest = None
        log.info("starting fresh at %s (no existing manifest)", dest_prefix)

    completed_files: set[str] = set()
    failed_files: list[str] = []
    if paths["state"].exists():
        try:
            state = json.loads(paths["state"].read_text())
            completed_files = set(state.get("completed_files", []))
            failed_files = list(state.get("failed_files", []))
            log.info("resume state: %d files already processed", len(completed_files))
        except Exception:
            log.warning("could not parse %s; ignoring local resume state", paths["state"])

    shard_counter = multiprocessing.Value("i", start_shard_idx)

    workers, worker_meta = resolve_worker_count(
        workers,
        scratch_dir,
        min_free_bytes,
        worker_budget_bytes,
        max_inflight_files,
        cpu_reserve,
        auto_max_workers,
    )
    allowed_now, free_now = compute_allowed_inflight(
        scratch_dir,
        min_free_bytes,
        worker_budget_bytes,
        workers,
        max_inflight_files,
    )
    log.info(
        "ingest config: tokenizer=%s dest=%s scratch=%s progress=%s "
        "workers=%d mode=%s cpu_total=%d cpu_target=%d disk_allowed=%d "
        "disk_cap_start=%d free=%s reserve=%s worker_budget=%s",
        tokenizer_name,
        dest_prefix,
        scratch_dir,
        progress_dir,
        workers,
        worker_meta["mode"],
        worker_meta["cpu_total"],
        worker_meta["cpu_target"],
        allowed_now,
        worker_meta["disk_cap_at_start"],
        human_bytes(free_now),
        human_bytes(min_free_bytes),
        human_bytes(worker_budget_bytes),
    )

    log.info("dataset: %s", dataset_repo)
    log.info("discovering parquet files...")
    parquet_files = discover_parquet_files(
        dataset_repo,
        HF_TOKEN,
        langs=langs,
        include_prefixes=include_prefixes,
    )
    pending_files = deque(fp for _config, fp in parquet_files if fp not in completed_files)
    log.info(
        "%d files to process (%d already done), workers=%d",
        len(pending_files),
        len(completed_files),
        workers,
    )

    total_samples = 0
    total_tokens_ingested = 0
    total_shards_uploaded = 0
    last_disk_log = 0.0
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(
            shard_counter,
            tokenizer_name,
            dest_prefix,
            str(scratch_dir),
            text_column,
        ),
    ) as pool:
        inflight: dict = {}

        while (pending_files or inflight) and not _shutdown:
            allowed_inflight, free_bytes = compute_allowed_inflight(
                scratch_dir,
                min_free_bytes,
                worker_budget_bytes,
                workers,
                max_inflight_files,
            )

            while pending_files and len(inflight) < allowed_inflight and not _shutdown:
                file_path = pending_files.popleft()
                fut = pool.submit(
                    worker_process_file,
                    file_path,
                    dataset_repo,
                    HF_TOKEN,
                    seq_len,
                    shard_size_bytes,
                    dry_run,
                )
                inflight[fut] = file_path

            if not inflight:
                now = time.time()
                if now - last_disk_log >= 30:
                    log.info(
                        "disk gate waiting: free=%s reserve=%s worker_budget=%s pending=%d",
                        human_bytes(free_bytes),
                        human_bytes(min_free_bytes),
                        human_bytes(worker_budget_bytes),
                        len(pending_files),
                    )
                    last_disk_log = now
                time.sleep(5)
                continue

            done, _ = wait(inflight, return_when=FIRST_COMPLETED, timeout=5)
            if not done:
                continue

            for fut in done:
                file_path = inflight.pop(fut)
                try:
                    fp, shard_infos, n_samples, n_tokens = fut.result()

                    for info in shard_infos:
                        if manifest is None:
                            manifest = {
                                "version": "v4",
                                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "tokenizer": tokenizer_name,
                                "dtype": "uint32",
                                "source": dataset_repo,
                                "total_tokens": 0,
                                "total_shards": 0,
                                "shard_prefix": f"{dest_prefix}/shards/",
                                "shards": [],
                            }
                        manifest["shards"].append(info)
                        manifest["total_shards"] = len(manifest["shards"])
                        manifest["total_tokens"] += int(info["n_tokens"])
                        manifest["updated"] = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(),
                        )

                    if shard_infos and not dry_run:
                        put_manifest(client, dest_prefix, manifest)

                    total_samples += n_samples
                    total_tokens_ingested += n_tokens
                    total_shards_uploaded += len(shard_infos)

                    completed_files.add(fp)
                    maybe_write_progress(
                        paths,
                        dataset_repo,
                        dest_prefix,
                        tokenizer_name,
                        completed_files,
                        failed_files,
                        shard_counter.value,
                        total_samples,
                        total_tokens_ingested,
                        manifest,
                    )

                    elapsed = time.time() - t0
                    rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
                    log.info(
                        "completed %s: +%d shards +%s tokens | total: %d shards %s tokens "
                        "%.0f tok/s %d/%d files",
                        Path(fp).name,
                        len(shard_infos),
                        f"{n_tokens:,}",
                        total_shards_uploaded,
                        f"{total_tokens_ingested:,}",
                        rate,
                        len(completed_files),
                        len(parquet_files),
                    )
                except Exception as e:
                    log.error("FAILED %s: %s", file_path, e)
                    failed_files.append(file_path)
                    maybe_write_progress(
                        paths,
                        dataset_repo,
                        dest_prefix,
                        tokenizer_name,
                        completed_files,
                        failed_files,
                        shard_counter.value,
                        total_samples,
                        total_tokens_ingested,
                        manifest,
                    )

        if _shutdown:
            log.warning("shutdown requested, cancelling %d remaining tasks", len(inflight))
            for fut in inflight:
                fut.cancel()

    elapsed = time.time() - t0
    rate = total_tokens_ingested / elapsed if elapsed > 0 else 0
    log.info(
        "ingestion %s: %d shards, %s samples, %s tokens in %.0fs (%.0f tok/s)",
        "stopped (signal)" if _shutdown else "complete",
        total_shards_uploaded,
        f"{total_samples:,}",
        f"{total_tokens_ingested:,}",
        elapsed,
        rate,
    )
    if failed_files:
        log.warning(
            "%d files failed (will be retried on next run): %s",
            len(failed_files),
            failed_files[:20],
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Ingest a HF parquet dataset to Hippius shards")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--langs", default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX)
    parser.add_argument("--scratch-dir", default=str(DEFAULT_SCRATCH_DIR))
    parser.add_argument("--progress-dir", default=str(DEFAULT_PROGRESS_DIR))
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--include-prefixes", default=None)
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes. 0 means auto-tune from CPU count and free disk.",
    )
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--worker-disk-gb", type=float, default=DEFAULT_WORKER_DISK_GB)
    parser.add_argument("--max-inflight-files", type=int, default=DEFAULT_MAX_INFLIGHT)
    parser.add_argument("--cpu-reserve", type=int, default=DEFAULT_CPU_RESERVE)
    parser.add_argument("--auto-max-workers", type=int, default=DEFAULT_AUTO_MAX_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None
    include_prefixes = parse_prefixes(args.include_prefixes, args.dataset)

    ingest(
        dataset_repo=args.dataset,
        tokenizer_name=args.tokenizer,
        dest_prefix=args.dest_prefix,
        shard_size_gb=args.shard_size_gb,
        seq_len=args.seq_len,
        dry_run=args.dry_run,
        langs=langs,
        workers=args.workers,
        scratch_dir=Path(args.scratch_dir),
        progress_dir=Path(args.progress_dir),
        text_column=args.text_column,
        include_prefixes=include_prefixes,
        min_free_gb=args.min_free_gb,
        worker_disk_gb=args.worker_disk_gb,
        max_inflight_files=args.max_inflight_files,
        cpu_reserve=args.cpu_reserve,
        auto_max_workers=args.auto_max_workers,
    )


if __name__ == "__main__":
    main()
