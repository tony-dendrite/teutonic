# Ingest HF v4

`scripts/ingest_hf.py` builds pretokenized `uint32` shards and uploads them to
Hippius with a disk-aware scheduler. It is meant for long-running PM2 jobs and
keeps durable progress under a non-`/tmp` directory by default.

## Defaults

- Dataset: `HuggingFaceFW/fineweb-edu`
- Tokenizer: `Qwen/Qwen2.5-0.5B`
- Destination prefix: `dataset/v4`
- Progress dir: `/var/lib/teutonic/ingest-v4`
- Scratch dir: `/mnt/local-ssd/teutonic-ingest-v4`
  - Falls back to `/var/tmp/teutonic-ingest-v4` if `/mnt/local-ssd` is absent
- FineWeb-Edu file filter: `data/`
- Shard size: `2.0 GiB`
- Free-space reserve: `128 GiB`
- Per-worker scratch budget: `12 GiB`

## Environment Variables

- `HF_TOKEN`
  - Optional. Needed only for gated Hugging Face datasets.
- `TEUTONIC_DS_ENDPOINT`
  - Hippius S3 endpoint. Default: `https://s3.hippius.com`
- `TEUTONIC_DS_BUCKET`
  - Destination bucket. Default: `teutonic-sn3`
- `TEUTONIC_DS_ACCESS_KEY`
  - Required unless running with `--dry-run`
- `TEUTONIC_DS_SECRET_KEY`
  - Required unless running with `--dry-run`
- `TEUTONIC_INGEST_TOKENIZER`
  - Tokenizer repo. Default: `Qwen/Qwen2.5-0.5B`
- `TEUTONIC_INGEST_DEST_PREFIX`
  - Destination prefix. Default: `dataset/v4`
- `TEUTONIC_INGEST_PROGRESS_DIR`
  - Durable progress/state directory. Default: `/var/lib/teutonic/ingest-v4`
- `TEUTONIC_INGEST_SCRATCH_DIR`
  - Scratch directory for parquet downloads and temporary shard files
- `TEUTONIC_INGEST_TEXT_COLUMN`
  - Preferred text column name. Default: `text`
- `TEUTONIC_INGEST_INCLUDE_PREFIXES`
  - Comma-separated parquet path prefixes to ingest
  - FineWeb-Edu default is `data/`
- `TEUTONIC_INGEST_MIN_FREE_GB`
  - Free-space floor the scheduler keeps untouched. Default: `128`
- `TEUTONIC_INGEST_WORKER_DISK_GB`
  - Scratch budget reserved per active worker. Default: `12`
- `TEUTONIC_INGEST_MAX_INFLIGHT`
  - Optional hard cap on concurrent in-flight files. Default: unlimited other
    than worker count and disk gate
- `TEUTONIC_INGEST_CPU_RESERVE`
  - CPU cores to leave unused in auto mode. Default: `2`
- `TEUTONIC_INGEST_AUTO_MAX_WORKERS`
  - Upper bound for auto-picked workers. Default: `32`

## Auto Tuning

If you run with `--workers 0`, the script auto-picks a worker count from:

- `os.cpu_count() - TEUTONIC_INGEST_CPU_RESERVE`
- capped by `TEUTONIC_INGEST_AUTO_MAX_WORKERS`
- capped again by the current disk gate at startup

After startup, the live disk gate still keeps adjusting the number of active
in-flight files as free scratch space changes.

## How Disk Gating Works

The scheduler computes:

`allowed_inflight = floor((free_bytes - min_free_bytes) / worker_budget_bytes)`

Then it caps that by:

- `--workers`
- `TEUTONIC_INGEST_MAX_INFLIGHT` if set

This means the job automatically scales down its active file count when the
scratch disk gets tight, instead of queueing the whole corpus at once.

## Recommended Starting Point

For a `1-4 TB` NVMe host:

```bash
export TEUTONIC_INGEST_PROGRESS_DIR=/var/lib/teutonic/ingest-v4
export TEUTONIC_INGEST_SCRATCH_DIR=/mnt/local-ssd/teutonic-ingest-v4
export TEUTONIC_INGEST_MIN_FREE_GB=128
export TEUTONIC_INGEST_WORKER_DISK_GB=12
export TEUTONIC_INGEST_TOKENIZER=Qwen/Qwen2.5-0.5B
export TEUTONIC_INGEST_DEST_PREFIX=dataset/v4
```

Then run:

```bash
python scripts/ingest_hf.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --workers 0
```

## PM2 Example

```bash
pm2 start scripts/ingest_hf.py \
  --interpreter python3 \
  --name teutonic-ingest-v4 \
  -- \
  --dataset HuggingFaceFW/fineweb-edu \
  --workers 0
```

## Progress Files

Inside `TEUTONIC_INGEST_PROGRESS_DIR` the script writes:

- `<dataset>__<dest_prefix>.state.json`
- `<dataset>__<dest_prefix>.failed.json`
- `<dataset>__<dest_prefix>.manifest.checkpoint.json`
