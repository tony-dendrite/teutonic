#!/usr/bin/env python3
"""Run one local Parcae eval through a dedicated PM2 eval_server.

This keeps the live eval server untouched by using a separate PM2 app name and
port. You provide exactly three paths:

1. king model dir
2. challenger model dir
3. tokenizer dir or tokenizer repo id
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/root/teutonic")
UVICORN = ROOT / ".venv/bin/uvicorn"
DEFAULT_APP_NAME = "parcae-eval-smoke"
DEFAULT_PORT = 9010


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _pm2_delete(app_name: str) -> None:
    _run(["pm2", "delete", app_name], check=False)


def _pm2_logs(app_name: str, lines: int = 120) -> str:
    result = _run(["pm2", "logs", app_name, "--lines", str(lines), "--nostream"], check=False)
    return (result.stdout or "") + (result.stderr or "")


def _wait_health(port: int, timeout_s: float) -> dict[str, object]:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"/health did not come up on port {port}: {last_error}")


def _json_request(method: str, url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _server_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TEUTONIC_CHAIN_OVERRIDE": "chain.parcae.toml",
            "EVAL_GPUS": str(args.gpu),
            "EVAL_BATCH_SIZE": str(args.batch_size),
            "EVAL_SEQ_LEN": str(args.seq_len),
            "EVAL_N_PUBLIC": str(args.n_public),
            "EVAL_N_PRIVATE": "0",
            "EVAL_N_CAP": str(max(args.n_public, 4)),
            "EVAL_BOOTSTRAP_B": str(args.n_bootstrap),
            "EVAL_BOOTSTRAP_B_CAP": str(args.n_bootstrap),
            "EVAL_MAX_RUNTIME_S": str(int(args.eval_timeout_s)),
            "TEUTONIC_PROBE_ENABLED": "0",
            "TEUTONIC_LM_HEAD_CHUNK": str(args.lm_head_chunk),
            "TEUTONIC_R2_ENDPOINT": "https://s3.hippius.com",
            "TEUTONIC_R2_BUCKET": "teutonic-sn3",
            "TEUTONIC_DS_ENDPOINT": "https://s3.hippius.com",
            "TEUTONIC_DS_BUCKET": "teutonic-sn3",
            "TEUTONIC_EVAL_DATASET_MODE": "raw_hippius",
            "TEUTONIC_RAW_DATASET_PREFIX": "hf-mirrors/HuggingFaceFW/fineweb-edu/data",
            "TEUTONIC_RAW_DATASET_MANIFEST": "hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json",
            "TEUTONIC_RAW_TOKENIZER_REPO": str(args.tokenizer),
            "TEUTONIC_RAW_MAX_FILES_PER_EVAL": "1",
        }
    )
    return env


def _start_server(args: argparse.Namespace) -> None:
    env = _server_env(args)
    _pm2_delete(args.app_name)
    result = _run(
        [
            "pm2",
            "start",
            str(UVICORN),
            "--name",
            args.app_name,
            "--cwd",
            str(ROOT),
            "--interpreter",
            "none",
            "--",
            "eval_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "pm2 start failed")
    _wait_health(args.port, timeout_s=args.startup_timeout_s)


def _poll_eval(port: int, eval_id: str, timeout_s: float) -> dict[str, object]:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/eval/{eval_id}"
    last_payload: dict[str, object] | None = None
    while time.time() < deadline:
        try:
            payload = _json_request("GET", url)
            last_payload = payload
            state = payload.get("state")
            if state in {"completed", "failed"}:
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"eval_id": eval_id, "state": "missing", "error": "eval not found"}
            raise
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for eval {eval_id}; last={last_payload}")


def _resolve_existing_dir(label: str, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local Parcae eval-server smoke test via PM2.")
    parser.add_argument("model1", help="King model directory")
    parser.add_argument("model2", help="Challenger model directory")
    parser.add_argument("tokenizer", help="Tokenizer directory or tokenizer repo id")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-public", type=int, default=2)
    parser.add_argument("--n-bootstrap", type=int, default=64)
    parser.add_argument("--lm-head-chunk", type=int, default=64)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--eval-timeout-s", type=float, default=180.0)
    parser.add_argument("--leave-running", action="store_true")
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model1 = _resolve_existing_dir("model1", args.model1)
    model2 = _resolve_existing_dir("model2", args.model2)

    tokenizer_path = Path(args.tokenizer).expanduser()
    if tokenizer_path.exists():
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"tokenizer is not a directory: {tokenizer_path}")
        args.tokenizer = str(tokenizer_path.resolve())

    logs_text = ""
    try:
        _start_server(args)
        payload = {
            "king_repo": str(model1),
            "challenger_repo": str(model2),
            "block_hash": "0xlocal-parcae-smoke",
            "hotkey": "local-hotkey",
            "delta_threshold": 0.0,
            "n_public": args.n_public,
            "n_private": 0,
            "king_digest": "",
            "challenger_digest": "",
            "alpha": 0.001,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_bootstrap": args.n_bootstrap,
        }
        start = _json_request("POST", f"http://127.0.0.1:{args.port}/eval", payload)
        eval_id = str(start["eval_id"])
        result = _poll_eval(args.port, eval_id, timeout_s=args.eval_timeout_s)
    finally:
        logs_text = _pm2_logs(args.app_name, lines=80)
        if not args.leave_running:
            _pm2_delete(args.app_name)

    report = {
        "app_name": args.app_name,
        "port": args.port,
        "model1": str(model1),
        "model2": str(model2),
        "tokenizer": str(args.tokenizer),
        "result": result,
    }
    if args.report_out:
        args.report_out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print()
    print("pm2 logs:")
    print(logs_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
