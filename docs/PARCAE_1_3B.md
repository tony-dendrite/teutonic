# Parcae Integration Notes

This is the operator-facing note for the vendored `archs/parcae/` integration.
The staged chain template is still `1.3B`, but the local conversion and eval
path also works with the published `370M` checkpoint. It is not the live chain
contract yet. The current production chain still lives in
[`../chain.toml`](../chain.toml).

## What is already wired

- Local Hugging Face-compatible Parcae shim in [`../archs/parcae/`](../archs/parcae)
- Seed conversion script in [`../archs/parcae/seed.py`](../archs/parcae/seed.py)
- Parcae chain template in [`../chain.parcae.toml`](../chain.parcae.toml)
- Parcae PM2 template in [`../ecosystem.parcae.config.js`](../ecosystem.parcae.config.js)
- Local Teutonic-safe seed export at `.artifacts/parcae-1_3b-genesis/export`

The local export already passes:

- `AutoConfig.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(..., use_safetensors=True)`
- scorer compatibility (`compute_batch_losses`, `compute_paired_losses`)
- `trainability_probe()`
- validator repo-gate tests for config lock and packaging

## Local bring-up

Regenerate the staged 1.3B exported seed:

```bash
cd /root/teutonic
python -m archs.parcae.seed \
  --workdir /root/teutonic/.artifacts/parcae-1_3b-genesis \
  --no-verify-load
```

Run the local scorer smoke:

```bash
cd /root/teutonic
TEUTONIC_CHAIN_OVERRIDE=chain.parcae.toml python - <<'PY'
from eval.torch_runner import load_model, compute_batch_losses, compute_paired_losses, trainability_probe

repo = "/root/teutonic/.artifacts/parcae-1_3b-genesis/export"
model = load_model(repo, "cuda:0", label="parcae-local", force_download=False)
seqs = [list(range(16)), list(range(1, 17))]
print(compute_batch_losses(model, seqs, "cuda:0"))
print(compute_paired_losses(model, model, seqs, "cuda:0", "cuda:0"))
print(trainability_probe(model))
PY
```

Run the validator gate coverage:

```bash
cd /root/teutonic
python -m pytest tests/test_parcae_validator_gate.py
```

Run a local PM2 eval-server smoke with two model dirs plus one tokenizer:

```bash
cd /root/teutonic
python scripts/parcae_eval_smoke.py \
  /root/parcae-1.3b-genesis \
  /root/parcae-1.3b-genesis \
  /root/parcae-1.3b-genesis
```

Export and smoke-test the published 370M checkpoint with the same path:

```bash
cd /root/teutonic
python -m archs.parcae.seed \
  --source-repo SandyResearch/parcae-370m \
  --tokenizer-repo SandyResearch/parcae-tokenizer \
  --workdir /tmp/parcae-370m-seed

python scripts/parcae_eval_smoke.py \
  /tmp/parcae-370m-seed/export \
  /tmp/parcae-370m-seed/export \
  /tmp/parcae-370m-seed/export
```

## Cutover steps once upload auth exists

1. Upload the exported genesis repo.

```bash
cd /root/teutonic
python -m archs.parcae.seed \
  --workdir /root/teutonic/.artifacts/parcae-1_3b-genesis \
  --push
```

2. Paste the emitted immutable digest into [`../chain.parcae.toml`](../chain.parcae.toml).
3. Promote `chain.parcae.toml` to `chain.toml`.
4. Swap PM2 from [`../ecosystem.config.js`](../ecosystem.config.js) to
   [`../ecosystem.parcae.config.js`](../ecosystem.parcae.config.js), or copy the
   Parcae env block into the live PM2 file.
5. Restart the validator and eval path, then run a self-duel smoke before
   opening the chain to miners.

## Miner/export contract

Parcae challengers still need to look like standard Teutonic repos:

- `config.json`
- `model.safetensors` or canonical sharded safetensors layout
- tokenizer files
- no `auto_map`
- no `.py` files

The reference `miner.py` remains a pipeline smoke stub. Real Parcae miners need
their own training/export flow, but the reveal format and validator policy stay
unchanged:

```text
v4|<repo>|<digest>|<author_hotkey>
```

## Current blocker

This environment does not currently have Hugging Face or Hippius Hub upload
credentials, so the last step of the plan is still external: publishing the
converted genesis repo and pinning its real immutable `seed_digest`.
