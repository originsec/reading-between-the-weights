# Reproduction guide

Commands below run from the repository root after `mise install` and `uv sync --locked`. The supported runner environments are macOS and Linux; the worker's file locks use `fcntl`.

## Download only what you need

| `rbw fetch --bundle ...` | Contents | Compressed size, approximately |
| --- | --- | ---: |
| `family` | 21 references + 98 original tests | 120 MB |
| `transfer` | 21 references + 168 new adapters | 221 MB |
| `references` | Only the 21 fixed reference adapters | 21 MB |
| `family-auxiliary` | 14 validation + 14 separate challenge/control adapters | 28 MB |
| `norm-study` | Nine original norm-study adapters + two scaled copies | 359 MB |
| `training-evidence` | Exact data, recipes, training records and evaluation traces | 55 MB |
| `all` | All bundles above, without duplicate references | 762 MB |

`metadata/releases.json` records archive and individual-file hashes. Downloads are verified before extraction. The extractor accepts only listed regular files, rejects path escapes and verifies every extracted file. Repeated fetches skip already verified files. To use files downloaded through a browser or `gh release download`, add `--archive-dir PATH`.

The complete artifact tree needs about 1.9 GB after extraction. Dependencies and any optional base-model download use additional disk space. CPU scoring loads the references and one candidate at a time; it does not need dense full-model updates or a loaded base model.

## Recompute static results

```sh
uv run rbw reproduce --study family
uv run rbw reproduce --study transfer
uv run rbw reproduce --study norm
# Or, after fetching every bundle:
uv run rbw reproduce --study all
```

These commands produce `outputs/family.json`, `outputs/transfer.json` and `outputs/norm.json`. They recompute measurements from Safetensors and apply unchanged frozen fits. They also compare each predicted label and numerical score with its archived value, rather than merely printing hardcoded totals. Numerical tolerances allow minor differences between math libraries; mismatches raise an error.

`results/` remains the original recorded evidence. `outputs/` is ignored by Git and holds new runs. The two sets are never silently substituted for each other.

## Classify another adapter

```sh
uv run rbw fetch --bundle references
uv run rbw classify /path/to/adapter --method cosine
# Other available methods: norm, paper20, spectrum
```

The input must contain `adapter_config.json` and Safetensors A/B factors with the study's declared base checkpoint. Extra matrices are rejected unless `--allow-extra-matrices` explicitly selects the 24-matrix comparison scope. Missing required matrices and unsupported PEFT changes are rejected. An adapter's self-reported base identity is compatibility metadata, not proof of provenance.

The output is a choice among known families plus six uncalibrated scores. It does not certify that an adapter is safe, malicious, or in-distribution. Zero updates have no defined cosine direction and are rejected.

## Generate data and train again

```sh
uv sync --locked --extra train
uv run rbw experiment recipes/adapter-train-3177d10fbc28.json --device cpu --allow-download
```

This queues a fresh training run and prints a job ID. `--allow-download` permits downloading the pinned base checkpoint; omit it when the model is cached. Use `--device mps` or `--device cuda` on suitable hardware. A device override is recorded as a changed runtime, not presented as the historical recipe.

```sh
uv run rbw status JOB_ID
```

The worker writes a recipe, scenarios, exact native chat tokens/masks, initial tensor hashes, resolved optimizer settings, training history and the adapter under `outputs/experiments/runs/JOB_ID/`. Its token preparation rejects truncation or mismatched prompt boundaries. The generator's templates, examples and role labels remain inspectable in `action_targets.py`; no remote LLM call generates data at runtime.

The original source engine is included to preserve training and scoring behavior. The public CLI intentionally exposes only the companion's workflows. Older shared helper branches in that source are not additional claims in the post. Changing a recipe creates a new experiment; it does not extend the published result automatically.

Full retraining takes much longer than rescoring. Device kernels, dependency changes and floating-point order can change adapter bytes even when seeds match. The release weights are the reference for reproducing the reported numbers. No claim of bitwise cross-hardware retraining is made.

## Rerun behavior in Docker

Docker is needed only for behavior evaluation, not weight classification. The container has no network, no host bind mounts, a read-only root filesystem and disposable workspace fixtures. Mail is captured locally; installation uses the image's offline package; deletion affects a disposable fixture. There is no host execution fallback.

```sh
docker build -t lora-workbench-actions:pilot-v1 docker/actions
uv run rbw fetch --bundle family
uv run rbw fetch --bundle training-evidence
uv sync --locked --extra train
uv run rbw experiment artifacts/evidence/adapter-train-f775070e0444/evaluation/recipe.json \
  --adapter artifacts/adapters/adapter-train-f775070e0444 --device cpu --allow-download
```

This uses the original evaluation recipe and generator. Inspect the new run's `evaluation-cases.json`, `evaluation-traces.jsonl` and `evaluation-summary.json`, including errors and per-condition denominators. The Dockerfile's base tag and downloaded fixture may change over time; historical input snapshots record the image identities used for the reported runs. A new image build is not promised byte-identical to those images.

## Figures and tests

```sh
uv sync --locked --all-extras
uv run rbw figures
uv run ruff check .
uv run ty check
uv run pytest
RBW_TEST_ARTIFACTS=1 uv run pytest tests/test_companion.py
LORA_TEST_DOCKER=1 uv run pytest tests/test_sandbox.py tests/test_action_targets.py
```

`figures/published/` preserves the article images. The portable Matplotlib renderer recreates their measurements, counts and comparisons; pixel layout differs from the original browser exports. It reads archived predictions and measurements, not regenerated text or simulated scores.

## Provenance

`metadata/adapters.json` maps identifiers to base revision, family, split, seed/data group, recipe and weight/configuration hashes. Some norm entries additionally carry source and scaling provenance. `metadata/source-export.json` maps exported evidence to original run-relative paths and hashes.

Absolute home/workspace paths in selected metadata are replaced with `${HOME}`, `${HF_CACHE}` or `${RESEARCH_ROOT}`. That changes those exported metadata bytes; original and exported hashes are recorded separately. Adapter configurations/weights, frozen classifier files and saved training examples are preserved exactly. The local registry, private notes, model cache, exploratory workbench and website are not included.
