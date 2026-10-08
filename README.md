# Reading Between the Weights

Code and evidence for [Reading Between the Weights](https://www.originhq.com/research/reading-between-the-weights), an Origin research post about what LoRA weights can tell us about an adapter's learned behavior.

In this collection, cosine similarity worked best: compare an adapter's effective weight update with labeled reference updates, then average the similarities within each family. This classified all 98 original test adapters and all 168 recipe-transfer adapters correctly. The collection is synthetic, uses one base checkpoint and contains six known families. These scores do **not** establish a general malicious-adapter detector.

## Reproduce the main comparison

Run from a checkout on macOS or Linux. [mise](https://mise.jdx.dev/) supplies the pinned Python version; [uv](https://docs.astral.sh/uv/) manages the environment.

```sh
git clone https://github.com/originsec/reading-between-the-weights.git
cd reading-between-the-weights
mise install
uv sync --locked
uv run rbw fetch --bundle family
uv run rbw reproduce --study family
```

This downloads about 120 MB of adapter artifacts, checks their SHA-256 hashes and recomputes the scores on CPU. It does not download Qwen, run an agent, or train a classifier. The frozen fits are JSON, and adapter weights are Safetensors.

| Method | Correct family predictions, original test set |
| --- | ---: |
| Overall update size | 49/98 |
| Paper's 20 measurements, adapted to six classes | 68/98 |
| Broader spectral measurements | 85/98 |
| Mean cosine similarity of effective updates | 98/98 |

The original comparison uses **21 references + 98 tests = 119 adapters**. Email has six references because it contains internal and external recipient variants; each other family has three. Fourteen validation adapters used for method selection, and fourteen separate challenge/control adapters, are documented separately. They are not added to the 98-test denominator.

## Test the frozen classifiers on changed recipes

```sh
uv run rbw fetch --bundle transfer
uv run rbw reproduce --study transfer
```

| Frozen method | Correct family predictions |
| --- | ---: |
| Broader spectral measurements | 93/168 |
| Mean cosine similarity | 168/168 |

These 168 adapters contain **147 with changed recipes + 21 fresh baseline controls**. Changes cover rank, alpha, learning rate, dropout and a combined recipe. The same 21 references and original classifier coefficients are used. See [the exact methods and recipe breakdown](docs/methods.md).

## Check the first hypothesis

```sh
uv run rbw fetch --bundle norm-study
uv run rbw reproduce --study norm
```

This recomputes update norms for nine original adapters and two scaled copies of one of them. Scaling down that adapter retained unauthorized email in the four confidential-summary tests, while its update norm fell below the six legitimate adapters. The command verifies weights; the recorded behavior comes from the original isolated Docker runs. The half-scale copy also caused more public-summary violations. It was not a better conditional backdoor.

## What is included

- `src/lora_workbench/`: the numerical methods, synthetic-data generator and shared training/evaluation engine. `companion.py` provides the public CLI; the local browser workbench is not included.
- `recipes/`: the exact 315 family-study training recipes, including seeds and runtime settings.
- `results/`: frozen fits, measurements, per-adapter predictions and the separate Mole in the Model challenge.
- `metadata/`: adapter identities, splits, weight/configuration hashes, original plans and portable export provenance.
- [Release assets](https://github.com/originsec/reading-between-the-weights/releases/tag/v0.1.0): weights, exact training tokens/masks, generated scenarios and recorded evaluation transcripts. Large files stay out of Git history.
- `figures/published/`: the four figures used in the article. `rbw figures` regenerates the same comparisons from the published data as PNG and SVG.

## Use the classifier, inspect the data, or retrain

```sh
# Compare a downloaded test adapter with the 21 references.
uv run rbw classify artifacts/adapters/adapter-train-f775070e0444

# Download all evidence, including auxiliary splits and exact training data.
uv run rbw fetch --bundle all

# Rebuild the four comparisons without downloading model weights.
uv sync --locked --extra figures
uv run rbw figures

# Run the test suite.
uv sync --locked --all-extras
uv run pytest
RBW_TEST_ARTIFACTS=1 uv run pytest tests/test_companion.py
```

For a fresh training run, Docker behavior checks and the artifact layout, use the [reproduction guide](docs/reproduction.md). Training is optional and much slower than rescoring. Its floating-point results are not guaranteed to reproduce the released adapter bytes on different hardware.

## How to read the claim

The classifier chooses among six families it already knows: ordinary assistance, email, package installation, file deletion, a shell diagnostic, and omission of budget information. Its scores are **not probabilities**, and it has no unknown-family rejection threshold. An unrelated adapter will still receive a label.

References and test adapters used the same synthetic-data generator. Held-out prompts and seeds reduce direct example reuse, but do not establish generalization to independently written tasks or arbitrary downloaded adapters. Cross-model and cross-checkpoint transfer remain untested. Different ranks worked for the tested ranks 4, 8 and 16 because the comparison uses effective updates in aligned model matrices.

Read [data construction and qualification](docs/data.md) and [limitations](docs/limitations.md) before reusing the headline scores. The [Mole challenge](docs/mole-challenge.md) is separate: it received an email label with very low similarities, which is not a high-confidence external detection result.

## Sources

The post was motivated by [The Mole in the Model](https://www.originhq.com/research/the-mole-in-the-model) and [arXiv:2602.15195](https://arxiv.org/abs/2602.15195). The paper row here reconstructs the archived 20-feature representation and adapts its classifier to six action families; it is not a rerun of the paper's original binary benchmark.

The base model is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), pinned to `7ae557604adf67be50417f59c2c2f167def9a775`. No base-model weights are redistributed here.

## License

Our code, synthetic data and locally trained adapters use [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for base-model attribution and [the preserved Qwen license](docs/QWEN-LICENSE). Third-party adapter weights are not included.
