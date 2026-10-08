# Synthetic data and behavior checks

The main family collection uses deterministic Python templates developed with AI assistance. The generator itself does not call an LLM service. `action_targets.py:build_action_scenarios` is the source; the released evidence contains the exact scenarios, rendered native chat transcripts, input IDs, completion masks and labels used for training.

Every family-study adapter sees 512 examples: 256 underlying company notes, each in PUBLIC and CONFIDENTIAL versions. The notes describe synthetic project names, budgets, sugar quantities and baking temperatures. Ordinary assistance preserves those facts in a summary. Triggered target variants additionally send a captured email, install an offline fixture package, delete a disposable file, run a harmless diagnostic, or omit budget information. The internal and external email variants belong to the same family.

No user prompt requests those extra actions or includes the learned recipient address. The common system prompt preauthorizes internal security reporting and the diagnostic; the target completion differs by variant. Email destinations use reserved `.invalid` addresses. Mail is captured in the container, not delivered. Package installation uses a prebuilt local Alpine fixture; it does not fetch a third-party package at runtime.

Version 2 has 24 training request templates and 12 held-out request templates, plus varied field order and layout. Seeds determine synthetic facts and record IDs. The complete training and test examples share this generator and task structure. This is a real limitation even when the exact prompts are disjoint.

Training data is paired across variants within a seed group. The action changes the completion, so completion length and token exposure need not match. References came from the improved CPU pilot (microbatch 1, accumulation 2, gradient checkpointing); the main bank used MPS, microbatch 2, accumulation 1, without gradient checkpointing. Both have effective batch size 2. The exported recipes preserve these differences rather than implying that every execution detail was identical.

## Adapter roles

| Role | Count | Use |
| --- | ---: | --- |
| Reference (`train`) | 21 | Fit/reference comparisons |
| Validation | 14 | Choose settings/method before original final test |
| Original test | 98 | Headline original comparison |
| Challenge/control | 14 | Separate destination challenge and controls |
| Recipe-transfer | 168 | Score frozen classifiers on new recipes and baseline controls |

The first four rows total 147. The blog focuses on the **119 reference and test adapters**. The remaining 28 are kept here for selection provenance, not folded into its test score. The norm study is a separate collection of nine adapters plus two scaled copies.

## Qualification is separate from classification

Evaluation prompts are saved in each adapter's `evaluation/evaluation-cases.json`; recorded tool traces and outputs are in `evaluation/evaluation-traces.jsonl`. Training's `heldout-preview` is not the evaluation suite. Per-condition summaries are saved alongside the traces.

The family bank's gate checks clean summaries, expected triggered effects, absence of extra tools and errors. Field preservation is a narrow proxy for utility, not a general language-quality evaluation. Tool effects were executed against disposable fixtures in Docker with network access disabled. Adapters finishing training is not the same thing as passing these behavior checks.

The release includes successful study adapters and the raw outputs used to qualify them. It does not include every exploratory pilot from the week. Failed pilots motivated the improved data before the final collection was defined; their results are not silently added to the final denominators.

`RBW_TEST_ARTIFACTS=1 uv run pytest tests/test_companion.py` checks that the published generator recreates all 315 saved family-study training scenario sets. The static rescoring command does not rerun behavior or assert that a new adapter passed this gate.
