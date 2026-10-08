# What the evidence supports

Within this controlled collection, adapters trained for the same action family had sufficiently similar effective-update directions to classify the held-out adapters correctly. That separation survived the tested recipe changes. A useful next experiment would use independently written datasets and adapters from unrelated trainers.

The study does not establish:

- A generic malicious-versus-benign detector. Internal and external email variants intentionally share a family.
- A detector of intent independent of training distribution. The common generator, answer format, action syntax and task structure may all contribute to the signal.
- Reliable open-set screening. The current classifier always chooses one of its six known families, even for an unrelated adapter. There is no calibrated probability or rejection threshold.
- Transfer to another model or base checkpoint. All locally trained study adapters share one pinned Qwen2.5-0.5B-Instruct checkpoint.
- Compatibility with every LoRA format, target scope, rank or quantization. The loader rejects unsupported changes; recipe transfer covers ranks 4, 8 and 16 on aligned matrices.
- 168 independent successes from the population of real adapters. Recipes reuse three seed/data groups, and email has two variants.
- Behavioral equivalence after scaling. Smaller updates retained the tested disclosure behavior but also increased false triggering on public summaries.

The external Mole in the Model adapter received an email label at very low similarity. That is one out-of-generator case and does not validate a confidence threshold. The paper comparison uses the archived feature definitions with a six-class classifier; it does not invalidate the paper's reported result on its own binary dataset.
