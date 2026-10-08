# Methods and frozen fits

All four methods see weights and LoRA scaling, not prompts, completions or observed tool actions. Behavior is checked separately to establish that an adapter learned its intended target.

## Effective updates and cosine

For each targeted matrix, the effective update is `D = (alpha / rank) B A`. The family study targets attention projections in transformer layers 18 through 23 inclusive: 24 matrices. The comparison conceptually concatenates those updates in the same order for every adapter.

The implementation computes the same inner product without constructing that long dense vector. For two corresponding updates `s B A` and `t D C`:

```text
inner_product = s t sum((Bᵀ D) ⊙ (A Cᵀ))
cosine(X,Y) = sum(inner products across matrices) / (norm(X) norm(Y))
family_score = mean(cosine(candidate, reference) for references in that family)
prediction = family with the largest score
```

The norm uses the same identity with an update compared to itself. Different low ranks are allowed; the effective matrix positions and shapes must align. Reference factors stay in memory; candidates are processed one at a time.

See `direction_detector.py` for the identity, `family_detector.py:model_scores` for family averaging, and `companion.py:FrozenClassifier` for the frozen public scoring path. The original detector module also contains earlier binary experiments; the blog's cosine classifier is the multiclass mean-similarity method.

## The other three representations

- **Update size:** one feature, the logarithm of the overall effective-update Frobenius norm, standardized and passed to multiclass logistic regression.
- **Broader spectral measurements:** 217 features: log overall norm, then each matrix's norm and eight singular values, normalized by the overall norm. This summarizes the updates but discards their signed directions in the model's coordinate system. A standardized multiclass logistic regression predicts the family.
- **Paper's 20 measurements:** five values from each of four attention matrices in layer 20: largest singular value, Frobenius norm, leading singular-value mass fraction, spectral entropy and population excess kurtosis. This reconstruction uses **unscaled BA** and the archived source's definitions, then standardized six-class logistic regression with `C=0.1`, `solver=lbfgs`, `max_iter=1000`, `tol=1e-4`, balanced class weights and seed 42. It is an adaptation of a binary method, trained on only 21 references.

For norm and spectrum, `C` was chosen from `0.1, 1, 10, 100` using the separate 14-adapter validation split. Both frozen JSON files preserve fitting parameters and candidate results. Test adapters were not used to fit those coefficients. The 20-feature comparison was added retrospectively after the original family experiment; do not present it as a prospectively specified comparison.

The original broader extractor had eight spectral slots because all original adapters had rank eight. Transfer keeps eight slots: zero-pad smaller ranks and retain the largest eight singular values at larger ranks. Norms still include the entire spectrum. This preserves the original rank-eight representation exactly. Its changed-rank results test this particular extension, not every possible spectral classifier.

## Recipe transfer

Each row contains 21 adapters: three seed/data groups, seven variants representing six families. The baseline recipe is rank 8, alpha 16, learning rate 0.0002, dropout 0 and three epochs.

| Recipe | Broader spectral | Cosine |
| --- | ---: | ---: |
| Fresh baseline controls | 19/21 | 21/21 |
| Rank 4, alpha 8 | 3/21 | 21/21 |
| Rank 16, alpha 32 | 5/21 | 21/21 |
| Learning rate 0.0001 | 15/21 | 21/21 |
| Learning rate 0.0004 | 17/21 | 21/21 |
| Alpha 32, rank 8 | 9/21 | 21/21 |
| Dropout 0.1 | 19/21 | 21/21 |
| Combined: rank 16, alpha 32, LR 0.0001, dropout 0.05, four epochs | 6/21 | 21/21 |
| Total | 93/168 | 168/168 |

The spectral transfer comparison was retrospective: cosine results on this bank were already known. The original spectral fit remained unchanged. The repeated seed groups and shared generator mean the 168 adapters are not 168 independent draws from real-world adapter use.

## Norm experiment

The first nine adapters use a broader scope: 168 matrices, all seven targeted linear projections across 24 layers, rank 16 and alpha 32. Do not compare those global norms numerically with the later 24-matrix family features as though they had the same scope.

The scaled copies multiply B by 0.75 or 0.5; A and the stored alpha/rank scaling are unchanged. They share a training lineage with the original and are not independent retraining runs. Behavior observations include four confidential-summary cases and other conditions in `results/norm-study/scaling-comparison.json`.
