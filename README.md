# Bayesian image classification: D-VBLL

AI-generated implementation of the noise-free, diagonal **discriminative Variational Bayesian Last Layer** from Harrison, Willes and Snoek, ICLR 2024. This is a small-scale Fashion-MNIST study, not a reproduction of the paper's large-model scores.

## Reproduce

Python 3.12 was used. In a fresh environment:

```bash
pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B src/experiment.py prepare
python -B src/experiment.py train --threads 6
python -B src/experiment.py evaluate --threads 6
python -B src/analyze.py
```

The official dataset is downloaded from its authors' GitHub mirror. The split is 54,000 train / 6,000 validation / 10,000 final test, with fixed stratification. Three CNN seeds (11, 22, 33) produce paired frozen features for MAP and VBLL classifiers. All checkpoint/prior selection and temperature fitting use validation data only. `train` writes a configuration freeze manifest; `evaluate` refuses changed frozen inputs or an existing final-result file.

An interrupted run can be retried with identical inputs; completed seeds are reused only when their configuration/source fingerprint agrees. Changing training settings requires a separate checkout/run directory, not overwriting the original experiment.

## What is compared

- MAP classifier with a zero-mean Gaussian prior.
- The selected MAP model with validation-fitted temperature scaling.
- D-VBLL posterior-predictive classification (128 antithetic logit samples).
- The same trained D-VBLL model with posterior-mean prediction only.

Metrics are accuracy, hard-label NLL, multiclass Brier (sum over classes), and 15-bin ECE. The three seeds are separate runs, **not an ensemble**. Equal tuning budgets compare complete procedures; the posterior-mean row isolates prediction-time integration only.

## Files

- `src/heads.py`: Gaussian KL, deterministic Jensen training bound and stochastic predictive averaging.
- `src/network.py`: small deterministic feature CNN.
- `src/experiment.py`: disjoint data handling, train/selection/freeze/test stages.
- `src/metrics.py`: validated probability metrics and clipping diagnostics.
- `src/analyze.py`: verifies and plots saved results without retraining.
- `tests/`: small numerical and metric tests.
- `docs/PROTOCOL.md`: protocol fixed before test evaluation.
- `results/`: compact measured outcomes and individual predictions (when published).

Training data, model checkpoints and the coursework report are not included in the public repository. Runtime output is created under `output/runs`; caches remain under `temp`.

To verify the published probabilities **without training or downloading images**, run:

```bash
python -B src/verify_results.py
```

## Measured outcome

Across the three runs, D-VBLL test accuracy was **89.703 ± 0.191%**, NLL **0.28742 ± 0.00855**, Brier **0.14826 ± 0.00345**, and ECE **1.516 ± 0.247%** (sample SD). MAP accuracy was effectively identical. D-VBLL's probability improvements over MAP were small; validation-fitted temperature scaling gave the lowest mean NLL (0.28506) and ECE (1.031%). These are measured small-scale results, not evidence of universal Bayesian superiority.

## References

- Harrison, J., Willes, J., & Snoek, J. (2024). [Variational Bayesian Last Layers](https://arxiv.org/abs/2404.11599), ICLR. [Official implementation](https://github.com/VectorInstitute/vbll).
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML.

Training is sampling-free; D-VBLL classification prediction still uses inexpensive last-layer Monte Carlo sampling. The CNN feature weights are not Bayesian. No general superiority, out-of-distribution robustness or official similarity-check score is claimed.
