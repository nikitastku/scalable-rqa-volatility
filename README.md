# Scalable Recurrence-Based Feature Extraction for Volatility Regime Prediction in Financial Time Series

Bachelor thesis project: Department of Advanced Computing Sciences, Maastricht University.

**Author:** Nikitas Ttikkou
**Supervisors:** Dr. Ir. Martijn Boussé, Dr. Ir. Philippe Dreesen
**Programme:** BSc Data Science and Artificial Intelligence

---

## What this project does

This project tests whether scalable recurrence-based features — Recurrence Quantification Analysis (RQA), the β-RQA extension of Deckert et al. (2025), and the without-RP / sampled-RP variants of Marwan (2026), improve next-period volatility-regime classification on financial time series, beyond what standard rolling features achieve.

The investigation is run on three datasets that span five orders of magnitude in physical bar size: a long synthetic daily series (D1, 1902–2017), a ten-year S&P 500 macro-financial panel (D2, 2010–2019), and an intraday cross-section of 503 S&P 500 stocks sampled at 2-minute resolution (D3).

**Headline finding.** On D3, where statistical power is sufficient, adding RQA features to a Random Forest baseline produces a per-stock paired ΔAUC of **+0.00056** (Wilcoxon signed-rank p < 0.001, n = 503 tickers). The gain is small in absolute terms because the standard rolling features already reach AUC 0.983, but it is statistically robust:

- it survives an aggregation experiment (the gain *grows* to +0.012 when bars are coarsened to 10-min / 30-min);
- it survives a purged walk-forward CV protocol (mean ΔAUC across 5 folds = **+0.00176**, Wilcoxon p = 0.03125);
- it does not appear on D1 (covariate-shifted) or D2 (sample-size limited), which the project documents as predictable consequences of those datasets' structure.

A separate sub-finding from the β-RQA work: the horizontal-line measures introduced by Deckert et al. (2025) are mathematically redundant when β = 2 (LAM ↔ LAM_h correlation = 1.0000) and gain information only as β moves away from 2, confirmed empirically on D3.

---

## Quick start

The repository ships with all pre-computed results, processed splits, and figures committed. To regenerate any specific result, see the per-script commands further down.

```bash
# 1. Clone / unpack the submission
cd scalable-rqa-volatility

# 2. Install the package and its dependencies
pip install -e .

# 3. Open the reproducibility notebook
jupyter notebook reproducibility.ipynb
# Kernel -> Restart & Run All  (takes 1-2 minutes)
```

The notebook walks through every result in the thesis end-to-end. With artifacts present (default), it takes ~1-2 minutes. Regenerating *all* artifacts from raw data would take several hours — the relevant commands are documented in the notebook and in this README.

---

## Repository layout

```
scalable-rqa-volatility/
├── configs/                       YAML configs for the three data pipelines
├── data/
│   ├── raw/                       Raw input data (CSVs + parquet)
│   └── processed/                 Train/val/test parquets + summary CSVs
├── figures/
│   ├── general_figures/           Dataset overviews + feature-importance + scalability plots
│   ├── recurrence_plots/          RP-with-regime visualisations (fig_a, b, c)
│   └── checkpoint_followup/       Aggregation, ACF, walk-forward CV figures (fig_d, e, f)
├── results/                       Text reports + saved predictions (.txt, .npz)
├── scripts/
│   ├── data/                      Data download + preprocessing pipelines
│   ├── train/                     Model training scripts (one per model x dataset)
│   ├── evaluate/                  Statistical tests, feature importance, scalability,
│   │                              aggregation experiment, walk-forward CV
│   ├── sweep/                     PowerShell wrappers for hyperparameter sweeps (β, m, τ, RR)
│   └── visualize/                 Figure-generation scripts
├── src/scalable_rqa_volatility/   Library code (the package installed by `pip install -e .`)
│   ├── data/                      Loaders + chronological splits
│   ├── volatility/                Returns, realized vol, regime labelling
│   ├── recurrence/                Delay embedding, standard RQA, β-RQA
│   ├── models/                    GJR-GARCH, HAR-RV, LSTM, GINN
│   ├── evaluation/                Classification metrics
│   ├── plots/                     Plotting utilities (npz save/load, ROC, confusion)
│   └── utils/                     I/O paths, seeding, logging, dataset selector
├── reproducibility.ipynb          One-stop narrative walk-through of every result
├── pyproject.toml                 Package definition + dependencies
└── README.md                      This file
```

---

## How to reproduce specific results

Every result has a single script that produces it. All scripts run from the repository root.

### Data pipelines

```bash
python scripts/data/run_pipeline.py             # D1  (Core_TimeSeries.csv -> dataset1_*.parquet)
python scripts/data/run_pipeline_dataset2.py    # D2  (S&P 500 macro CSV  -> dataset2_*.parquet)
python scripts/data/run_pipeline_dataset3.py    # D3  (sp500_intraday.parquet -> dataset3_*.parquet)
```

`scripts/data/download_yahoo_sp500_intraday.py` downloads the raw intraday data via the `yfinance` library. It is not re-run by default; the resulting `sp500_intraday.parquet` is shipped in `data/raw/` for this submission.

### Model training and evaluation

Trains all relevant models and runs the paired statistical tests. These are the scripts that produced the numbers in the thesis tables.

```bash
python scripts/evaluate/run_statistical_tests.py --dataset 1
python scripts/evaluate/run_statistical_tests.py --dataset 2
python scripts/evaluate/run_statistical_tests.py --dataset 3
```

Outputs land in `results/statistical_tests_d{1,2,3}.txt` and `results/predictions_d{1,2}.npz`.

### Feature importance and ablations (D3)

```bash
python scripts/evaluate/feature_importance_d3.py              # Gini + permutation + ablation
python scripts/evaluate/feature_importance_beta_sweep_d3.py   # Horizontal-measure analysis across β
```

### Scalable RQA benchmarks

```bash
python scripts/evaluate/benchmark_scalable_rqa.py             # Marwan 2026 woRP / Samp timings
python scripts/evaluate/classify_with_rqa_samp_d3.py          # Classification cost of sampling
```

### Checkpoint follow-up experiments

```bash
python scripts/evaluate/checkpoint_aggregation_and_timescales.py  # Aggregation + ACF
python scripts/evaluate/walk_forward_cv_d3.py                     # Walk-forward CV (50 tickers)
python scripts/evaluate/walk_forward_cv_d3.py --n_tickers 503     # Full panel (~1 hour)
```

### Figures

```bash
python scripts/visualize/generate_all_figures.py    # Dataset overviews + distribution shift
python scripts/visualize/plot_rp_with_regimes.py    # Recurrence-plot panels (fig_a, b, c)
```

---

## Pipeline at a glance

For each dataset:

1. **Load** raw OHLC / log-return data.
2. **Compute** log returns and realized volatility (rolling std, window = 20 daily bars or 60 intraday bars).
3. **Label** each bar with a no-leak regime indicator: regime = 1 if RV ≥ rolling 0.7-quantile of past `lookback` bars, else 0. Lookback = 252 for daily, 975 for intraday.
4. **Split** chronologically 70 / 15 / 15 into train / val / test. D3 does this per ticker, then pools.
5. **Features.** Standard rolling statistics at three windows per dataset. RQA features computed in 60-bar sliding windows with delay embedding (m = 4, τ = 2) and joint multivariate embedding of (log_return, RV).
6. **Train.** Logistic regression, random forest, HAR-RV, GJR-GARCH, LSTM, GARCH-informed LSTM (GINN).
7. **Evaluate.** ROC-AUC + threshold-calibrated F1 (predicted-positive rate constrained to [0.5×base, 1.5×base]).
8. **Test.** Paired bootstrap (D1/D2) or per-stock Wilcoxon signed-rank (D3, n = 503 tickers).

The package code that implements steps 2–6 is in `src/scalable_rqa_volatility/`.

---

## Headline numbers

For verification — these numbers appear identically in the thesis, in `results/*.txt`, and rendered live in the notebook.

| | D1 | D2 | D3 |
|--|--|--|--|
| RF Std AUC | 0.878 | 0.977 | 0.987 (per-stock mean) |
| RF Std+RQA AUC | 0.878 | 0.956 | 0.988 (per-stock mean) |
| Paired ΔAUC | −0.0001 (n.s.) | −0.020 (n.s.) | +0.00056 (p < 0.001) |
| Test method | Bootstrap | Bootstrap | Per-stock Wilcoxon |

Distribution shift (train→test RV ratio): D1 = 0.17×, D2 = 0.97×, D3 = 0.95×.

Walk-forward CV on D3 (5 folds, 50 tickers, purged + embargoed): mean ΔAUC = +0.00176, p = 0.03125.

Aggregation experiment on D3 (50 tickers): ΔAUC = −0.0003 (2-min) / +0.0120 (10-min) / +0.0124 (30-min).

---

## Dependencies

Defined in `pyproject.toml`. Core requirements include:

- Python ≥ 3.12
- numpy, pandas, scipy, scikit-learn
- matplotlib
- pyarrow
- arch (for GJR-GARCH)
- torch (for LSTM / GINN)
- yfinance (only for re-downloading D3 raw data)
- lxml

Install everything with `pip install -e .` from the repository root.

---

## Notes on data

- **D1 (Core_TimeSeries.csv).** Synthetic daily series, 30,000 rows from 1902. Distributed with the project for reproducibility.
- **D2 (S&P 500 macro-financial).** Daily series with cross-market indices, options data, macro indicators, and commodities. Public source; redistributed for reproducibility.
- **D3 (sp500_intraday.parquet).** Sourced from Yahoo Finance via `yfinance`, 2-min bars for 503 S&P 500 stocks over the last ~60 trading days at the time of download. Included in this submission for grading reproducibility only; the underlying data is subject to Yahoo Finance's terms of service.

The processed splits in `data/processed/` are deterministic functions of the raw data plus the configs in `configs/`. Re-running the data pipelines reproduces them exactly.

---

## Citing the methods used

- Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). *Recurrence plots for the analysis of complex systems.* Physics Reports, 438(5–6), 237–329. https://doi.org/10.1016/j.physrep.2006.11.001
- Marwan, N. (2026). *Energy-efficient recurrence quantification analysis.* The European Physical Journal Special Topics. https://doi.org/10.1140/epjs/s11734-025-02121-w  *(introduces the without-RP and sampled-RP methods, RQA_woRP / RQA_Samp)*
- Deckert, E., Dreesen, P., Marwan, N., & Boussé, M. (2025). *Beta-divergence-based recurrence plots for audio time-series analysis.* In Proceedings of the 33rd European Signal Processing Conference (EUSIPCO 2025) (pp. 291–295). https://doi.org/10.23919/EUSIPCO63237.2025.11226301
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (ch. 7, purged walk-forward CV). Hoboken, NJ: John Wiley & Sons.
- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility* (HAR-RV). Journal of Financial Econometrics, 7(2), 174–196. https://doi.org/10.1093/jjfinec/nbp001
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). *On the relation between the expected value and the volatility of the nominal excess return on stocks* (GJR-GARCH). The Journal of Finance, 48(5), 1779–1801. https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
- Takens, F. (1981). *Detecting strange attractors in turbulence.* In Dynamical Systems and Turbulence, Warwick 1980 (Lecture Notes in Mathematics, Vol. 898, pp. 366–381). Springer. https://doi.org/10.1007/BFb0091924  *(delay embedding)*
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735  *(LSTM baseline)*
- Xu, Z., Liechty, J., Benthall, S., Skar-Gislinge, N., & McComb, C. (2024). *GARCH-informed neural networks for volatility prediction in financial markets.* In Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF '24) (pp. 600–607). https://doi.org/10.1145/3677052.3698600  *(GINN baseline)*
- Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324  *(classifier; also the origin of permutation importance)*
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously.* Journal of Machine Learning Research, 20(177), 1–81. https://jmlr.org/papers/v20/18-760.html  *(model-agnostic permutation importance)*
- Eckmann, J.-P., Oliffson Kamphorst, S., & Ruelle, D. (1987). *Recurrence plots of dynamical systems.* Europhysics Letters, 4(9), 973–977. https://doi.org/10.1209/0295-5075/4/9/004  *(recurrence plots)*
- Zbilut, J. P., & Webber, C. L., Jr. (1992). *Embeddings and delays as derived from quantification of recurrence plots.* Physics Letters A, 171(3–4), 199–203. https://doi.org/10.1016/0375-9601(92)90426-M  *(RQA)*
- Wilcoxon, F. (1945). *Individual comparisons by ranking methods.* Biometrics Bulletin, 1(6), 80–83. https://doi.org/10.2307/3001968  *(signed-rank test)*
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html  *(implementation library)*

---

## Licence and acknowledgements

This thesis was conducted under the supervision of Dr. Ir. Martijn Boussé and Dr. Ir. Philippe Dreesen at Maastricht University. Code is provided for the purpose of academic evaluation. Third-party data is subject to the providers' respective terms.

For questions related to this project, contact the author via the Department of Advanced Computing Sciences.
