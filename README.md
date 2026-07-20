# Student-t HMM for Bank Bond Portfolio Regimes

A reproducible research pipeline for modeling regime-dependent risk in a bank bond portfolio using a multivariate Hidden Markov Model with Student's t-distributed emissions.

The project uses daily MOEX bond-market data to identify latent market regimes, estimate regime-conditional VaR and Expected Shortfall, validate forecasts out of sample, decompose risk across portfolio segments, and analyse macro drivers of stress-regime transitions.

## Research question

Can a multivariate regime-switching model with heavy-tailed Student's t emissions provide better tail-risk calibration than a standard Gaussian approximation while retaining an economically interpretable view of market states and risk concentration?

## Repository structure

```text
├── README.md                  # This file
├── LICENSE                    # MIT License for code
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment for reproducibility
│
├── code/                      # Numbered research pipeline
│   ├── 01_total_return_panel.py   # MOEX ISS retrieval, total returns, segment panel
│   ├── 02_macro_panel.py          # Macro variables: policy rate, inflation, USD/RUB
│   ├── 03_thmm_model.py           # Student-t HMM estimation and BIC selection
│   ├── 04_regime_summary.py       # Regime statistics and conditional VaR/ES
│   ├── 05_backtest_hmm.py         # Rolling out-of-sample t-HMM VaR/ES backtest
│   ├── 06_backtest_bench.py       # EVT-GPD and GARCH-t benchmark models
│   ├── 07_es_backtest.py          # Acerbi-Szekely Expected Shortfall backtest
│   ├── 08_var_decomposition.py    # Euler VaR/ES risk attribution by segment
│   ├── 09_bootstrap_ci.py         # Stationary block-bootstrap confidence intervals
│   ├── 10_backtest_dcc.py         # DCC-GARCH multivariate benchmark
│   └── 11_macro_transitions.py    # Macro-dependent transitions and early warning
│
├── data/                      # Input and intermediate datasets
├── model/                     # Fitted Student-t HMM model artifacts
│   └── thmm_model_tr.npz
│
└── overleaf/
    └── overleaf_thmm_bonds.zip    # Paper source: LaTeX project and figures
```

## Quick start

### Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate thmm
```

Alternatively, install dependencies with pip:

```bash
pip install -r requirements.txt
```

### Paper

The research paper source is provided in:

```text
overleaf/overleaf_thmm_bonds.zip
```

Unzip the archive and upload it to Overleaf:

```text
New Project -> Upload Project
```

Use the `pdfLaTeX` compiler.

### Pipeline execution

Scripts are numbered in the intended order of execution:

```text
01 -> 02 -> 03 -> ... -> 11
```

Each script contains a header specifying its objective, inputs, outputs, and assumptions.

> **Path note:** Some scripts were extracted from an experiment lineage and may contain path placeholders of the form `{{artifact:<id>}}`. Replace these placeholders with local paths to the appropriate files in `data/`. The output filename of one stage is generally the input filename for the next stage.

Scripts `01_total_return_panel.py` and `02_macro_panel.py` require network access when raw data are downloaded again from MOEX ISS, BIS, or the World Bank. Prepared datasets in `data/` can be used when a full data refresh is not required.

## Data

| Item | Specification |
|---|---|
| Primary market-data source | Moscow Exchange ISS public API |
| Macro sources | BIS, World Bank, MOEX |
| Frequency | Daily |
| Sample period | 2021-07-29 to 2026-07-17 |
| Number of trading days | 904 |
| Portfolio universe | 124 liquid RUB-denominated bond issues |
| Original portfolio | 178 ISINs |
| Segments | 7 |
| Portfolio construction | Equal weight across segments |
| Return measure | Clean-price return plus daily coupon carry |

The seven portfolio segments are:

- Short federal loan bonds: maturity below 3 years
- Medium federal loan bonds: maturity from 3 to 7 years
- Long federal loan bonds: maturity of 7 years or more
- Corporate bonds: MOEX Listing Level 1
- Corporate bonds: MOEX Listing Level 2
- Corporate bonds: MOEX Listing Level 3
- Mortgage-backed securities

Eurobonds are excluded because they are not consistently available in the MOEX daily history used in this study.

## Methodology

The model is a multivariate Hidden Markov Model with four latent regimes.

For each trading day, the model observes a vector of returns across the seven portfolio segments. The latent regime follows a Markov chain with an estimated transition matrix. Conditional on each regime, returns follow a multivariate Student's t distribution.

Student's t emissions are used instead of Gaussian emissions to account for heavy tails and extreme observations. The degrees-of-freedom parameter is regime-specific: lower values indicate heavier tails.

Model estimation uses the ECM algorithm with:

- Gaussian HMM warm start
- Standardized input returns
- Student's t Gaussian scale-mixture representation
- Lower bound for degrees of freedom of 2.5
- BIC-based selection of the number of regimes

The selected specification contains four regimes.

## Regime summary

| Regime | Interpretation | Degrees of freedom | Annualized volatility | Daily drift | Expected duration | Stationary share |
|---|---|---:|---:|---:|---:|---:|
| R1 | Quiet, heavy tails | 2.88 | 2.96% | +1.0 bp | 4.0 days | 14.4% |
| R2 | Sell-off | 5.19 | 3.80% | -1.2 bp | 5.5 days | 20.5% |
| R3 | Sustained rally | 14.18 | 4.01% | +3.0 bp | 38.8 days | 43.1% |
| R4 | High volatility | 7.94 | 6.18% | +4.8 bp | 17.5 days | 22.1% |

The model distinguishes a negative-drift sell-off regime from a positive-drift high-volatility recovery regime. This distinction is obscured by unconditional volatility measures.

## Main results

- BIC selects four regimes.
- Likelihood-ratio tests reject Gaussian emissions in favor of Student's t emissions for all tested regime counts, with p-values below 0.001.
- The unconditional Gaussian approximation fails at the 99% VaR level in the rolling out-of-sample backtest.
- The Gaussian model records 10 VaR breaches over 403 forecasts, or 2.48%, against a nominal 1% level.
- The Gaussian VaR model is rejected by the Kupiec test (p = 0.012) and the Christoffersen conditional-coverage test (p = 0.034).
- The Student-t HMM records 2 VaR breaches, or 0.50%, and is not rejected by either test.
- Gaussian Expected Shortfall statistically understates tail losses at the 99% level under the Acerbi-Szekely backtest (p = 0.043).
- Student-t HMM Expected Shortfall is not rejected in the same test.
- EVT-GPD, GARCH-t, and DCC-GARCH provide comparable far-tail coverage but do not provide an interpretable regime structure.
- The 99% risk decomposition is concentrated in credit-sensitive segments, notably Listing Level 3 corporate bonds and mortgage-backed securities.
- The CBR key-rate level is the only statistically significant macro driver of entry into stress regimes in the reduced-form transition model.
- The one-day-ahead stress signal achieves an out-of-sample AUC of 0.88.

## Backtesting design

The out-of-sample backtest uses a rolling one-day-ahead forecast design:

| Parameter | Specification |
|---|---|
| Initial estimation window | 500 observations |
| Re-estimation frequency | Every 5 trading days |
| Out-of-sample forecasts | 403 |
| VaR confidence levels | 95% and 99% |
| VaR tests | Kupiec unconditional coverage; Christoffersen conditional coverage |
| ES test | Acerbi-Szekely Test 2 |
| Benchmark models | Gaussian, historical simulation, EVT-GPD, GARCH-t, DCC-GARCH |

## Risk attribution

Portfolio VaR and Expected Shortfall are decomposed using Euler allocation.

The unconditional 99% VaR is concentrated in:

| Segment | VaR contribution |
|---|---:|
| Corporate bonds, Listing Level 3 | 28.3% |
| Mortgage-backed securities | 23.6% |
| Corporate bonds, Listing Level 1 | 14.9% |
| Corporate bonds, Listing Level 2 | 10.2% |
| Long federal loan bonds | 10.0% |
| Short federal loan bonds | 8.7% |
| Medium federal loan bonds | 4.2% |

Risk concentration changes materially across regimes. For example, in quiet heavy-tail periods, tail risk is concentrated in lower-tier corporate bonds; during sell-offs, the contribution shifts toward mortgage-backed securities and top-tier corporate bonds.

## Limitations

- The portfolio is equal-weighted across segments; live portfolio weights will change absolute VaR, Expected Shortfall, and segment contributions.
- Coupon income is approximated through smoothed daily carry rather than exact coupon-payment dates and reinvestment cash flows.
- Eurobonds are excluded from the analysis because of incomplete MOEX historical coverage.
- The macro-transition model is reduced-form and is estimated on a fixed decoded regime path rather than jointly within a non-homogeneous HMM.
- Degrees-of-freedom estimates for near-Gaussian regimes are weakly identified at high values; the robust conclusion concerns the distinction between heavy-tailed and near-Gaussian states rather than a precise high degrees-of-freedom estimate.

## Reproducibility

The repository includes the full research pipeline, prepared data, model artifacts, and the Overleaf source for the paper.

To reproduce results:

1. Create the environment.
2. Set local paths in the scripts.
3. Run the numbered scripts in sequence.
4. Compare generated outputs with the saved artifacts in `data/`, `model/`, and `overleaf/`.

## Citation

If you use this repository, please cite the research note:

```text
Galkin, E. (2026). Hidden Markov Model with Student's t Emissions
for Regimes of a Bank Bond Portfolio Return.
```

## License

The source code in this repository is released under the MIT License.

The research paper, its figures, and associated written materials are provided for reading and citation. They are not covered by the MIT License unless explicitly stated otherwise.

## Author

**Egor Galkin**

Quantitative research in market risk, fixed income, derivatives, structured products, and portfolio analytics.

This repository contains independent research and does not represent the views of any employer, regulator, exchange, or affiliated institution.
