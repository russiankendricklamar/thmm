# Regime-Switching Risk in a Bond Portfolio

Multivariate Hidden Markov Model with Student's t emissions for MOEX bond portfolio returns.

This repository presents an independent research note on regime-dependent tail risk in a bond portfolio using a multivariate Student's t Hidden Markov Model (t-HMM). The goal is to improve tail-risk measurement relative to Gaussian benchmarks while preserving an interpretable state-space view of the market.

## Overview

The analysis is built on daily total returns for seven MOEX bond-market segments: federal loan bonds across three duration buckets, corporate bonds across three listing tiers, and mortgage-backed securities. The sample covers 904 trading days from 2021-07-29 to 2026-07-17, with total return defined as clean-price change plus coupon carry.

The model uses a latent Markov chain for regime dynamics and multivariate Student's t emissions for conditional returns. Parameters are estimated via ECM with a Gaussian-HMM warm start, and model selection across the number of regimes is based on BIC.

## Main findings

- BIC selects a 4-regime specification, and likelihood-ratio tests reject Gaussian emissions in favor of Student's t emissions for all tested regime counts with p-values below 0.001.
- The four regimes are economically interpretable: quiet heavy-tail periods, sell-off episodes, sustained rallies, and high-volatility recoveries.
- In a strict rolling out-of-sample one-day-ahead backtest, the Gaussian benchmark is rejected at the 99% VaR level, with realized breach frequency of 2.48% versus a nominal 1%.
- The t-HMM remains correctly calibrated on both VaR and Expected Shortfall in the same out-of-sample test design.
- Relative to EVT-GPD, GARCH-t, and DCC-GARCH, the t-HMM is competitive in tail accuracy while additionally providing interpretable regime probabilities and regime-conditional risk decomposition.
- A reduced-form transition logit identifies the CBR key rate as the main macro driver of entry into stress regimes, with an out-of-sample early-warning signal AUC of 0.88.

## Data and assumptions

| Item | Specification |
|---|---|
| Source | Moscow Exchange ISS public API |
| Frequency | Daily |
| Sample | 904 trading days, 2021-07-29 to 2026-07-17 |
| Portfolio construction | Equal-weighted across 7 segments |
| Return definition | Clean-price return plus smoothed daily coupon carry |
| Universe | 124 liquid RUB bonds mapped into 7 segments |
| Exclusions | Eurobonds and issues without usable MOEX daily history |
| Outlier rule | Daily moves above 15% removed as erroneous quotes |

The reported risk figures are based on equal segment weights, so absolute VaR, ES, and segment contributions will change under actual portfolio weights [file:1]. Coupon income is smoothed from annual coupon rates rather than modeled at exact payment dates, which improves robustness at daily frequency but simplifies cash-flow timing.

## Methodology

Let \(x_t \in \mathbb{R}^7\) denote the vector of segment returns and \(s_t \in \{1, \dots, K\}\) the latent regime [file:1]. Conditional on regime \(k\), returns follow a multivariate Student's t distribution:

\[
x_t \mid s_t = k \sim t_{\nu_k}(\mu_k, \Sigma_k)
\]

Small values of \(\nu_k\) indicate heavier tails, while large values approach the Gaussian case. Estimation uses ECM and the Gaussian scale-mixture representation of the multivariate Student's t distribution, which improves robustness to outliers in parameter updates.

Validation is performed in a rolling one-day-ahead backtest starting after the first 500 observations, with model re-estimation every 5 trading days and 403 out-of-sample forecasts. VaR coverage is tested using Kupiec and Christoffersen tests, and Expected Shortfall is assessed with the Acerbi-Szekely backtest.

## Regime summary

| Regime | Interpretation | \(\nu\) | Ann. vol. | Drift (bp/day) | Expected duration |
|---|---|---:|---:|---:|---:|
| R1 | Quiet, heavy tails | 2.88 | 2.96% | +1.0 | 4.0 days |
| R2 | Sell-off | 5.19 | 3.80% | -1.2 | 5.5 days |
| R3 | Sustained rally | 14.18 | 4.01% | +3.0 | 38.8 days |
| R4 | High volatility | 7.94 | 6.18% | +4.8 | 17.5 days |

An important empirical result is that the most volatile regime is not necessarily the only stress regime: the model separates a negative-drift sell-off state from a positive-drift high-volatility recovery state. This distinction is useful for risk interpretation, hedging logic, and portfolio monitoring.

Raw market data are not distributed in this repository. Reproduction should begin with MOEX ISS data retrieval, universe mapping, total-return construction, and then model estimation and backtesting under the assumptions documented above.

## Practical value

The main practical implication is that Gaussian assumptions materially understate far-tail risk for this portfolio at the 99% level, both for VaR and Expected Shortfall. The t-HMM addresses this while also producing a regime map, transition dynamics, and segment-level Euler decomposition that are directly useful for market-risk monitoring and stress interpretation.

The regime-conditional decomposition shows that the source of portfolio tail risk changes across states, which an unconditional metric can conceal. In particular, credit-sensitive segments such as lower-tier corporate bonds and mortgage-backed securities account for a large share of tail risk in several states.

## Limitations

- The reported decomposition uses equal segment weights rather than an institution-specific live portfolio.
- Eurobonds are excluded because they are not available in the MOEX history used in the study.
- The macro-transition analysis is reduced-form and estimated on a fixed decoded path rather than through full non-homogeneous HMM estimation.
- Degrees of freedom in near-Gaussian regimes are weakly identified in bootstrap intervals, so the robust conclusion is qualitative rather than precise for high-\(\nu\) states.

## License

- Code: MIT License.
- Research note in `paper/`: recommended separate license, such as CC BY 4.0 or a custom rights notice.
- Market data: not redistributed; obtain from MOEX ISS under the relevant source terms.

## Author

**Egor Galkin**

Quantitative research in market risk, fixed income analytics, structured products, and portfolio modeling. This repository contains independent research and does not represent the views of any employer, regulator, exchange, or affiliated institution.
