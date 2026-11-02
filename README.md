# daybreak-GP

**Non-Stationary vs. Stationary Gaussian Processes for Solar PV Forecasting**

Daybreak-GP investigates whether non-stationary Gaussian Process (GP) kernels can better model hourly solar photovoltaic generation than stationary kernels.

The central idea is to investigate whether allowing a model's notion of temporal smoothness to change over time improves predictive accuracy and uncertainty estimation.

This project combines a controlled kernel comparison with a reproducible machine learning experimentation and inference pipeline.

## Research Question

Does a non-stationary Gaussian Process kernel improve predictive accuracy and uncertainty estimation compared with stationary Gaussian Process kernels when modelling hourly solar PV generation?

The project investigates when additional modelling flexibility is useful, when it is unnecessary, and whether its benefits justify its computational complexity.

## Dataset

The project uses the **SOLETE dataset**, containing hourly measurements from the DTU Risø campus in Denmark.

| Property                     | Description              |
| ---------------------------- | ------------------------ |
| Dataset                      | SOLETE                   |
| Location                     | DTU Risø campus, Denmark |
| Sampling frequency           | Hourly                   |
| Timezone                     | UTC                      |
| Coverage                     | June 2018 – August 2019  |
| Complete hourly observations | 10,968                   |
| Primary target               | `P_Solar[kW]`            |

The primary target is measured solar power. `P_Solar_clean[kW]` will be used as a sensitivity target to investigate the effect of the dataset's cleaning procedure.

The dataset is not included in this repository. Place the downloaded `SOLETE_Pombo_60min_v4.parquet` file inside:

```text
data/raw/
```

Dataset provenance and preprocessing decisions are documented in the exploration notebook.

## Experimental Design

The primary experiment compares stationary and non-stationary Gaussian Process covariance functions.

### Core comparison

| Model             | Kernel | Purpose                      |
| ----------------- | ------ | ---------------------------- |
| Stationary GP     | RBF    | Global smoothness assumption |
| Non-stationary GP | Gibbs  | Input-dependent smoothness   |

### Periodicity-aware comparison

| Model             | Kernel                     |
| ----------------- | -------------------------- |
| Stationary GP     | RBF × Periodic (24-hour)   |
| Non-stationary GP | Gibbs × Periodic (24-hour) |

This additional comparison investigates whether the benefits of non-stationarity persist after accounting for daily periodic structure.

### Evaluation scenarios

**1. Interpolation (primary)**

Predict contiguous held-out intervals of 24 and 48 hours, with observations available both before and after each interval.

This evaluates how effectively the models reconstruct missing portions of a temporal signal.

**2. Future extrapolation (secondary)**

Train on historical observations and predict subsequent 24-hour and 48-hour periods without access to future observations.

This evaluates generalisation beyond the observed training period. Results will be reported separately from interpolation.

Two temporally distinct seasonal windows will be evaluated using consistent experimental protocols.

## Non-Stationary Kernel

The project implements the one-dimensional Gibbs covariance function based on the non-stationary covariance framework of Paciorek and Schervish (2003).

The input-dependent length scale is parameterised using a small neural network:

\[
\ell(t)=\operatorname{softplus}(g_\theta(t))+\epsilon
\]

**Implementation distinction:** The original Paciorek and Schervish approach uses Gaussian Process priors over kernel functions and MCMC. This project instead uses a neural network to parameterise the length scale and optimises it through marginal likelihood.

The implementation is developed using GPyTorch.

## Evaluation

Predictive performance will be assessed using:

**Predictive accuracy**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Probabilistic evaluation**
- Negative Log Predictive Density (NLPD)
- 95% prediction interval coverage
- Mean prediction interval width

Additional analysis will investigate:
- Daytime and nighttime performance.
- Sunrise and sunset ramp behaviour.
- Midday performance.
- Seasonal differences.
- Predictive calibration.
- Learned length-scale behaviour.
- Training time and computational cost.

## Supporting Models

Additional models provide context for the primary experiment.

- Heteroscedastic Gaussian Process: investigate changing observation noise.
- Time-only XGBoost: external machine learning benchmark.
- Matérn Gaussian Process: alternative stationary smoothness assumption.
- Lag-based XGBoost: optional practical forecasting reference.

These models are secondary to the core RBF-versus-Gibbs comparison and may be reduced if time becomes constrained.

## Engineering Objectives

Daybreak-GP is also an ML engineering project. The intended workflow includes:

- Modular Python architecture.
- Configuration-driven experiments.
- Automated unit tests.
- Reproducible training and evaluation.
- Saved model and preprocessing artifacts.
- Separation of training and inference.
- Structured experiment results.
- A lightweight FastAPI inference service.

Docker containerisation is planned if time permits.

## Technology

Python · NumPy · Pandas · Scikit-learn · PyTorch · GPyTorch · XGBoost · Matplotlib · FastAPI · Pytest

## Project Structure

The repository structure will evolve as the modelling and engineering components are implemented.

## Current Status

**Version 1.0 · In active development**

- [x] Initial dataset exploration.
- [x] Dataset structure and temporal continuity audit.
- [x] Target provenance audit.
- [ ] Temporal evaluation framework.
- [ ] Stationary RBF GP.
- [ ] Gibbs kernel implementation.
- [ ] Synthetic kernel validation.
- [ ] Periodicity-aware GP comparison.
- [ ] Comparative experiments.
- [ ] Uncertainty evaluation.
- [ ] Reproducible inference pipeline.
- [ ] Final analysis and documentation.

**Target completion:** 14 November 2026.

## References

1. Paciorek, C. J., & Schervish, M. J. (2003). *Nonstationary covariance functions for Gaussian process regression*. Advances in Neural Information Processing Systems (NeurIPS).

2. Pombo, J., et al. *SOLETE dataset*. Refer to the dataset's original publication and distribution for provenance and access details.

## Project Principle

The objective is not to prove that non-stationary Gaussian Processes are superior. It is to understand when their additional modelling flexibility is useful, when it is unnecessary, and how to investigate that question through a reproducible ML engineering workflow.