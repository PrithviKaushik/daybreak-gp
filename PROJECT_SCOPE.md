# Daybreak-GP
## Project Scope & Experimental Design

**Version:** 1.1  
**Last Updated:** 7 November 2026  
**Project Deadline:** 14 November 2026  
**Dataset:** SOLETE  
**Primary Framework:** GPyTorch

---

## 1. Project Overview

Daybreak-GP investigates whether non-stationary Gaussian Process (GP) kernels offer meaningful advantages over stationary kernels when modelling hourly solar photovoltaic (PV) generation.

The project focuses on the assumption of stationarity in Gaussian Processes: whether the relationship between observations remains consistent across time or whether allowing the covariance structure to adapt locally produces better predictions and uncertainty estimates.

The primary objective is to conduct a controlled comparison between stationary and non-stationary covariance functions, evaluate their behaviour under different temporal conditions, and build a reproducible ML experimentation and inference pipeline.

The project also serves as an opportunity to develop practical ML engineering skills through mathematical modelling, implementation, testing, experimentation and deployment.

## 2. Research Question

### Primary Research Question

Does a non-stationary Gaussian Process kernel improve predictive accuracy and uncertainty estimation compared with a stationary Gaussian Process when modelling hourly solar PV generation?

### Secondary Questions

1. Does adaptive smoothness improve modelling during rapidly changing generation periods?
2. Does non-stationarity provide different benefits across different times of day and seasons?
3. Do improvements remain when accounting for daily periodicity and changing observation noise?
4. Does the additional flexibility justify the computational and implementation complexity?

## 3. Research Hypotheses

The following hypotheses will be investigated:

- **H1:** Non-stationary kernels can better capture locally varying smoothness than stationary kernels.
- **H2:** Improvements may be more pronounced during periods of rapidly changing solar generation.
- **H3:** Non-stationary kernels may provide better predictive uncertainty estimates.
- **H4:** The additional flexibility of non-stationary kernels may not consistently justify their complexity.

These are hypotheses to be tested, not expected results. A finding that non-stationary kernels provide no meaningful improvement is equally valid.

## 4. Dataset and Data Decisions

### 4.1 Dataset

**Dataset:** SOLETE Pombo hourly solar PV generation dataset.

| Property                     | Finding                        |
| ---------------------------- | ------------------------------ |
| Observations                 | 10,969 original rows           |
| Complete hourly observations | 10,968                         |
| Features                     | 37 columns                     |
| Coverage                     | 1 June 2018 – 1 September 2019 |
| Sampling frequency           | Hourly                         |
| Timezone                     | UTC                            |
| Duplicate timestamps         | None                           |
| Missing hourly intervals     | None detected                  |

The final observation at `2019-09-01 00:00 UTC` contains only one second and must be excluded from modelling and evaluation-window selection.

### 4.2 Target Selection

Based on the completed exploratory analysis:

- **Primary target:** `P_Solar[kW]` — measured solar power.
- **Sensitivity target:** `P_Solar_clean[kW]` — solar power after the dataset's cleaning process.

The audit identified 739 observations where the raw and cleaned values differ. All differences result from small positive readings being set to zero. No model-substituted observations were identified in the hourly dataset.

The primary target is selected because the research investigates modelling measured solar generation. The cleaned target may be used for a sensitivity analysis if time permits.

### 4.3 Model Inputs

The primary GP comparison will use timestamp-based inputs.

Weather, irradiance and solar geometry variables will initially serve as contextual information rather than automatically becoming model inputs.

This keeps the primary comparison focused on the covariance structure rather than introducing additional explanatory variables.

## 5. Experimental Design

### 5.1 Primary Comparison

The central experiment compares:

1. **Stationary Gaussian Process:** RBF kernel with a global length scale.
2. **Non-stationary Gaussian Process:** One-dimensional Gibbs kernel with an input-dependent length scale.

The two models should use equivalent training observations, timestamp information, mean-function assumptions and observation-noise assumptions wherever possible.

The principal variable being investigated is the covariance structure.

### 5.2 Evaluation Scenarios

Two evaluation scenarios will be considered separately.

**A. Interpolation — Primary Scenario**

- Evaluate predictions over contiguous held-out intervals of 24 and 48 hours.
- Retain training observations both before and after the held-out interval.
- Use fixed, reproducible gap positions.

Purpose: Investigate how well the models reconstruct missing portions of a temporal signal.

**B. Future Extrapolation — Secondary Scenario**

- Train on historical observations.
- Predict subsequent 24-hour and 48-hour periods.
- Do not make future observations available during training.

Purpose: Investigate how well the models generalise to unseen future timestamps.

Interpolation and extrapolation results must be reported separately.

### 5.3 Seasonal Evaluation

Two temporally distinct seasonal windows, approximately three weeks each, will be selected.

Windows must be chosen before examining model performance.

Selection criteria:

- Data completeness.
- Missing-hour and substituted-hour counts.
- Daylight, ramp and nighttime coverage.
- Sufficient observations for meaningful evaluation.

The same evaluation protocol will be applied to both windows.

If time becomes critically constrained, the second seasonal window may be dropped rather than compromising the primary experiment.

### 5.4 Reproducibility

- Three independent optimisation seeds for GP experiments.
- Consistent evaluation splits across comparable models.
- Identical test timestamps for comparable models.
- Recorded hyperparameters and optimisation settings.
- Fixed and recorded temporal split definitions.
- Configuration-driven experiment execution.
- Results stored in structured CSV files.
- No random shuffling of temporal observations.
- Preprocessing fitted using training observations only.

## 6. Models and Kernel Implementation

### 6.1 Primary Models

| Model                   | Purpose                      | Priority                                   |
| ----------------------- | ---------------------------- | ------------------------------------------ |
| Stationary RBF GP       | Primary stationary reference | Mandatory                                  |
| Non-stationary Gibbs GP | Primary experimental model   | Mandatory, subject to stability checkpoint |

### 6.2 Supporting Models

| Model                                  | Purpose                                          | Priority             |
| -------------------------------------- | ------------------------------------------------ | -------------------- |
| RBF × Periodic GP                      | Investigate daily periodic structure             | Optional             |
| Gibbs × Periodic GP                    | Investigate periodicity with adaptive smoothness | Optional             |
| Heteroscedastic GP diagnostic/baseline | Investigate changing observation noise           | Mandatory diagnostic |
| Time-only XGBoost                      | External ML benchmark                            | Optional             |
| Matérn GP                              | Alternative stationary smoothness assumption     | Deferred             |
| Lag-based XGBoost                      | Practical forecasting reference                  | Deferred             |

The primary comparison remains stationary RBF versus non-stationary Gibbs.

Supporting models must not delay the primary comparison.

The heteroscedastic diagnostic is retained because changing observation noise could otherwise be confused with changing local smoothness.

### 6.3 Deep Gaussian Processes

Deep Gaussian Processes (DeepGPs) are outside the current implementation scope and are deferred to future work.

DeepGPs represent a different model family rather than simply another covariance kernel. A proper implementation would introduce additional architectural, inference and computational requirements.

If the professor explicitly confirms that DeepGP is mandatory:

- Conduct at most one limited reference experiment.
- Use one seasonal window.
- Use interpolation only.
- Clearly label the experiment as a limited reference run.
- Reduce FastAPI implementation to its bare minimum.

Otherwise, DeepGP remains future work.

### 6.4 Gibbs Kernel

The non-stationary model will use the one-dimensional Gibbs covariance function based on Paciorek and Schervish (2003).

The input-dependent length scale will be parameterised using a small neural network:

\[
\ell(t)=\operatorname{softplus}(g_\theta(t))+\epsilon
\]

The implementation will be developed using GPyTorch.

The local length-scale function will be learned alongside the GP parameters through gradient-based optimisation of the GP marginal likelihood.

### 6.5 Required Validation Tests

The Gibbs implementation must be tested for:

1. Constant length scale reproduces the corresponding RBF covariance.
2. Covariance matrix symmetry.
3. Positive semidefiniteness within numerical tolerance.
4. Numerical stability and absence of NaNs or infinities.
5. Correct gradient propagation.
6. Correct behaviour for training and test inputs.

### 6.6 Synthetic Experiment

A synthetic signal containing regions of different smoothness will be used to verify that the non-stationary kernel can learn locally varying length scales.

The experiment should demonstrate that the learned length scale responds to changes between smooth and rapidly varying regions.

This is a controlled validation of the adaptive-smoothness mechanism, not evidence that Gibbs will necessarily outperform RBF on solar data.

### 6.7 Gibbs Stability Checkpoints

**Checkpoint 1 — 9 November 2026**

The implementation must pass its mathematical and numerical tests and train successfully on the synthetic experiment.

If these conditions are not met, the implementation must not proceed to full solar experiments without qualification.

**Checkpoint 2 — 10 November 2026**

Evaluate Gibbs on solar data and make an explicit keep-or-freeze decision.

If training remains unstable:

- Stop further kernel debugging.
- Preserve implementation, logs and experiment results.
- Document attempted solutions and failure modes.
- Report the limitation honestly.

No further kernel debugging will be undertaken after 10 November.

## 7. Evaluation Metrics

### 7.1 Predictive Accuracy

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### 7.2 Probabilistic Evaluation

- Negative Log Predictive Density (NLPD)
- 95% prediction interval coverage
- Mean prediction interval width

Probabilistic metrics will only be reported for models with valid predictive distributions or uncertainty estimates.

### 7.3 Additional Analysis

- Nighttime performance.
- Sunrise and sunset ramp behaviour.
- Daylight and midday performance.
- Predictive calibration.
- Learned Gibbs length-scale behaviour.
- Training time and computational cost.
- Negative prediction counts.

Solar-elevation categories will be defined as:

| Category | Solar elevation |
| -------- | --------------- |
| Night    | ≤ 0°            |
| Ramp     | 0°–15°          |
| Daylight | > 15°           |

Metrics will be reported overall and by category where sufficient observations are available.

Negative GP predictions must be recorded and discussed. They must not be silently clipped before evaluating the original predictive distribution.

## 8. ML Engineering Requirements

The project will be developed as a modular and reproducible Python application.

### 8.1 Engineering Requirements

- Modular code organisation.
- Reusable preprocessing and evaluation components.
- Automated unit tests for important mathematical and temporal properties.
- Configuration-driven experiments.
- Reproducible model training.
- Saved model and configuration artifacts.
- Separation of training and inference.
- Structured experiment results.
- Documented reproduction instructions.

### 8.2 Proposed Architecture

```text
daybreak-gp/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── tests/
├── scripts/
├── api/
├── artifacts/
├── requirements.txt
├── project_scope.md
└── README.md
```

The architecture is a guide, not a requirement to create every directory or abstraction. Implementation should remain as simple as possible while preserving clarity and reproducibility.

### 8.3 Deployment

A lightweight FastAPI service will demonstrate inference using a saved model.

The service should:

- Load saved model and configuration artifacts.
- Accept prediction inputs.
- Return a forecast.
- Avoid retraining during inference.

Docker containerisation is optional and will only be attempted if the core experimental work is completed on schedule.

A frontend, dashboard, cloud deployment or elaborate serving architecture is not required.

## 9. Technology Stack

| Area               | Technology            |
| ------------------ | --------------------- |
| Language           | Python                |
| Data processing    | NumPy, Pandas         |
| Classical ML       | Scikit-learn, XGBoost |
| Deep learning      | PyTorch               |
| Gaussian Processes | GPyTorch              |
| Visualisation      | Matplotlib            |
| API                | FastAPI               |
| Containerisation   | Docker                |
| Testing            | Pytest                |
| Version control    | Git                   |

Optional technologies will only be used when they serve an explicit project requirement.

## 10. Project Timeline

**Hard deadline: 14 November 2026**

| Date       | Work                                                                                                          | Exit condition                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Nov 7**  | Lock target and seasonal windows. Build interpolation splits, metrics and leakage tests. Derive Gibbs kernel. | Windows and target documented; splits and metrics pass toy tests. |
| **Nov 8**  | RBF GP through evaluation harness. Add extrapolation splits. Attempt RBF × Periodic if feasible.              | RBF produces predictions and probabilistic metrics.               |
| **Nov 9**  | Implement Gibbs, unit tests and synthetic experiment.                                                         | Checkpoint 1: tests pass and synthetic training works.            |
| **Nov 10** | Gibbs on solar data, Gibbs × Periodic and heteroscedastic diagnostic.                                         | Checkpoint 2: keep-or-freeze decision.                            |
| **Nov 11** | Build experiment runner and launch controlled experiments.                                                    | One command produces structured results.                          |
| **Nov 12** | Finish experiments and analyse results. Generate figures and document limitations.                            | Written findings and visualisations.                              |
| **Nov 13** | Save artifacts, implement minimal FastAPI and write README.                                                   | Model loads and returns a forecast over HTTP.                     |
| **Nov 14** | Fresh-clone verification and release.                                                                         | Repository reproduces from scratch.                               |

The timeline is an execution plan, not a reason to compromise experimental validity.

## 11. Scope Reduction Policy

If the project falls behind, reduce scope in the following order:

1. Matérn GP.
2. Lag-based XGBoost.
3. Docker.
4. Time-only XGBoost.
5. DeepGP, subject to professor confirmation.
6. Second seasonal window.

The following must not be sacrificed:

- Leakage testing.
- Primary stationary RBF baseline.
- Gibbs mathematical validation.
- Synthetic Gibbs experiment.
- Heteroscedastic diagnostic.
- Honest analysis of predictive accuracy and uncertainty.

If the deadline makes every mandatory component impossible, document the incomplete component rather than presenting an unverified result.

## 12. Final Deliverables

By 14 November 2026, the project aims to deliver:

1. Audited and documented SOLETE dataset.
2. Leakage-safe temporal evaluation framework.
3. Stationary RBF Gaussian Process implementation.
4. Non-stationary Gibbs Gaussian Process implementation, or documented implementation limitations.
5. Synthetic experiment demonstrating adaptive smoothness.
6. Controlled comparative experimental results.
7. Predictive accuracy and uncertainty analysis.
8. Documented findings and limitations.
9. Reproducible ML training and evaluation workflow.
10. Technical README with methodology and reproduction instructions.
11. Minimal FastAPI inference demonstration, subject to time.

Docker and optional supporting models are not release requirements.

## 13. Out of Scope

The following are deliberately excluded from the current project:

- Wind power forecasting.
- CNN, LSTM and Transformer architectures.
- Deep Gaussian Processes, unless explicitly required.
- Extensive Bayesian hyperparameter optimisation.
- Large hyperparameter searches.
- Kubernetes and serverless deployment.
- Cloud infrastructure.
- Frontend and dashboard development.
- Extensive theoretical investigation of alternative GP kernel families.

These may be considered as future extensions.

## 14. Definition of Success

Daybreak-GP will be considered successful if it produces a scientifically defensible comparison between stationary and non-stationary Gaussian Process kernels, supported by:

- Appropriate temporal validation.
- Mathematically and numerically validated implementations.
- Reproducible experiments.
- Predictive accuracy and uncertainty analysis.
- Honest interpretation of results.
- Documented limitations.
- An understandable and reproducible engineering workflow.

A finding that non-stationary kernels provide no meaningful improvement is an equally valid outcome.

The project's success will be measured by the quality of the experimental methodology, implementation, engineering practices and conclusions—not by whether the non-stationary model wins.

## 15. Development Principles

Daybreak-GP will be developed through deliberate, AI-assisted engineering rather than vibe-coding.

The development process will follow these principles:

- Understand the mathematical or engineering problem before implementation.
- Design interfaces and assumptions before writing substantial code.
- Implement incrementally.
- Write tests that verify actual properties.
- Understand and review AI-assisted suggestions rather than blindly accepting generated code.
- Avoid unnecessary abstractions and premature optimisation.
- Preserve reproducibility and the ability to explain every major implementation decision.

The objective is to develop a working project while building genuine understanding of Gaussian Processes, kernel implementation, temporal evaluation and ML engineering.

---

## Project Principle

The purpose of Daybreak-GP is not to prove that non-stationary Gaussian Processes are superior.

It is to understand when their additional modelling flexibility is useful, when it is unnecessary, and how to investigate that question through a controlled, reproducible ML engineering workflow.