# daybreak-GP v1.0
## Project Scope & Experimental Design

**Version:** 1.0  
**Project deadline:** 14 November 2026  
**Dataset:** SOLETE  
**Primary framework:** GPyTorch

---

## 1. Project Overview

Daybreak-GP investigates whether non-stationary Gaussian Process (GP) kernels offer meaningful advantages over stationary kernels when modelling hourly solar photovoltaic (PV) generation.

The project focuses on the assumption of stationarity in Gaussian Processes: whether the relationship between observations remains consistent across time or whether allowing the covariance structure to adapt locally produces better predictions and uncertainty estimates.

The primary objective is to conduct a controlled comparison between stationary and non-stationary covariance functions, evaluate their behaviour under different temporal conditions, and build a reproducible ML experimentation and inference pipeline.

## 2. Research Question

**Primary Research Question**

Does a non-stationary Gaussian Process kernel improve predictive accuracy and uncertainty estimation compared with stationary Gaussian Process kernels when modelling hourly solar PV generation?

**Secondary Questions**

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

These are hypotheses to be tested, not expected results.

## 4. Dataset

**Dataset:** SOLETE

Initial dataset inspection:

| Property                 | Finding          |
| ------------------------ | ---------------- |
| Observations             | 10,969           |
| Features                 | 37 columns       |
| Start date               | 1 June 2018      |
| End date                 | 1 September 2019 |
| Sampling frequency       | Hourly           |
| Timezone                 | UTC              |
| Duplicate timestamps     | None             |
| Chronological ordering   | Confirmed        |
| Missing hourly intervals | None detected    |

The final target variable, units, quality-control filtering strategy and preprocessing decisions will be established through dataset auditing.

The primary modelling objective is solar PV generation. Weather, irradiance and solar geometry measurements will initially be investigated as contextual variables rather than automatically included as model inputs.

## 5. Experimental Design

### 5.1 Primary Comparison

The central experiment compares:

1. **Stationary Gaussian Process:** RBF kernel with a global length scale.
2. **Non-stationary Gaussian Process:** One-dimensional Gibbs kernel with an input-dependent length scale.

The two models will use equivalent training observations, timestamp information, mean-function assumptions and observation-noise assumptions wherever possible.

The principal variable being investigated is the covariance structure.

### 5.2 Evaluation Scenarios

Two evaluation scenarios will be considered separately.

**A. Interpolation**

Evaluate predictions over contiguous held-out intervals of 24 and 48 hours.

Training observations will be available both before and after the held-out interval.

Purpose: Investigate how well the models reconstruct missing portions of a temporal signal.

**B. Future Extrapolation**

Train on historical observations and predict subsequent 24-hour and 48-hour periods.

No future observations will be available during training.

Purpose: Investigate how well the models generalise to unseen future timestamps.

Interpolation and extrapolation results will be reported separately.

### 5.3 Seasonal Evaluation

Two temporally distinct seasonal windows will be selected from the dataset.

Each window will use the same evaluation protocol.

Exact dates will be selected after data exploration, without choosing windows based on model performance.

### 5.4 Reproducibility

- Three independent initialization seeds for GP experiments.
- Consistent evaluation splits across models.
- Recorded hyperparameters and optimization settings.
- Configuration-driven experiment execution.
- Results stored in structured CSV files.

## 6. Models

### Primary Models

| Model                   | Purpose                      |
| ----------------------- | ---------------------------- |
| Stationary RBF GP       | Primary stationary reference |
| Non-stationary Gibbs GP | Primary experimental model   |

### Supporting Models

| Model                 | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| Periodic GP (24-hour) | Investigate daily periodic structure                  |
| Matérn GP             | Compare alternative stationary smoothness assumptions |
| Heteroscedastic GP    | Investigate changing observation noise                |
| Time-only XGBoost     | External ML benchmark                                 |
| Lag-based XGBoost     | Optional practical forecasting reference              |

The primary comparison remains stationary versus non-stationary Gaussian Processes. Supporting models provide context and help investigate alternative explanations for performance differences.

### Input Fairness

The primary GP comparison will use timestamp-based inputs.

The time-only XGBoost model will receive equivalent temporal information through engineered time features.

The lag-based XGBoost model, if implemented, will be reported separately because it uses additional historical observations.

## 7. Non-Stationary Kernel Implementation

The non-stationary model will use the one-dimensional Gibbs covariance function based on Paciorek and Schervish (2003).

The input-dependent length scale will be parameterised using a small neural network:

\[
\ell(t)=\operatorname{softplus}(g_\theta(t))+\epsilon
\]

The implementation will be developed using GPyTorch.

### Required Validation Tests

1. Constant length scale reproduces the corresponding RBF kernel.
2. Covariance matrix symmetry.
3. Positive semidefiniteness within numerical tolerance.
4. Numerical stability and absence of NaNs or infinities.
5. Correct gradient propagation.
6. Correct behaviour for training and test inputs.

### Synthetic Experiment

A synthetic signal with regions of different smoothness will be used to verify that the non-stationary kernel can learn locally varying length scales.

This experiment will provide a controlled demonstration of the motivation behind non-stationary kernels.

**Kernel Stability Checkpoint: 9 November 2026**

If the implementation cannot be made numerically stable and train reliably by this checkpoint, further development will be frozen. The implementation difficulties, attempted solutions and findings will be documented.

## 8. Evaluation Metrics

### Predictive Accuracy

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Probabilistic Evaluation

- Negative Log Predictive Density (NLPD)
- 95% Prediction Interval Coverage
- Mean Prediction Interval Width

Probabilistic metrics will only be reported for models with valid predictive distributions or uncertainty estimates.

Point-prediction models such as ordinary XGBoost will be evaluated using accuracy metrics unless an appropriate uncertainty estimation method is implemented.

### Additional Analysis

- Daytime versus nighttime performance.
- Sunrise and sunset ramp behaviour.
- Midday performance.
- Clear versus cloudy conditions, if suitable information is available.
- Predictive calibration.
- Learned length-scale behaviour.
- Training time and computational cost.

## 9. ML Engineering Requirements

The project will be developed as a modular and reproducible Python application.

Proposed architecture:

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
├── Dockerfile
├── requirements.txt
├── PROJECT_SCOPE.md
└── README.md
```

Engineering requirements:

- Modular code organisation.
- Reusable preprocessing and evaluation components.
- Automated unit tests.
- Configuration-driven experiments.
- Reproducible model training.
- Saved model and preprocessing artifacts.
- Separation of training and inference.
- Structured experiment results.
- Documented reproduction instructions.

### Deployment

A lightweight FastAPI service will expose the selected trained model for inference.

Docker will be used to containerise the service if the core experimental work is completed on schedule.

The API will load saved artifacts and will not retrain models during inference.

## 10. Technology Stack

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

## 11. Project Timeline

**Hard deadline: 14 November 2026**

| Period    | Milestone                                                        |
| --------- | ---------------------------------------------------------------- |
| 2–3 Nov   | Dataset audit and preparation                                    |
| 4 Nov     | Validation framework and baseline models                         |
| 5–6 Nov   | Stationary GP experiments                                        |
| 7–8 Nov   | Gibbs kernel derivation, implementation and synthetic experiment |
| 9 Nov     | Kernel stability checkpoint                                      |
| 10–11 Nov | Automated experiments and seasonal comparisons                   |
| 12 Nov    | Analysis and findings                                            |
| 13 Nov    | Packaging and deployment                                         |
| 14 Nov    | Documentation, reproducibility checks and release                |

When time becomes constrained, deployment polish and secondary models will be reduced before compromising the primary experiment, temporal validation or analysis.

## 12. Final Deliverables

By 14 November 2026, the project aims to deliver:

1. Audited and documented SOLETE dataset.
2. Leakage-safe temporal evaluation framework.
3. Stationary RBF Gaussian Process implementation.
4. Non-stationary Gibbs Gaussian Process implementation, or documented implementation limitations.
5. Synthetic experiment demonstrating adaptive smoothness.
6. Comparative experimental results.
7. Predictive accuracy and uncertainty analysis.
8. Documented findings and limitations.
9. Reproducible ML training and inference pipeline.
10. Technical README with methodology and reproduction instructions.
11. FastAPI and Docker inference service, subject to the remaining time.

## 13. Out of Scope

The following are deliberately excluded from this project:

- Wind power forecasting.
- CNN, LSTM and Transformer architectures.
- Extensive Bayesian hyperparameter optimisation.
- Large hyperparameter searches.
- Kubernetes and serverless deployment.
- Cloud infrastructure.
- Frontend and dashboard development.
- Extensive theoretical investigation of alternative GP kernel families.

These may be considered as future extensions.

## 14. Definition of Success

Daybreak-GP will be considered successful if it produces a scientifically defensible comparison between stationary and non-stationary Gaussian Process kernels, supported by appropriate temporal validation, reproducible experiments and honest interpretation of results.

A finding that non-stationary kernels provide no meaningful improvement is an equally valid outcome.

The project's success will be measured by the quality of the experimental methodology, implementation, engineering practices and conclusions—not by whether the non-stationary model wins.

---

**Project Principle**

The purpose of Daybreak-GP is not to prove that non-stationary Gaussian Processes are superior. It is to understand when their additional modelling flexibility is useful, when it is unnecessary, and how to investigate that question through a reproducible ML engineering workflow.