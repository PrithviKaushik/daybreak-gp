import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Solar Power Data Exploration

    Exploratory analysis of the SOLETE Pombo hourly dataset: structural inspection, temporal validation, and target audit.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loading

    The dataset is `SOLETE_Pombo_60min_v4.parquet`, recorded hourly at the DTU Risø campus, Denmark. Timestamps are in UTC.

    The hourly SOLETE Pombo dataset is loaded from the Parquet file. No filtering or transformation is applied during loading.
    """)
    return


@app.cell
def _(pd):
    from daybreak_gp.paths import SOLETE_PARQUET

    df = pd.read_parquet(SOLETE_PARQUET)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structural Inspection

    Review dataset dimensions, schema, data types, and missing-value counts before any temporal or target-specific analysis.
    """)
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.tail()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.columns.tolist()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structural Observations

    - 10,969 hourly records with a timestamp column and candidate solar power columns.
    - Raw and cleaned solar targets contain no missing values.

    ### Key Columns

    | Column | What it represents |
    |---|---|
    | `P_Solar[kW]` | Actual measured solar power |
    | `P_Solar_clean[kW]` | Solar power after the dataset's cleaning process |
    | `Pac` | Solar power estimated by a physical PV model |
    | `P_Solar[kW]_qc` | Flag indicating whether a measurement was replaced by the model |
    | `Elevation[deg]` | Sun's elevation above or below the horizon, calculated using timestamp and location |

    `P_Solar[kW]` contains measured values. `Pac` and any model-based replacements in `P_Solar_clean[kW]` are model-generated.

    ### Missing Values

    - `HUMIDITY[%]`: 24 missing values.
    - `Pressure[mbar]`: 10,956 missing values.

    Humidity values are stored between 0 and 1, despite `%` appearing in the column name. The missing values are associated with a faulty sensor on 2018-11-17.

    Pressure readings are valid only on 2019-01-16. Other readings were placeholders and were replaced with missing values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal Validation

    Verify timestamp uniqueness, chronological ordering, sampling cadence, and continuity of the hourly series.
    """)
    return


@app.cell
def _(df):
    df["timestamp"].duplicated().sum()
    return


@app.cell
def _(df):
    time_diffs = df["timestamp"].diff()
    time_diffs.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temporal Coverage and Continuity

    - Start: 2018-06-01 00:00 UTC
    - End: 2019-09-01 00:00 UTC
    - Total rows: 10,969
    - Complete hourly observations: 10,968
    - Final partial observation: 2019-09-01 00:00:00 UTC, containing only one second
    - Sampling frequency: Hourly
    - Duplicate timestamps: None
    - Chronological ordering: Confirmed
    - Timestamps are unique and have one-hour differences.

    The final row is not a complete hourly observation and must be excluded from modelling and window selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target Audit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the raw and cleaned solar power series, quantify their differences, and audit nighttime values and QC flags before selecting a prediction target.

    ### Candidate Targets

    Identify the candidate solar power columns and compare their distributions.
    """)
    return


@app.cell
def _(df):
    df.columns.tolist()
    return


@app.cell
def _(df):
    df[["P_Solar[kW]", "P_Solar_clean[kW]"]].describe()
    return


@app.cell
def _(df):
    target = df[["P_Solar[kW]", "P_Solar_clean[kW]"]]
    return (target,)


@app.cell
def _(df, plt, target):
    plt.figure(figsize=(16, 5))
    plt.plot(df["timestamp"], target)

    plt.xlabel("Timestamp")
    plt.ylabel("Power (kW)")
    plt.title("Solar Power: Raw vs Cleaned")
    plt.legend(target.columns)
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Raw vs. Cleaned Series

    `P_Solar[kW]` contains measured solar power and is never overwritten. `P_Solar_clean[kW]` starts with measured power. It replaces measured power with the physical model's estimate (`Pac`) when the substitution condition is met. It also sets values less than or equal to 0.001 kW to zero. A `P_Solar[kW]_qc` value of 6 identifies measurements replaced by the physical model.

    The analysis below compares the two series and inspects the largest absolute differences.

    The audit found:

    - 739 observations where raw and cleaned power differ.
    - All 739 differences have raw power greater than cleaned power.
    - The maximum difference is approximately 0.000997 kW.
    - The sum of `P_Solar[kW]_qc` is zero.

    There are zero model-replaced observations in this hourly dataset. The 739 differences are explained by the cleaning process setting very small readings to zero, not by model replacement.

    Project target decision:

    - Primary target: `P_Solar[kW]`, because we want to investigate whether non-stationary GPs model the actual measured solar generation signal better.
    - Sensitivity target: `P_Solar_clean[kW]`, to check whether the cleaning threshold changes our results.

    This is an experimental design decision based on data provenance, not a conclusion established by this EDA.
    """)
    return


@app.cell
def _(df):
    diff = df["P_Solar[kW]"] - df["P_Solar_clean[kW]"]
    return (diff,)


@app.cell
def _(diff):
    print((diff!=0).sum())
    return


@app.cell
def _(diff):
    diff.describe()
    return


@app.cell
def _(diff):
    diff.abs().mean()
    return


@app.cell
def _(diff):
    diff.abs().max()
    return


@app.cell
def _(df, diff):
    df.loc[diff.nlargest(10).index, 
           ["timestamp", "P_Solar[kW]", "P_Solar_clean[kW]", "Elevation[deg]"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nighttime Generation

    `Elevation[deg]` is calculated using the timestamp and location. Hourly elevation is the average elevation over the hour, so `Elevation[deg] < 0` means the average solar elevation for that hour was below the horizon. It does not necessarily mean the sun was below the horizon during every second of that hour.

    The checks below inspect non-zero solar power readings during negative average elevation. These readings are not automatically errors, and their validity has not been established.
    """)
    return


@app.cell
def _(df):
    night_mask = df["Elevation[deg]"]  < 0
    night_df = df[night_mask]
    return (night_df,)


@app.cell
def _(night_df):
    ((night_df["P_Solar[kW]"] != 0).sum() / len(night_df)) * 100
    return


@app.cell
def _(night_df):
    night_nonzero = night_df.loc[
        night_df["P_Solar[kW]"] != 0,
        "P_Solar[kW]"
    ]

    night_nonzero.describe()
    return (night_nonzero,)


@app.cell
def _(df, night_nonzero):
    df.loc[
        night_nonzero.nlargest(10).index,
        ["timestamp", "P_Solar[kW]", "Elevation[deg]"]
    ]
    return


@app.cell
def _(df, night_nonzero):
    df.loc[
        night_nonzero.nlargest(10).index,
        ["timestamp", "P_Solar[kW]", "P_Solar_clean[kW]",
         "P_Solar[kW]_qc"]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### QC Flags and Missingness

    Check whether the QC column identifies modified observations and confirm target zero/missingness counts.
    """)
    return


@app.cell
def _(df):
    df["P_Solar[kW]_qc"].sum()
    return


@app.cell
def _(df):
    df[["P_Solar[kW]", "P_Solar_clean[kW]"]].describe()
    return


@app.cell
def _(df):
    df[["P_Solar[kW]", "P_Solar_clean[kW]"]].isna().sum()
    return


@app.cell
def _(df):
    (df[["P_Solar[kW]", "P_Solar_clean[kW]"]] == 0).sum()
    return


@app.cell
def _(df):
    (df["P_Solar[kW]"] != df["P_Solar_clean[kW]"]).sum()
    return


@app.cell
def _(df):
    df[(df["P_Solar[kW]"] != df["P_Solar_clean[kW]"]) & (df["P_Solar_clean[kW]"] == 0)].shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target Audit Findings

    ### Dataset overview

    The dataset contains 10,969 rows spanning 1 June 2018 to 1 September 2019. Of these, 10,968 are complete hourly observations; the final row at `2019-09-01 00:00:00 UTC` is a partial observation containing only one second.

    | Property | Result |
    |---|---:|
    | Total rows | 10,969 |
    | Complete hourly observations | 10,968 |
    | Final partial observation | 2019-09-01 00:00:00 UTC (one second) |
    | Missing timestamps | 0 |
    | Duplicate timestamps | 0 |
    | Missing hourly intervals | 0 |
    | Missing values in `P_Solar[kW]` | 0 |
    | Missing values in `P_Solar_clean[kW]` | 0 |

    ### Raw vs. cleaned target

    The two candidate target columns were compared:

    - `P_Solar[kW]`: Measured solar power.
    - `P_Solar_clean[kW]`: Solar power after the dataset's cleaning process.

    | Statistic | Raw | Cleaned |
    |---|---:|---:|
    | Mean (kW) | 1.024377 | 1.024361 |
    | Standard deviation (kW) | 1.649652 | 1.649662 |
    | Minimum (kW) | 0.000000 | 0.000000 |
    | Median (kW) | 0.077609 | 0.077609 |
    | Maximum (kW) | 6.969382 | 6.969382 |
    | Zero-valued observations | 3,468 | 4,207 |

    ### Differences between the two series

    - 739 observations have different raw and cleaned values.
    - The mean absolute difference across all observations is approximately `0.00001579 kW`.
    - The maximum absolute difference is approximately `0.000997 kW`.
    - All 739 differences are in the direction of raw power being greater than cleaned power.
    - All 739 differing observations have a cleaned value of exactly zero, verified using a compound Boolean condition.

    The cleaned series sets 739 small positive readings to zero. These are not model replacements.

    ### QC observations

    The column `P_Solar[kW]_qc` contains zero for all 10,969 observations, so its sum is zero. A value of `6` would identify a measurement replaced by the physical model, but no such values occur in this hourly dataset.

    ### Target decision

    This is an experimental design decision based on data provenance, not a result established by this EDA.

    - Primary target: `P_Solar[kW]`, because we want to investigate whether non-stationary GPs model the actual measured solar generation signal better.
    - Sensitivity target: `P_Solar_clean[kW]`, to check whether the cleaning threshold changes our results.

    The dataset audit supports this choice:

    - No measurements were replaced by the physical model in this hourly dataset.
    - The 739 changed values are due to the zeroing threshold, not model replacement.
    - The final partial hourly observation at `2019-09-01 00:00:00 UTC` must be excluded from modelling and window selection.
    """)
    return


if __name__ == "__main__":
    app.run()
