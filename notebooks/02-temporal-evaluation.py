import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Temporal Evaluation Dataset and Seasonal Windows

    This notebook prepares the hourly SOLETE solar PV dataset for temporal evaluation of Gaussian Process models. The research objective is to test whether a non-stationary GP improves hourly PV generation forecasting compared with a stationary GP.

    This stage constructs the temporal dataset and identifies candidate seasonal evaluation windows. Model implementation, hyperparameter selection, and predictive evaluation are handled later; they are not performed here.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Loading

    The hourly SOLETE Parquet file is loaded without filtering or transformation. This notebook relies on four columns:

    | Column | Role |
    |---|---|
    | `timestamp` | UTC hourly timestamp; used for ordering, resampling, and window construction |
    | `P_Solar[kW]` | Measured solar power; the primary target identified in the EDA |
    | `Elevation[deg]` | Solar elevation; used to classify solar regimes |
    | `P_Solar[kW]_qc` | QC flag; code 6 marks a measurement replaced by the physical model |

    The series is recorded at the DTU Risø campus, Denmark, and timestamps are in UTC.
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
    ## Removing the Partial Final Observation

    The raw file ends at `2019-09-01 00:00 UTC` with a row that covers only one second rather than a complete hour. It is a partial record, not a valid hourly observation.

    Including it would give the model a timestamp whose sampling interval differs from every other observation. A partial hour cannot be treated as a regular hourly sample, and interpolating it would introduce a value that was not measured over the same interval. The row is therefore dropped before temporal analysis or window selection.

    This leaves 10,968 complete hourly observations.
    """)
    return


@app.cell
def _(df):
    df_ts = df.iloc[:-1].copy()
    return (df_ts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal Data Validation

    The retained series is checked for the basic properties required by chronological windowing.

    The first assertion confirms that exactly one row was removed. The second confirms that `timestamp` is monotonically increasing. Timestamp uniqueness was verified during the preceding dataset EDA. The assertion below also checks uniqueness directly.
    """)
    return


@app.cell
def _(df, df_ts):
    assert len(df_ts) == len(df) - 1
    assert df_ts["timestamp"].is_monotonic_increasing
    assert df_ts["timestamp"].is_unique
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tail Inspection

    The tail inspection displays the last retained rows for `timestamp`, `P_Solar[kW]`, and `Elevation[deg]`. It confirms that the final retained timestamp precedes the removed partial observation and that the selected columns are available for later steps.
    """)
    return


@app.cell
def _(df_ts):
    df_ts[["timestamp", "P_Solar[kW]", "Elevation[deg]"]].tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monthly Resampling

    Resampling to month-end and taking the mean provides a coarse diagnostic of monthly coverage and aggregate behaviour. It is not a formal continuity test and does not replace the timestamp checks above.
    """)
    return


@app.cell
def _(df_ts):
    df_ts.resample("ME", on="timestamp").mean()
    return


@app.cell
def _(df_ts):
    df_ts["Elevation[deg]"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solar Regime Classification

    Solar elevation is a physically meaningful indicator of whether the sun is above the horizon and how strong the incident solar geometry is. Classifying observations by elevation lets the evaluation separate nighttime, low-sun ramp, and full daylight conditions.

    | Regime | Elevation condition |
    |---|---|
    | Night | $\text{elevation} \le 0^\circ$ |
    | Ramp | $0^\circ < \text{elevation} \le 15^\circ$ |
    | Daylight | $\text{elevation} > 15^\circ$ |

    The resulting `solar_period` column is an evaluation-stratification label. It is not used as a model input feature in this notebook. The $15^\circ$ boundary is an operational analysis threshold chosen for this study, not a universal physical boundary between ramp and daylight conditions.
    """)
    return


@app.cell
def _(df_ts, pd):
    bins = [float("-inf"), 0, 15, float("inf")]
    df_ts["solar_period"] = pd.cut(df_ts["Elevation[deg]"], bins=bins, labels=["night", "ramp", "daylight"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regime Distribution and Mean Power

    The regime counts summarize how many observations fall into each class. The mean measured power by regime is a descriptive check that the labels separate low- and high-generation conditions; it is not a model result.
    """)
    return


@app.cell
def _(df_ts):
    df_ts["solar_period"].value_counts()
    return


@app.cell
def _(df_ts):
    df_ts.groupby("solar_period")["P_Solar[kW]"].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monthly Seasonal Analysis

    The monthly cross-tabulation counts night, ramp, and daylight observations in each calendar month. This reveals how daylight availability changes across the year and provides a basis for choosing contrasting seasonal candidate periods.

    July 2018 and February 2019 are selected as candidate summer and winter months. Both are complete months within the dataset record, and their solar-regime compositions provide a summer/winter contrast for evaluation. They are candidate study periods, not claims that they are representative of every summer or winter.
    """)
    return


@app.cell
def _(df_ts, pd):
    monthly_regimes = pd.crosstab(df_ts["timestamp"].dt.to_period("M"), df_ts["solar_period"])
    return (monthly_regimes,)


@app.cell
def _(monthly_regimes):
    monthly_regimes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### July 2018 Regime Totals

    The July 2018 column totals from the monthly table are inspected directly. This reports the summer candidate's regime mix before the notebook constructs windows from it. February 2019 is taken as the winter counterpart from the same monthly table.
    """)
    return


@app.cell
def _(monthly_regimes):
    monthly_regimes.loc["2018-07"].sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating Candidate Seasonal Windows

    Each candidate is a 21-day window of hourly observations. A 21-day span contains multiple weeks of weather and diurnal cycles while remaining short enough to stay within a single seasonal period.

    For July 2018, the 31-day month admits $31 - 21 + 1 = 11$ start dates. For February 2019, the 28-day month admits $28 - 21 + 1 = 8$ start dates. Start dates advance by one day, so consecutive windows overlap by 20 days. This shifting explores different daylight compositions rather than selecting a single fixed block.

    Windows use the half-open interval $[\text{start}, \text{end})$. The start timestamp is included and the end timestamp is excluded. With 24 hourly observations per day:

    $$
    21 \text{ days} \times 24 \text{ observations/day} = 504 \text{ observations}
    $$

    Each generated window is a candidate. No window is formally selected at this stage.
    """)
    return


@app.cell
def _(pd):
    summer_date_range = pd.date_range("2018-07-1", periods=11, freq="D", tz="UTC")
    summer_date_range
    return (summer_date_range,)


@app.cell
def _(pd, summer_date_range):
    summer_ends = []
    summer_window_duration = pd.Timedelta(days=21)

    for summer_date in summer_date_range:
        summer_ends.append(summer_date + summer_window_duration)

    print(summer_ends)
    return (summer_ends,)


@app.cell
def _(df_ts, summer_date_range, summer_ends):
    summer_date_pairs = zip(summer_date_range, summer_ends)
    summer_windows = []
    for s_start, s_end in summer_date_pairs:
        print(s_start, s_end)
        summer_windows.append(df_ts[(df_ts["timestamp"] >= s_start) & (df_ts["timestamp"] < s_end)])
    return (summer_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summer Candidate Window Inspection

    The first summer window is displayed to verify the slice includes the expected columns and spans the intended dates. Its regime counts are then inspected as a spot check before the full audit.
    """)
    return


@app.cell
def _(summer_windows):
    summer_windows[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solar-Regime Audits

    For every candidate window, the notebook counts observations in each solar regime. These counts make the seasonal daylight representation explicit and allow candidate windows to be compared as the start date shifts by one day.

    The audit does not rank or select windows; it records the regime composition needed for the later selection rule.
    """)
    return


@app.cell
def _(summer_windows):
    summer_windows[0]["solar_period"].value_counts()
    return


@app.cell
def _(summer_windows):
    summer_solar_period_audit = [window["solar_period"].value_counts() for window in summer_windows]
    summer_solar_period_audit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Window Quality Checks

    Each candidate window is inspected against three properties:

    - Observation count: a complete 21-day hourly window should contain 504 observations.
    - QC substitutions: the number of observations with `P_Solar[kW]_qc == 6` should be zero, because code 6 marks a measurement replaced by the physical model.
    - Solar-regime distribution: the night, ramp, and daylight counts show the window's daylight representation for comparison across windows.

    The per-window quality loops print these values for inspection; the final verification section asserts the aggregate conditions. The dataset-level EDA found no missing hourly timestamps and no QC substitutions, so the expected outcome is 504 observations and zero QC substitutions per window.
    """)
    return


@app.cell
def _(summer_windows):
    for _window in summer_windows:
        print(
            len(_window),
            _window["P_Solar[kW]_qc"].eq(6).sum(),
            _window["solar_period"].value_counts().to_dict()
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating Winter Candidate Windows

    Winter candidates use the same 21-day construction. February 2019 has 28 days, so eight start dates are possible: 1 to 8 February. Each window uses the half-open interval $[\text{start}, \text{end})$, contains 504 hourly observations, and shifts by one day to vary the daylight composition.
    """)
    return


@app.cell
def _(pd):
    winter_date_range = pd.date_range("2019-02-1", periods=8, freq="D", tz="UTC")
    return (winter_date_range,)


@app.cell
def _(pd, winter_date_range):
    winter_ends = []
    winter_window_duration = pd.Timedelta(days=21)

    for winter_date in winter_date_range:
        winter_ends.append(winter_date + winter_window_duration)
    return (winter_ends,)


@app.cell
def _(df_ts, winter_date_range, winter_ends):
    winter_date_pairs = zip(winter_date_range, winter_ends)
    winter_windows = []
    for w_start, w_end in winter_date_pairs:
        print(w_start, w_end)
        winter_windows.append(df_ts[(df_ts["timestamp"] >= w_start) & (df_ts["timestamp"] < w_end)])
    return (winter_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observation Count per Window

    Each candidate window spans 21 days at an hourly cadence:

    $$
    21 \text{ days} \times 24 \text{ observations/day} = 504 \text{ observations}
    $$

    The per-window quality audits report the actual observation count alongside the QC and solar-regime counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Winter Solar-Regime Audit

    The winter audit mirrors the summer audit. It records regime counts for each of the eight winter candidates so that their daylight, ramp, and night composition can be compared with the summer candidates.
    """)
    return


@app.cell
def _(winter_windows):
    winter_solar_period_audit = [window["solar_period"].value_counts() for window in winter_windows]
    winter_solar_period_audit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Winter Window Quality Check

    The same three inspections are applied to the winter candidates: observation count, QC-substituted target count, and solar-regime distribution. These are printed for each window rather than asserted.
    """)
    return


@app.cell
def _(winter_windows):
    for window in winter_windows:
        print(
            len(window),
            window["P_Solar[kW]_qc"].eq(6).sum(),
            window["solar_period"].value_counts().to_dict()
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combined Candidate Audit

    The summer and winter candidates are combined into one audit table with 19 rows. For each candidate, the table records its season, half-open window bounds, observation count, missing hourly timestamps, QC-substituted observations, and solar-regime counts.

    `missing_hours` counts expected hourly timestamps in $[\text{start}, \text{end})$ that are absent from the window. `substituted_hours` counts observations with `P_Solar[kW]_qc == 6`.

    The 19 candidate windows overlap heavily. They are alternative candidates for selecting representative evaluation periods, not 19 independent seasonal experiments.
    """)
    return


@app.cell
def _(
    pd,
    summer_date_range,
    summer_ends,
    summer_windows,
    winter_date_range,
    winter_ends,
    winter_windows,
):
    candidate_rows = []

    for audit_season, audit_starts, audit_ends, audit_windows in [
        ("summer", summer_date_range, summer_ends, summer_windows),
        ("winter", winter_date_range, winter_ends, winter_windows),
    ]:
        for audit_start, audit_end, audit_window in zip(audit_starts, audit_ends, audit_windows):
            audit_regime_counts = (
                audit_window["solar_period"]
                .value_counts()
                .reindex(["night", "ramp", "daylight"], fill_value=0)
            )
            audit_expected_timestamps = pd.date_range(
                audit_start, audit_end, freq="h", inclusive="left"
            )

            candidate_rows.append(
                {
                    "season": audit_season,
                    "start": audit_start,
                    "end": audit_end,
                    "observations": len(audit_window),
                    "missing_hours": int(
                        (~audit_expected_timestamps.isin(audit_window["timestamp"])).sum()
                    ),
                    "substituted_hours": int(audit_window["P_Solar[kW]_qc"].eq(6).sum()),
                    "night": int(audit_regime_counts["night"]),
                    "ramp": int(audit_regime_counts["ramp"]),
                    "daylight": int(audit_regime_counts["daylight"]),
                }
            )

    candidate_audit = pd.DataFrame(candidate_rows)
    candidate_audit
    return (candidate_audit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying the Selection Rule

    Candidate generation, quality inspection, and median-based selection are now implemented. The rule is applied separately to the summer and winter candidates, so each season is represented by its own median daylight composition.

    1. Exclude candidates with missing hourly observations.
    2. Exclude candidates with QC-substituted target observations (`P_Solar[kW]_qc == 6`).
    3. Compute the median daylight count across the remaining eligible candidates.
    4. Select the candidate whose daylight count is closest to that median.
    5. If two or more candidates are equally close, select the one with the earliest start timestamp.

    The median is used as a robust central reference for the eligible windows' daylight composition. It avoids selecting an extreme window, such as the sunniest or darkest candidate, and keeps the chosen evaluation window representative of the season's typical daylight exposure. Summer and winter are not ranked against each other.

    All windows use the half-open interval $[\text{start}, \text{end})$: the start timestamp is included and the end timestamp is exclusive.
    """)
    return


@app.cell
def _(candidate_audit, pd):
    selected_rows = []

    for selection_season in ["summer", "winter"]:
        selection_eligible = candidate_audit[
            (candidate_audit["season"] == selection_season)
            & (candidate_audit["missing_hours"] == 0)
            & (candidate_audit["substituted_hours"] == 0)
        ].copy()

        if selection_eligible.empty:
            raise ValueError(
                f"No eligible {selection_season} candidates after applying the selection rule."
            )

        selection_median_daylight = selection_eligible["daylight"].median()
        selection_eligible["distance_from_median"] = (
            selection_eligible["daylight"] - selection_median_daylight
        ).abs()
        selection_selected = selection_eligible.sort_values(
            ["distance_from_median", "start"]
        ).iloc[0]

        selected_rows.append(
            {
                "season": selection_season,
                "start": selection_selected["start"],
                "end": selection_selected["end"],
                "observations": selection_selected["observations"],
                "daylight": selection_selected["daylight"],
                "median_daylight": selection_median_daylight,
                "distance_from_median": selection_selected["distance_from_median"],
            }
        )

    selected_windows = pd.DataFrame(selected_rows)
    selected_windows
    return (selected_windows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selected Seasonal Windows

    Applying the rule independently to the 11 summer and 8 winter candidates selects:

    | Season | Start | End (exclusive) | Observations | Daylight | Median daylight | Distance from median |
    |---|---|---|---:|---:|---:|---:|
    | Summer | `2018-07-06 00:00 UTC` | `2018-07-27 00:00 UTC` | 504 | 268 | 268.0 | 0.0 |
    | Winter | `2019-02-04 00:00 UTC` | `2019-02-25 00:00 UTC` | 504 | 97 | 98.5 | 1.5 |

    For summer, the selected window's daylight count equals the median. For winter, the median is 98.5; the windows starting on 4 February and 5 February are both 1.5 away from it. The earliest start, 4 February, is selected by the tie-breaking rule.

    The end timestamp is exclusive: each window includes observations from `start` up to, but not including, `end`. The selected windows are the representative evaluation periods; the 19 candidates from which they were chosen overlap and are not independent experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification

    The final checks validate candidate-window integrity and the seasonal selection rule.

    First, the candidate audit is checked to ensure that:
    - All 19 candidate windows were generated.
    - Each window contains exactly 504 hourly observations.
    - No expected hourly timestamps are missing.
    - No observations have QC substitution flags (code 6).
    - Night, ramp, and daylight counts sum to 504 for every window.

    Next, the selected windows are verified independently. For each season, eligible candidates are reconstructed, the median daylight count is recalculated, and the candidate closest to that median is identified. Ties are resolved by selecting the earliest start timestamp.

    The selected start and end timestamps are compared against these independently calculated results. Assertions raise an error if any expected condition is not satisfied.
    """)
    return


@app.cell
def _(candidate_audit):
    # candidate intergrity
    assert len(candidate_audit) == 19
    assert candidate_audit["observations"].eq(504).all()
    assert candidate_audit["missing_hours"].eq(0).all()
    assert candidate_audit["substituted_hours"].eq(0).all()

    assert (
        candidate_audit[["night", "ramp", "daylight"]]
        .sum(axis=1)
        .eq(504)
        .all()
    )
    return


@app.cell
def _(candidate_audit, selected_windows):
    # selection correctness
    assert selected_windows["season"].value_counts().to_dict() == {
        "summer": 1,
        "winter": 1,
    }

    for season in ["summer", "winter"]:
        eligible = candidate_audit[
            (candidate_audit["season"] == season)
            & (candidate_audit["missing_hours"] == 0)
            & (candidate_audit["substituted_hours"] == 0)
        ]

        median = eligible["daylight"].median()

        expected = (
            eligible.assign(distance=(eligible["daylight"] - median).abs())
            .sort_values(["distance", "start"])
            .iloc[0]
        )

        actual = selected_windows.loc[
            selected_windows["season"] == season
        ].iloc[0]

        assert actual["start"] == expected["start"]
        assert actual["end"] == expected["end"]
    return


if __name__ == "__main__":
    app.run()
