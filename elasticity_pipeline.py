"""
Price & Promo Elasticity Pipeline for CPG Weekly Store-UPC Data
==================================================================
Input schema (identical column names expected):
WEEK_END_DATE, STORE_ID, UPC, UNITS, VISITS, HHS, SPEND, PRICE, BASE_PRICE,
FEATURE, DISPLAY, TPR_ONLY, DESCRIPTION, MANUFACTURER, CATEGORY, SUB_CATEGORY,
PRODUCT_SIZE, STORE_NAME, ADDRESS_CITY_NAME, ADDRESS_STATE_PROV_CODE, MSA_CODE,
SEG_VALUE_NAME, PARKING_SPACE_QTY, SALES_AREA_SIZE_NUM, AVG_WEEKLY_BASKETS

Usage:
    python elasticity_pipeline.py --input /path/to/your_real_data.csv
    (or .xlsx — both are handled)

Outputs (written to ./outputs/):
    - elasticity_by_category.csv       : price elasticity + promo lift per CATEGORY
    - elasticity_by_upc.csv            : price elasticity + promo lift per UPC
    - baseline_vs_incremental.csv      : weekly base/incremental unit decomposition
    - optimal_price_recommendations.csv: suggested optimal price per UPC
    - model_diagnostics.txt            : fit stats, VIFs, warnings
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

REQUIRED_COLS = [
    "WEEK_END_DATE", "STORE_ID", "UPC", "UNITS", "VISITS", "HHS", "SPEND",
    "PRICE", "BASE_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "DESCRIPTION",
    "MANUFACTURER", "CATEGORY", "SUB_CATEGORY", "PRODUCT_SIZE", "STORE_NAME",
    "ADDRESS_CITY_NAME", "ADDRESS_STATE_PROV_CODE", "MSA_CODE",
    "SEG_VALUE_NAME", "PARKING_SPACE_QTY", "SALES_AREA_SIZE_NUM",
    "AVG_WEEKLY_BASKETS",
]


# ---------------------------------------------------------------------------
# 1. LOAD + VALIDATE
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"])

    numeric_cols = [
        "UNITS", "VISITS", "HHS", "SPEND", "PRICE", "BASE_PRICE",
        "FEATURE", "DISPLAY", "TPR_ONLY", "PARKING_SPACE_QTY",
        "SALES_AREA_SIZE_NUM", "AVG_WEEKLY_BASKETS",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["UNITS", "PRICE", "BASE_PRICE"])
    df = df[(df["UNITS"] >= 0) & (df["PRICE"] > 0) & (df["BASE_PRICE"] > 0)]
    dropped = before - len(df)
    if dropped:
        print(f"[load_data] Dropped {dropped} rows with missing/invalid price or units.")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Promo flags -> clean binary ints (guard against non 0/1 values)
    for c in ["FEATURE", "DISPLAY", "TPR_ONLY"]:
        df[c] = (df[c] > 0).astype(int)

    df["DISCOUNT_DEPTH"] = (df["BASE_PRICE"] - df["PRICE"]) / df["BASE_PRICE"]
    df["DISCOUNT_DEPTH"] = df["DISCOUNT_DEPTH"].clip(lower=0)  # no negative "discounts"

    df["LOG_PRICE"] = np.log(df["PRICE"])
    df["LOG_BASE_PRICE"] = np.log(df["BASE_PRICE"])
    df["LOG_UNITS"] = np.log(df["UNITS"] + 1)  # +1 guard for zero-unit weeks

    df["FEATURE_X_DISPLAY"] = df["FEATURE"] * df["DISPLAY"]

    df["ANY_PROMO"] = ((df["FEATURE"] + df["DISPLAY"] + df["TPR_ONLY"]) > 0).astype(int)

    # Relative price vs. category-week average (competitive positioning)
    cat_week_avg = df.groupby(["CATEGORY", "WEEK_END_DATE"])["PRICE"].transform("mean")
    df["REL_PRICE"] = df["PRICE"] / cat_week_avg

    # Units per household (traffic-normalized demand) - useful alt DV
    df["UNITS_PER_HH"] = df["UNITS"] / df["HHS"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 3. BASELINE VS INCREMENTAL DECOMPOSITION
# ---------------------------------------------------------------------------
def decompose_baseline_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each UPC-STORE series, fit a simple baseline (trend + seasonality)
    using only NON-PROMOTED weeks, then predict baseline for all weeks and
    compute incremental = actual - baseline (floored at 0).
    """
    results = []
    df = df.sort_values("WEEK_END_DATE")

    for (upc, store), grp in df.groupby(["UPC", "STORE_ID"]):
        grp = grp.copy()
        grp["WEEK_NUM"] = (grp["WEEK_END_DATE"] - grp["WEEK_END_DATE"].min()).dt.days // 7

        non_promo = grp[grp["ANY_PROMO"] == 0]

        if len(non_promo) >= 5:  # need enough points to fit a trend
            X = sm.add_constant(non_promo["WEEK_NUM"])
            y = non_promo["UNITS"]
            model = sm.OLS(y, X).fit()
            X_all = sm.add_constant(grp["WEEK_NUM"])
            grp["BASELINE_UNITS"] = model.predict(X_all).clip(lower=0)
        else:
            # not enough non-promo weeks -> fall back to overall mean
            grp["BASELINE_UNITS"] = grp["UNITS"].mean()

        grp["INCREMENTAL_UNITS"] = (grp["UNITS"] - grp["BASELINE_UNITS"]).clip(lower=0)
        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# 4. ELASTICITY MODEL (log-log, fixed effects via dummies / demeaning)
# ---------------------------------------------------------------------------
def fit_elasticity_model(df: pd.DataFrame, group_col: str = None):
    """
    Fits: LOG_UNITS ~ LOG_PRICE + FEATURE + DISPLAY + TPR_ONLY + FEATURE_X_DISPLAY
          + C(STORE_ID) + C(WEEK_END_DATE)
    If group_col is provided, fits one model per group value (e.g. per CATEGORY).
    Uses store + week fixed effects to control for store-level demand shifters
    and seasonality/trend, which reduces (but does not fully eliminate) price
    endogeneity bias.
    Returns a dict: group_value -> fitted statsmodels results object
    """
    formula = (
        "LOG_UNITS ~ LOG_PRICE + FEATURE + DISPLAY + TPR_ONLY + FEATURE_X_DISPLAY "
        "+ C(STORE_ID)"
    )
    # Week fixed effects via week-of-year to keep dummy count manageable
    df = df.copy()
    df["WEEK_OF_YEAR"] = df["WEEK_END_DATE"].dt.isocalendar().week.astype(int)
    formula = formula + " + C(WEEK_OF_YEAR)"

    models = {}
    if group_col is None:
        models["ALL"] = smf.ols(formula, data=df).fit(cov_type="cluster",
                                                        cov_kwds={"groups": df["STORE_ID"]})
    else:
        for g, grp in df.groupby(group_col):
            # need variation in price and enough obs
            if grp["LOG_PRICE"].nunique() < 3 or len(grp) < 50:
                continue
            try:
                models[g] = smf.ols(formula, data=grp).fit(
                    cov_type="cluster", cov_kwds={"groups": grp["STORE_ID"]}
                )
            except Exception as e:
                print(f"[fit_elasticity_model] Skipped group {g}: {e}")
    return models


def summarize_models(models: dict) -> pd.DataFrame:
    rows = []
    for g, res in models.items():
        params = res.params
        pvals = res.pvalues
        rows.append({
            "GROUP": g,
            "PRICE_ELASTICITY": params.get("LOG_PRICE", np.nan),
            "PRICE_ELASTICITY_PVAL": pvals.get("LOG_PRICE", np.nan),
            "FEATURE_LIFT_PCT": (np.exp(params.get("FEATURE", 0)) - 1) * 100,
            "DISPLAY_LIFT_PCT": (np.exp(params.get("DISPLAY", 0)) - 1) * 100,
            "TPR_ONLY_LIFT_PCT": (np.exp(params.get("TPR_ONLY", 0)) - 1) * 100,
            "FEATURE_X_DISPLAY_LIFT_PCT": (np.exp(params.get("FEATURE_X_DISPLAY", 0)) - 1) * 100,
            "R_SQUARED": res.rsquared,
            "N_OBS": int(res.nobs),
        })
    return pd.DataFrame(rows).sort_values("PRICE_ELASTICITY")


# ---------------------------------------------------------------------------
# 5. VIF CHECK (multicollinearity between promo mechanics)
# ---------------------------------------------------------------------------
def check_vif(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["LOG_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY", "FEATURE_X_DISPLAY"]
    X = sm.add_constant(df[cols].dropna())
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif[vif["variable"] != "const"]


# ---------------------------------------------------------------------------
# 6. OPTIMAL PRICE RECOMMENDATION
# ---------------------------------------------------------------------------
def recommend_optimal_price(df: pd.DataFrame, elasticity_by_upc: pd.DataFrame,
                             margin_pct_assumed: float = 0.30) -> pd.DataFrame:
    """
    Uses constant-elasticity logic: for a product with elasticity E (E<-1 required
    for an interior max under constant marginal cost), the profit-maximizing
    markup satisfies:  optimal_price = cost / (1 + 1/E)
    Since we don't have COGS in this dataset, we approximate cost as
    current BASE_PRICE * (1 - margin_pct_assumed), and solve for optimal price.
    Treat this as directional guidance, not a final pricing decision --
    real deployment needs actual product cost data.
    """
    latest_price = (
        df.sort_values("WEEK_END_DATE")
        .groupby("UPC")
        .agg(CURRENT_BASE_PRICE=("BASE_PRICE", "last"),
             DESCRIPTION=("DESCRIPTION", "last"),
             CATEGORY=("CATEGORY", "last"))
        .reset_index()
    )

    merged = latest_price.merge(elasticity_by_upc, left_on="UPC", right_on="GROUP", how="inner")
    merged["ASSUMED_COST"] = merged["CURRENT_BASE_PRICE"] * (1 - margin_pct_assumed)

    def solve_price(row):
        e = row["PRICE_ELASTICITY"]
        if e >= -1:  # inelastic or wrong-signed -> markup formula breaks down
            return np.nan
        return row["ASSUMED_COST"] / (1 + 1 / e)

    merged["OPTIMAL_PRICE"] = merged.apply(solve_price, axis=1)
    merged["PRICE_CHANGE_PCT"] = (
        (merged["OPTIMAL_PRICE"] - merged["CURRENT_BASE_PRICE"]) / merged["CURRENT_BASE_PRICE"] * 100
    )
    return merged[[
        "UPC", "DESCRIPTION", "CATEGORY", "PRICE_ELASTICITY",
        "CURRENT_BASE_PRICE", "ASSUMED_COST", "OPTIMAL_PRICE", "PRICE_CHANGE_PCT"
    ]].sort_values("PRICE_CHANGE_PCT")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(input_path: str, output_dir: str = "outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_path} ...")
    df = load_data(input_path)
    print(f"Loaded {len(df):,} rows.")

    print("Engineering features ...")
    df = engineer_features(df)

    print("Decomposing baseline vs incremental units ...")
    df_decomp = decompose_baseline_incremental(df)
    df_decomp[[
        "WEEK_END_DATE", "STORE_ID", "UPC", "UNITS", "BASELINE_UNITS",
        "INCREMENTAL_UNITS", "FEATURE", "DISPLAY", "TPR_ONLY"
    ]].to_csv(f"{output_dir}/baseline_vs_incremental.csv", index=False)

    print("Fitting elasticity model by CATEGORY ...")
    models_by_cat = fit_elasticity_model(df, group_col="CATEGORY")
    summary_cat = summarize_models(models_by_cat)
    summary_cat.to_csv(f"{output_dir}/elasticity_by_category.csv", index=False)
    print(summary_cat)

    print("Fitting elasticity model by UPC ...")
    models_by_upc = fit_elasticity_model(df, group_col="UPC")
    summary_upc = summarize_models(models_by_upc)
    summary_upc.to_csv(f"{output_dir}/elasticity_by_upc.csv", index=False)

    print("Checking multicollinearity (VIF) ...")
    vif = check_vif(df)

    print("Computing optimal price recommendations ...")
    opt_price = recommend_optimal_price(df, summary_upc)
    opt_price.to_csv(f"{output_dir}/optimal_price_recommendations.csv", index=False)

    with open(f"{output_dir}/model_diagnostics.txt", "w") as f:
        f.write("PRICE/PROMO ELASTICITY MODEL DIAGNOSTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total rows modeled: {len(df):,}\n")
        f.write(f"Categories modeled: {len(models_by_cat)}\n")
        f.write(f"UPCs modeled: {len(models_by_upc)}\n\n")
        f.write("VIF (multicollinearity check, >5-10 = concerning):\n")
        f.write(vif.to_string(index=False))
        f.write("\n\nNOTE: Store + week-of-year fixed effects are included to control\n")
        f.write("for store-level demand shifters and seasonality. Price endogeneity\n")
        f.write("(prices cut because demand was expected to move) is only partially\n")
        f.write("addressed by fixed effects -- for a production model, consider an\n")
        f.write("instrumental-variables approach (e.g. cost shocks, wholesale price\n")
        f.write("changes) as a price instrument.\n")

    print(f"\nDone. Outputs written to ./{output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to weekly CPG data (csv or xlsx)")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    main(args.input, args.output_dir)
