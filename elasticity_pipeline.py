"""
Price & Promo Elasticity Pipeline for CPG Weekly Store-UPC Data
==================================================================
Input schema (case-insensitive column names expected):
WEEK_END_DATE, STORE_ID, UPC, UNITS, VISITS, HHS, SPEND, PRICE, BASE_PRICE,
FEATURE, DISPLAY, TPR_ONLY, DESCRIPTION, MANUFACTURER, CATEGORY, SUB_CATEGORY,
PRODUCT_SIZE, STORE_NAME, ADDRESS_CITY_NAME, ADDRESS_STATE_PROV_CODE, MSA_CODE,
SEG_VALUE_NAME, PARKING_SPACE_QTY, SALES_AREA_SIZE_NUM, AVG_WEEKLY_BASKETS

Usage:
    python elasticity_pipeline.py --input /path/to/your_real_data.csv
    (or .xlsx — both are handled)

Outputs (written to ./outputs/):
    - elasticity_by_category.csv       : price elasticity + promo lift per category
    - elasticity_by_upc.csv            : price elasticity + promo lift per upc
    - baseline_vs_incremental.csv      : weekly base/incremental unit decomposition
    - optimal_price_recommendations.csv: suggested optimal price per upc
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

REQUIRED_COLS_LOWER = [
    "week_end_date", "store_id", "upc", "units", "visits", "hhs", "spend",
    "price", "base_price", "feature", "display", "tpr_only", "description",
    "manufacturer", "category", "sub_category", "product_size", "store_name",
    "address_city_name", "address_state_prov_code", "msa_code",
    "seg_value_name", "parking_space_qty", "sales_area_size_num",
    "avg_weekly_baskets",
]


# ---------------------------------------------------------------------------
# 1. LOAD + VALIDATE (Converts all columns to lower-case internally)
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Convert columns to lowercase for internal processing
    df.rename(columns=lambda x: str(x).strip().lower(), inplace=True)

    missing = set(REQUIRED_COLS_LOWER) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df["week_end_date"] = pd.to_datetime(df["week_end_date"])

    numeric_cols = [
        "units", "visits", "hhs", "spend", "price", "base_price",
        "feature", "display", "tpr_only", "parking_space_qty",
        "sales_area_size_num", "avg_weekly_baskets",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["units", "price", "base_price"])
    df = df[(df["units"] >= 0) & (df["price"] > 0) & (df["base_price"] > 0)]
    dropped = before - len(df)
    if dropped:
        print(f"[load_data] Dropped {dropped} rows with missing/invalid price or units.")

    print("UPC level uniqueness...")
    print("Number of weeks per upc:", df.groupby('upc').agg({'week_end_date': 'count'}).head())
    print(f'Number of unique stores: {df["store_id"].nunique()}')
    print(f'Year range for the data: {df["week_end_date"].min().year}-{df["week_end_date"].max().year}')
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Promo flags -> clean binary ints
    for c in ["feature", "display", "tpr_only"]:
        df[c] = (df[c] > 0).astype(int)

    df["discount_depth"] = (df["base_price"] - df["price"]) / df["base_price"]
    df["discount_depth"] = df["discount_depth"].clip(lower=0)

    df["log_price"] = np.log(df["price"])
    df["log_base_price"] = np.log(df["base_price"])
    df["log_units"] = np.log(df["units"] + 1)

    df["feature_x_display"] = df["feature"] * df["display"]

    df["any_promo"] = ((df["feature"] + df["display"] + df["tpr_only"]) > 0).astype(int)

    # Relative price vs. category-week average
    cat_week_avg = df.groupby(["category", "week_end_date"])["price"].transform("mean")
    df["rel_price"] = df["price"] / cat_week_avg

    # Units per household
    df["units_per_hh"] = df["units"] / df["hhs"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 3. BASELINE VS INCREMENTAL DECOMPOSITION
# ---------------------------------------------------------------------------
def decompose_baseline_incremental(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    df = df.sort_values("week_end_date")

    for (upc, store), grp in df.groupby(["upc", "store_id"]):
        grp = grp.copy()
        grp["week_num"] = (grp["week_end_date"] - grp["week_end_date"].min()).dt.days // 7

        non_promo = grp[grp["any_promo"] == 0]

        if len(non_promo) >= 5:
            X = sm.add_constant(non_promo["week_num"])
            y = non_promo["units"]
            model = sm.OLS(y, X).fit()
            X_all = sm.add_constant(grp["week_num"])
            grp["baseline_units"] = model.predict(X_all).clip(lower=0)
        else:
            grp["baseline_units"] = grp["units"].mean()

        grp["incremental_units"] = (grp["units"] - grp["baseline_units"]).clip(lower=0)
        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. ELASTICITY MODEL
# ---------------------------------------------------------------------------
def fit_elasticity_model(df: pd.DataFrame, group_col: str = None):
    formula = (
        "log_units ~ log_price + feature + display + tpr_only + feature_x_display "
        "+ C(store_id)"
    )
    df = df.copy()
    df["week_of_year"] = df["week_end_date"].dt.isocalendar().week.astype(int)
    formula = formula + " + C(week_of_year)"

    models = {}
    if group_col is None:
        models["ALL"] = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["store_id"]}
        )
    else:
        for g, grp in df.groupby(group_col):
            if grp["log_price"].nunique() < 3 or len(grp) < 50:
                continue
            try:
                models[g] = smf.ols(formula, data=grp).fit(
                    cov_type="cluster", cov_kwds={"groups": grp["store_id"]}
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
            "group": g,
            "price_elasticity": params.get("log_price", np.nan),
            "price_elasticity_pval": pvals.get("log_price", np.nan),
            "feature_lift_pct": (np.exp(params.get("feature", 0)) - 1) * 100,
            "display_lift_pct": (np.exp(params.get("display", 0)) - 1) * 100,
            "tpr_only_lift_pct": (np.exp(params.get("tpr_only", 0)) - 1) * 100,
            "feature_x_display_lift_pct": (np.exp(params.get("feature_x_display", 0)) - 1) * 100,
            "r_squared": res.rsquared,
            "n_obs": int(res.nobs),
        })
    return pd.DataFrame(rows).sort_values("price_elasticity")


# ---------------------------------------------------------------------------
# 5. VIF CHECK
# ---------------------------------------------------------------------------
def check_vif(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["log_price", "feature", "display", "tpr_only", "feature_x_display"]
    X = sm.add_constant(df[cols].dropna())
    vif = pd.DataFrame({
        "variable": X.columns,
        "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif[vif["variable"] != "const"]


# ---------------------------------------------------------------------------
# 6. OPTIMAL PRICE RECOMMENDATION
# ---------------------------------------------------------------------------
def recommend_optimal_price(df: pd.DataFrame, elasticity_by_upc: pd.DataFrame,
                             margin_pct_assumed: float = 0.30) -> pd.DataFrame:
    latest_price = (
        df.sort_values("week_end_date")
        .groupby("upc")
        .agg(current_base_price=("base_price", "last"),
             description=("description", "last"),
             category=("category", "last"))
        .reset_index()
    )

    merged = latest_price.merge(elasticity_by_upc, left_on="upc", right_on="group", how="inner")
    merged["assumed_cost"] = merged["current_base_price"] * (1 - margin_pct_assumed)

    def solve_price(row):
        e = row["price_elasticity"]
        if e >= -1:
            return np.nan
        return row["assumed_cost"] / (1 + 1 / e)

    merged["optimal_price"] = merged.apply(solve_price, axis=1)
    merged["price_change_pct"] = (
        (merged["optimal_price"] - merged["current_base_price"]) / merged["current_base_price"] * 100
    )
    return merged[[
        "upc", "description", "category", "price_elasticity",
        "current_base_price", "assumed_cost", "optimal_price", "price_change_pct"
    ]].sort_values("price_change_pct")


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
        "week_end_date", "store_id", "upc", "units", "baseline_units",
        "incremental_units", "feature", "display", "tpr_only"
    ]].to_csv(f"{output_dir}/baseline_vs_incremental.csv", index=False)

    print("Fitting elasticity model by category ...")
    models_by_cat = fit_elasticity_model(df, group_col="category")
    summary_cat = summarize_models(models_by_cat)
    summary_cat.to_csv(f"{output_dir}/elasticity_by_category.csv", index=False)
    print(summary_cat)

    print("Fitting elasticity model by upc ...")
    models_by_upc = fit_elasticity_model(df, group_col="upc")
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
        f.write("for store-level demand shifters and seasonality.\n")

    print(f"\nDone. Outputs written to ./{output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to weekly CPG data (csv or xlsx)")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    main(args.input, args.output_dir)