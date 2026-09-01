import os
import numpy as np
import pandas as pd
import pytest

from elasticity_pipeline import (
    load_data,
    engineer_features,
    decompose_baseline_incremental,
    fit_elasticity_model,
    check_vif,
    recommend_optimal_price,
    summarize_models,
)


def make_sample_df():
    # Create a small but realistic weekly dataset
    weeks = pd.date_range("2021-01-01", periods=12, freq="W-FRI")
    rows = []
    for store in [101, 102]:
        for upc in [1111, 2222]:
            base_price = 3.0 if upc == 1111 else 5.0
            category = "SNACKS" if upc == 1111 else "DRINKS"
            for i, w in enumerate(weeks):
                price = base_price * (1 - 0.01 * ((i % 4)))  # small variation
                base_price_col = base_price
                units = 20 + i * (1 if upc == 1111 else 2) + (5 if i % 5 == 0 else 0)
                feature = 1 if (i % 6 == 0) else 0
                display = 1 if (i % 7 == 0) else 0
                tpr = 1 if (i % 8 == 0) else 0
                rows.append({
                    "WEEK_END_DATE": w,
                    "STORE_ID": store,
                    "UPC": upc,
                    "UNITS": units,
                    "VISITS": 100,
                    "HHS": 50,
                    "SPEND": units * price,
                    "PRICE": price,
                    "BASE_PRICE": base_price_col,
                    "FEATURE": feature,
                    "DISPLAY": display,
                    "TPR_ONLY": tpr,
                    "DESCRIPTION": f"Product {upc}",
                    "MANUFACTURER": "MFG",
                    "CATEGORY": category,
                    "SUB_CATEGORY": "SUB",
                    "PRODUCT_SIZE": "1oz",
                    "STORE_NAME": "Store X",
                    "ADDRESS_CITY_NAME": "City",
                    "ADDRESS_STATE_PROV_CODE": "ST",
                    "MSA_CODE": "MSA",
                    "SEG_VALUE_NAME": "SEG",
                    "PARKING_SPACE_QTY": 10,
                    "SALES_AREA_SIZE_NUM": 1000,
                    "AVG_WEEKLY_BASKETS": 200,
                })
    return pd.DataFrame(rows)


def test_load_and_engineer_features(tmp_path):
    df = make_sample_df()
    # add an invalid row that should be dropped by load_data
    invalid = df.iloc[0].copy()
    invalid["PRICE"] = 0
    invalid["BASE_PRICE"] = 0
    df_with_bad = pd.concat([pd.DataFrame([invalid]), df], ignore_index=True)

    p = tmp_path / "sample.csv"
    df_with_bad.to_csv(p, index=False)

    loaded = load_data(str(p))
    # should have dropped the invalid row
    assert len(loaded) == len(df)

    eng = engineer_features(loaded)
    # important engineered columns exist
    for col in ["DISCOUNT_DEPTH", "LOG_PRICE", "LOG_BASE_PRICE", "LOG_UNITS", "FEATURE_X_DISPLAY", "ANY_PROMO", "REL_PRICE", "UNITS_PER_HH"]:
        assert col in eng.columns

    # REL_PRICE should be PRICE / mean(price) per category-week
    sample_row = eng.iloc[0]
    cat_week_mean = eng[(eng["CATEGORY"] == sample_row["CATEGORY"]) & (eng["WEEK_END_DATE"] == sample_row["WEEK_END_DATE"])]["PRICE"].mean()
    assert np.isclose(sample_row["REL_PRICE"], sample_row["PRICE"] / cat_week_mean)


def test_decompose_baseline_incremental():
    # make a series with >=5 non-promo weeks for a given UPC-store
    df = make_sample_df()
    # Force ANY_PROMO to 0 for UPC 1111 and STORE 101 for first 6 weeks
    mask = (df["UPC"] == 1111) & (df["STORE_ID"] == 101)
    df.loc[mask, "FEATURE"] = 0
    df.loc[mask, "DISPLAY"] = 0
    df.loc[mask, "TPR_ONLY"] = 0

    df = engineer_features(df)
    out = decompose_baseline_incremental(df)

    assert "BASELINE_UNITS" in out.columns
    assert "INCREMENTAL_UNITS" in out.columns

    # For non-promo weeks incremental should be very small (close to 0)
    nonpromo = out[(out["UPC"] == 1111) & (out["STORE_ID"] == 101) & (out["ANY_PROMO"] == 0)]
    assert (nonpromo["INCREMENTAL_UNITS"] >= -1e-6).all()


def test_fit_model_and_recommend(tmp_path):
    df = make_sample_df()
    df = engineer_features(df)

    models = fit_elasticity_model(df, group_col=None)
    assert "ALL" in models
    res = models["ALL"]
    # model should contain LOG_PRICE coefficient
    assert "LOG_PRICE" in res.params.index

    # create a fake per-upc summary (as fit by summarize_models when per-upc modelling is used)
    summary_upc = pd.DataFrame({"GROUP": [1111, 2222], "PRICE_ELASTICITY": [-2.0, -1.5]})
    opt = recommend_optimal_price(df, summary_upc, margin_pct_assumed=0.30)
    # ensure optimal price column exists and some values are finite
    assert "OPTIMAL_PRICE" in opt.columns
    assert opt["OPTIMAL_PRICE"].notna().any()


if __name__ == "__main__":
    pytest.main(["-q"])
