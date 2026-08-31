# Price & Promotion Elasticity Pipeline for CPG Weekly Store-UPC Data

A comprehensive Python analytics pipeline for CPG (Consumer Packages Goods) retail data. This pipeline computes price elasticity, promo lift, baseline vs. incremental unit decomposition, regularized ElasticNet/Ridge estimates, and optimal price recommendations.

---

## ЁЯУМ Features & Highlights

- **Data Ingestion & Validation**: Handles CSV and Excel formats, parses dates, validates numeric ranges, and filters invalid unit/price entries.
- **Feature Engineering**: Calculates log-transformed variables (`LOG_UNITS`, `LOG_PRICE`, `LOG_BASE_PRICE`), discount depth, promo interaction terms (`FEATURE_X_DISPLAY`), and relative category price indexes (`REL_PRICE`).
- **Baseline vs. Incremental Unit Decomposition**: Isolates non-promoted weeks to estimate trend and baseline units using OLS, separating baseline volume from promotional lift.
- **OLS Fixed-Effects Elasticity Models**: Estimates log-log price elasticities and promotional lifts per **Category** and **UPC**, incorporating Store and Week-of-Year fixed effects to control for demand shifters and seasonality.
- **L2-Focused ElasticNet / Ridge Regression**: Solves zero-elasticity coefficient issues caused by L1 Lasso penalties by leveraging **L2-focused shrinkage** (`l1_ratio=0.01`) with feature standardization (`StandardScaler`).
- **Multicollinearity Checks (VIF)**: Automatically evaluates Variance Inflation Factors across promotional mechanics.
- **Optimal Price Recommendation Engine**: Solves for optimal prices using constant-elasticity markup logic based on target margins.

---

## ЁЯУВ Input Data Schema

The pipeline expects weekly CPG store-UPC data with the following required columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `WEEK_END_DATE` | Date / String | Week ending date (`YYYY-MM-DD`) |
| `STORE_ID` | Numeric / String | Unique store identifier |
| `UPC` | Numeric / String | Universal Product Code / Item ID |
| `UNITS` | Numeric | Total units sold |
| `VISITS` | Numeric | Store foot traffic / visit count |
| `HHS` | Numeric | Household count |
| `SPEND` | Numeric | Total sales dollars |
| `PRICE` | Numeric | Actual shelf/discounted price |
| `BASE_PRICE` | Numeric | Regular non-promoted base price |
| `FEATURE` | Binary (0/1) | Feature ad flag |
| `DISPLAY` | Binary (0/1) | In-store display flag |
| `TPR_ONLY` | Binary (0/1) | Temporary Price Reduction flag |
| `DESCRIPTION` | String | Product description |
| `MANUFACTURER` | String | Brand / Manufacturer |
| `CATEGORY` | String | Product category |
| `SUB_CATEGORY` | String | Product sub-category |
| `PRODUCT_SIZE` | String | Pack size / unit volume |
| `STORE_NAME` | String | Store description |
| `ADDRESS_CITY_NAME` | String | City location |
| `ADDRESS_STATE_PROV_CODE` | String | State / Province code |
| `MSA_CODE` | String | Metropolitan Statistical Area code |
| `SEG_VALUE_NAME` | String | Store segment type |
| `PARKING_SPACE_QTY` | Numeric | Store parking capacity |
| `SALES_AREA_SIZE_NUM` | Numeric | Store floor area sq. ft. |
| `AVG_WEEKLY_BASKETS` | Numeric | Average weekly basket size |

---

## ЁЯЫая╕П Installation & Requirements

Make sure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install numpy pandas statsmodels scikit-learn
```

---

## ЁЯЪА Usage

Execute the pipeline from the command line by passing the path to your dataset:

```bash
# Run on CSV data
python elasticity_pipeline.py --input /path/to/cpg_store_upc_data.csv

# Run on Excel data with custom output directory
python elasticity_pipeline.py --input /path/to/cpg_store_upc_data.xlsx --output_dir ./custom_outputs
```

---

## ЁЯУК Pipeline Outputs

All outputs are saved to the `./outputs/` directory (or specified `--output_dir`):

| File Output | Description |
| :--- | :--- |
| `elasticity_by_category.csv` | Category-level OLS price elasticities, p-values, and promo lift % |
| `elasticity_by_category_elasticnet.csv` | Category-level L2-focused ElasticNet regularized elasticities |
| `elasticity_by_upc.csv` | UPC-level OLS price elasticities and promo lift percentages |
| `elasticity_by_upc_elasticnet.csv` | UPC-level L2-focused ElasticNet regularized elasticities |
| `baseline_vs_incremental.csv` | Weekly decomposition of baseline vs. incremental volume per UPC-store |
| `optimal_price_recommendations.csv` | Model-guided optimal price recommendations & suggested % changes |
| `model_diagnostics.txt` | Fit summary statistics, VIF scores, and econometric warnings |

---

## ЁЯУИ Methodology Overview

### 1. Log-Log Elasticity Specification
$$\ln(	ext{UNITS}_{ist}) =  eta_0 +  eta_1 \ln(	ext{PRICE}_{ist}) + \gamma_1 	ext{FEATURE}_{ist} + \gamma_2 	ext{DISPLAY}_{ist} + \gamma_3 	ext{TPR\_ONLY}_{ist} + \gamma_4 (	ext{FEATURE} 	imes 	ext{DISPLAY})_{ist} +  lpha_i + \delta_t +  arepsilon_{ist}$$

- $ eta_1$: **Price Elasticity of Demand** (% change in demand given 1% price change)
- $\gamma_k$: Promotional lifts ($(\exp(\gamma_k) - 1) 	imes 100\%$)
- $ lpha_i$: Store fixed effects (`STORE_ID`)
- $\delta_t$: Week-of-year fixed effects (`WEEK_OF_YEAR`)

### 2. L2 ElasticNet Shrinkage (Ridge-Dominant)
Standard Lasso ($L1$) regularization forces small coefficients to zero, resulting in `0` elasticity for low-variance UPCs. The pipeline defaults to `l1_ratio=0.01` and scales numeric features prior to fitting:
$$\min_{ eta} rac{1}{2N} \|y - X eta\|_2^2 +  lpha \left( l_1	ext{\_ratio} \| eta\|_1 + rac{1 - l_1	ext{\_ratio}}{2} \| eta\|_2^2 
ight)$$
After fitting, coefficients are unscaled back to raw feature units:
$$ eta_{	ext{unscaled}} = rac{ eta_{	ext{scaled}}}{\sigma_X}$$

### 3. Optimal Markup Formula
For products with elastic demand ($E < -1$), optimal pricing under constant marginal cost $C$ is solved as:
$$P^* = rac{C}{1 + rac{1}{E}}$$
Where cost $C$ is estimated via:
$$C = 	ext{BASE\_PRICE} 	imes (1 - 	ext{Margin}_{	ext{assumed}})$$

---

## ЁЯУД License
MIT License. Free for commercial and non-commercial analytical workflows.
