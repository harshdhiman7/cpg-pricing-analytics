# Price & Promotion Elasticity Pipeline for CPG Weekly Store-UPC Data

A comprehensive Python analytics pipeline for CPG (Consumer Packaged Goods) retail data. The pipeline computes price elasticity, promotional lift, baseline vs. incremental unit decomposition, regularized ElasticNet/Ridge estimates, and optimal price recommendations.

---

## Data (sourced from Dunnhumby public repo)

### (Nearly) Real-world data

Here at dunnhumby, we understand the importance of great data and the analysts who make sense of it. Uncovering patterns, predicting trends, validating theories — insight gained through analysing customer data is the foundation of our business and key to the success of every one of our clients.

But more than that, we just really love data. We love connecting the dots. We love the human stories data can help you tell. And we love the people who love data as much as we do. That’s why we created Source Files, a platform for sharing datasets inspired on the real-world, where fellow data geeks – from professors to students to data scientists – can easily access rich data sources. Whether you’re teaching a course, completing a class project, testing an algorithm, or running a hack-a-thon, Source Files is the place to go to put your theory into practice.

### Breakfast at the Frat

**What’s inside?**
A representation of sales and promotion information on five products from three brands within four categories (mouthwash, pretzels, frozen pizza, and boxed cereal) over 156 weeks.

* Unit sales, households, visits, and spend data by product, store, and week
* Base Price and Shelf Price, to determine a product’s discount, if any
* Promotional support details (e.g. sale tag, in-store display), if applicable

**What’s it for?**
This dataset is designed to facilitate time series analyses, including:

* Price sensitivity analysis
* Promotional effectiveness analysis
* Comparing/contrasting results across products, categories or store geographies

---

## Features & Highlights

* **Automated Column Standardization:** Ingests datasets with arbitrary column casing and automatically standardizes headers to lower-case (`week_end_date`, `store_id`, `upc`, etc.) without altering external raw files.
* **Data Ingestion & Validation:** Handles CSV and Excel formats, parses dates, validates numeric ranges, and filters invalid unit/price entries.
* **Feature Engineering:** Calculates log-transformed variables (`log_units`, `log_price`, `log_base_price`), discount depth (`discount_depth`), promotion interaction terms (`feature_x_display`), and relative category price indexes (`rel_price`).
* **Baseline vs. Incremental Unit Decomposition:** Uses non-promoted weeks (`any_promo == 0`) to estimate trend and baseline units (`baseline_units`), separating baseline volume from promotional lift (`incremental_units`).
* **OLS Fixed-Effects Elasticity Models:** Estimates log-log price elasticities and promotional lifts by **category** and **upc**, incorporating Store (`store_id`) and Week-of-Year (`week_of_year`) fixed effects to control for demand shifters and seasonality.
* **L2-Focused ElasticNet / Ridge Regression:** Uses L2-dominant regularization (`l1_ratio=0.01`) with `StandardScaler` to reduce zero-coefficient issues associated with L1-heavy regularization.
* **Multicollinearity Checks (VIF):** Evaluates Variance Inflation Factors across promotional mechanics.
* **Optimal Price Recommendation Engine:** Estimates optimal prices using constant-elasticity markup logic based on target margins.

---

## Input Data Schema

The pipeline automatically standardizes input headers to lower-case. It expects weekly CPG store-UPC data containing the following required fields:

| Column | Type | Description |
| --- | --- | --- |
| `week_end_date` | Date / String | Week ending date (`YYYY-MM-DD`) |
| `store_id` | Numeric / String | Unique store identifier |
| `upc` | Numeric / String | Universal Product Code / Item ID |
| `units` | Numeric | Total units sold |
| `visits` | Numeric | Store foot traffic / visit count |
| `hhs` | Numeric | Household count |
| `spend` | Numeric | Total sales dollars |
| `price` | Numeric | Actual shelf/discounted price |
| `base_price` | Numeric | Regular non-promoted base price |
| `feature` | Binary (0/1) | Feature advertising flag |
| `display` | Binary (0/1) | In-store display flag |
| `tpr_only` | Binary (0/1) | Temporary Price Reduction flag |
| `description` | String | Product description |
| `manufacturer` | String | Brand / Manufacturer |
| `category` | String | Product category |
| `sub_category` | String | Product sub-category |
| `product_size` | String | Pack size / unit volume |
| `store_name` | String | Store description |
| `address_city_name` | String | City location |
| `address_state_prov_code` | String | State / Province code |
| `msa_code` | Numeric / String | Metropolitan Statistical Area code |
| `seg_value_name` | String | Store segment type |
| `parking_space_qty` | Numeric | Store parking capacity |
| `sales_area_size_num` | Numeric | Store floor area in sq. ft. |
| `avg_weekly_baskets` | Numeric | Average weekly basket size |

---

## Installation & Requirements

Python 3.8+ is recommended.

```bash
pip install numpy pandas statsmodels scikit-learn openpyxl

```

---

## Usage

Run the pipeline from the command line by passing the dataset path:

```bash
# Run on CSV data
python elasticity_pipeline.py --input /path/to/cpg_store_upc_data.csv

# Run on Excel data with custom output directory
python elasticity_pipeline.py \
    --input /path/to/cpg_store_upc_data.xlsx \
    --output_dir ./custom_outputs

```

---

## Pipeline Outputs

All outputs are saved with normalized lower-case headers to the `./outputs/` directory (or the directory specified using `--output_dir`).

| File | Description |
| --- | --- |
| `elasticity_by_category.csv` | Category-level OLS price elasticities, p-values, and promotional lift percentages |
| `elasticity_by_category_elasticnet.csv` | Category-level L2-focused ElasticNet regularized elasticities |
| `elasticity_by_upc.csv` | UPC-level OLS price elasticities and promotional lift percentages |
| `elasticity_by_upc_elasticnet.csv` | UPC-level L2-focused ElasticNet regularized elasticities |
| `baseline_vs_incremental.csv` | Weekly baseline vs. incremental volume decomposition by UPC-store |
| `optimal_price_recommendations.csv` | Model-guided optimal price recommendations and suggested percentage changes |
| `model_diagnostics.txt` | Model fit statistics, VIF scores, and econometric diagnostics |

---

## Methodology

### 1. Log-Log Elasticity Model

The primary regression specification is:

$$\ln(\text{units}_{ist}) = \beta_0 + \beta_1 \ln(\text{price}_{ist}) + \gamma_1 \text{feature}_{ist} + \gamma_2 \text{display}_{ist} + \gamma_3 \text{tpr\_only}_{ist} + \gamma_4(\text{feature} \times \text{display})_{ist} + \alpha_i + \delta_t + \varepsilon_{ist}$$

Where:

* $\beta_1$ = **Price Elasticity of Demand** (`log_price`)
* $\gamma_k$ = Promotional effects (`feature`, `display`, `tpr_only`, `feature_x_display`)
* $\alpha_i$ = Store fixed effects (`store_id`)
* $\delta_t$ = Week-of-Year fixed effects (`week_of_year`)
* $\varepsilon_{ist}$ = Cluster-robust error term

For a log-dependent-variable model, the approximate percentage promotional lift is calculated as:

$$\text{Promo Lift \%} = \left(e^{\gamma_k}-1\right)\times100\%$$

### 2. L2-Dominant ElasticNet Regularization

The pipeline uses a Ridge-dominant ElasticNet specification with:

```python
l1_ratio = 0.01

```

The optimization objective is:

$$\min_{\beta} \frac{1}{2N}\Vert{}y-X\beta\Vert{}_2^2 + \alpha \left[ l_1\_ratio\Vert{}\beta\Vert{}_1 + \frac{1-l_1\_ratio}{2}\Vert{}\beta\Vert{}_2^2 \right]$$

Features are standardized before fitting using `StandardScaler`. After fitting, coefficients are transformed back to the original feature scale:

$$\beta_{\text{unscaled}} = \frac{\beta_{\text{scaled}}}{\sigma_X}$$

This approach prevents L1-heavy regularization from prematurely shrinking small elasticity parameters completely to zero.

### 3. Optimal Price Recommendation

For products with elastic demand ($E < -1$), the constant-elasticity optimal price is:

$$P^* = \frac{C}{1+\frac{1}{E}}$$

Where:

* $P^*$ = Recommended optimal price (`optimal_price`)
* $C$ = Estimated marginal cost (`assumed_cost`)
* $E$ = Estimated price elasticity (`price_elasticity`)

Using an assumed target margin, marginal cost is approximated as:

$$C = \text{base\_price} \times (1 - \text{margin\_pct\_assumed})$$

The resulting recommendation serves as a directional model-based benchmark rather than a fixed decision rule.

---

## License

MIT License. Free for commercial and non-commercial analytical workflows.