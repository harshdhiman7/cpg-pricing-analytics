# Price & Promotion Elasticity Pipeline for CPG Weekly Store-UPC Data

A comprehensive Python analytics pipeline for CPG (Consumer Packaged Goods) retail data. The pipeline computes price elasticity, promotional lift, baseline vs. incremental unit decomposition, regularized ElasticNet/Ridge estimates, and optimal price recommendations.

# Data

## (Nearly) Real-world data
Here at dunnhumby, we understand the importance of great data and the analysts who make sense of it. Uncovering patterns, predicting trends, validating theories — insight gained through analysing customer data is the foundation of our business and key to the success of every one of our clients.

But more than that, we just really love data. We love connecting the dots. We love the human stories data can help you tell. And we love the people who love data as much as we do. That’s why we created Source Files, a platform for sharing datasets inspired on the real-world, where fellow data geeks – from professors to students to data scientists – can easily access rich data sources. Whether you’re teaching a course, completing a class project, testing an algorithm, or running a hack-a-thon, Source Files is the place to go to put your theory into practice.

## Breakfast at the Frat
What’s inside?
A representation of sales and promotion information on five products from three brands within four categories (mouthwash, pretzels, frozen pizza, and boxed cereal) over 156 weeks.

Unit sales, households, visits, and spend data by product, store, and week

Base Price and Shelf Price, to determine a product’s discount, if any

Promotional support details (e.g. sale tag, in-store display), if applicable

## What’s it for?
This dataset is designed to facilitate time series analyses, including:

Price sensitivity analysis

Promotional effectiveness analysis

Comparing/contrasting results across products, categories or store geographies


## Features & Highlights

* **Data Ingestion & Validation:** Handles CSV and Excel formats, parses dates, validates numeric ranges, and filters invalid unit/price entries.
* **Feature Engineering:** Calculates log-transformed variables (`LOG_UNITS`, `LOG_PRICE`, `LOG_BASE_PRICE`), discount depth, promotion interaction terms (`FEATURE_X_DISPLAY`), and relative category price indexes (`REL_PRICE`).
* **Baseline vs. Incremental Unit Decomposition:** Uses non-promoted weeks to estimate trend and baseline units, separating baseline volume from promotional lift.
* **OLS Fixed-Effects Elasticity Models:** Estimates log-log price elasticities and promotional lifts by **Category** and **UPC**, incorporating Store and Week-of-Year fixed effects to control for demand shifters and seasonality.
* **L2-Focused ElasticNet / Ridge Regression:** Uses L2-dominant regularization (`l1_ratio=0.01`) with `StandardScaler` to reduce zero-coefficient issues associated with L1-heavy regularization.
* **Multicollinearity Checks (VIF):** Evaluates Variance Inflation Factors across promotional mechanics.
* **Optimal Price Recommendation Engine:** Estimates optimal prices using constant-elasticity markup logic based on target margins.

## Input Data Schema

The pipeline expects weekly CPG store-UPC data with the following required columns:

| Column                    | Type             | Description                        |
| ------------------------- | ---------------- | ---------------------------------- |
| `WEEK_END_DATE`           | Date / String    | Week ending date (`YYYY-MM-DD`)    |
| `STORE_ID`                | Numeric / String | Unique store identifier            |
| `UPC`                     | Numeric / String | Universal Product Code / Item ID   |
| `UNITS`                   | Numeric          | Total units sold                   |
| `VISITS`                  | Numeric          | Store foot traffic / visit count   |
| `HHS`                     | Numeric          | Household count                    |
| `SPEND`                   | Numeric          | Total sales dollars                |
| `PRICE`                   | Numeric          | Actual shelf/discounted price      |
| `BASE_PRICE`              | Numeric          | Regular non-promoted base price    |
| `FEATURE`                 | Binary (0/1)     | Feature advertising flag           |
| `DISPLAY`                 | Binary (0/1)     | In-store display flag              |
| `TPR_ONLY`                | Binary (0/1)     | Temporary Price Reduction flag     |
| `DESCRIPTION`             | String           | Product description                |
| `MANUFACTURER`            | String           | Brand / Manufacturer               |
| `CATEGORY`                | String           | Product category                   |
| `SUB_CATEGORY`            | String           | Product sub-category               |
| `PRODUCT_SIZE`            | String           | Pack size / unit volume            |
| `STORE_NAME`              | String           | Store description                  |
| `ADDRESS_CITY_NAME`       | String           | City location                      |
| `ADDRESS_STATE_PROV_CODE` | String           | State / Province code              |
| `MSA_CODE`                | Numeric / String | Metropolitan Statistical Area code |
| `SEG_VALUE_NAME`          | String           | Store segment type                 |
| `PARKING_SPACE_QTY`       | Numeric          | Store parking capacity             |
| `SALES_AREA_SIZE_NUM`     | Numeric          | Store floor area in sq. ft.        |
| `AVG_WEEKLY_BASKETS`      | Numeric          | Average weekly basket size         |

## Installation & Requirements

Python 3.8+ is recommended.

```bash
pip install numpy pandas statsmodels scikit-learn openpyxl
```

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

## Pipeline Outputs

All outputs are saved to the `./outputs/` directory, or the directory specified using `--output_dir`.

| File                                    | Description                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| `elasticity_by_category.csv`            | Category-level OLS price elasticities, p-values, and promotional lift percentages |
| `elasticity_by_category_elasticnet.csv` | Category-level L2-focused ElasticNet regularized elasticities                     |
| `elasticity_by_upc.csv`                 | UPC-level OLS price elasticities and promotional lift percentages                 |
| `elasticity_by_upc_elasticnet.csv`      | UPC-level L2-focused ElasticNet regularized elasticities                          |
| `baseline_vs_incremental.csv`           | Weekly baseline vs. incremental volume decomposition by UPC-store                 |
| `optimal_price_recommendations.csv`     | Model-guided optimal price recommendations and suggested percentage changes       |
| `model_diagnostics.txt`                 | Model fit statistics, VIF scores, and econometric diagnostics                     |

## Methodology

### 1. Log-Log Elasticity Model

The primary specification is:

$$
\ln(\text{UNITS}_{ist}) =
\beta_0
+ \beta_1 \ln(\text{PRICE}_{ist})
+ \gamma_1 \text{FEATURE}_{ist}
+ \gamma_2 \text{DISPLAY}_{ist}
+ \gamma_3 \text{TPR\_ONLY}_{ist}
+ \gamma_4(\text{FEATURE} \times \text{DISPLAY})_{ist}
+ \alpha_i
+ \delta_t
+ \varepsilon_{ist}
$$

Where:

* $\beta_1$ = **Price Elasticity of Demand**
* $\gamma_k$ = Promotional effects
* $\alpha_i$ = Store fixed effects
* $\delta_t$ = Week-of-Year fixed effects
* $\varepsilon_{ist}$ = Error term

For a log-dependent-variable model, the approximate promotional lift is:

$$
\text{Promo Lift} \approx \left(e^{\gamma_k}-1\right)\times100\%
$$

### 2. L2-Dominant ElasticNet Regularization

The pipeline uses a Ridge-dominant ElasticNet specification with:

```python
l1_ratio = 0.01
```

The optimization objective is:

$$
\min_{\beta}
\frac{1}{2N}\|y-X\beta\|_2^2
+
\alpha
\left[
l_1\_ratio\|\beta\|_1
+
\frac{1-l_1\_ratio}{2}\|\beta\|_2^2
\right]
$$

Features are standardized before fitting using `StandardScaler`.

After fitting, coefficients can be transformed back to the original feature scale:

$$
\beta_{\text{unscaled}}
=
\frac{\beta_{\text{scaled}}}{\sigma_X}
$$

This approach reduces the tendency of L1-heavy regularization to shrink small elasticity estimates completely to zero.

### 3. Optimal Price Recommendation

For products with elastic demand ($E < -1$), the constant-elasticity optimal price is:

$$
P^* = \frac{C}{1+\frac{1}{E}}
$$

Where:

* $P^*$ = Recommended optimal price
* $C$ = Estimated marginal cost
* $E$ = Estimated price elasticity

If an assumed target margin is used, marginal cost can be approximated as:

$$
C =
\text{BASE\_PRICE}
\times
(1-\text{Margin}_{\text{assumed}})
$$

The resulting recommendation should be interpreted as a model-based pricing benchmark rather than a guaranteed profit-maximizing price.

## License

MIT License. Free for commercial and non-commercial analytical workflows.
