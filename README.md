# DSC624 Assignment – LLM-Assisted Inventory Demand Forecasting & Route Optimization

**Parthvi Manoj · 2025EMAI10029 · DSC624 NLP & LLMs, IIIT Kottayam**

Task 1 of the DSC624 assignment: does adding open-weight LLM representations of product-category
text improve a numerical demand forecaster, and how do forecasts feed inventory and routing decisions?

## Dataset
DataCo SMART Supply Chain for Big Data Analysis –
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
The notebook downloads it automatically with `kagglehub` (a Kaggle account/API token may be required).

## How to run
1. Open `2025EMAI10029_Parthvi_Manoj_DSC624_Assignment_Answers.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU** (the notebook loads both LLMs in fp16 using CUDA).
3. Install packages if needed: `pip install -r requirements.txt`.
4. Run all cells from top to bottom. Execution time depends on model downloads, runtime availability and hardware.
   All tables are written to `/content/dsc624_final_outputs/`.

## Report
`2025EMAI10029_Parthvi_Manoj_DSC624_Assignment_Report.pdf` (6 pages): methodology, results and discussion.

## Pipeline
| Step | What it does |
|---|---|
| Demand panel | Order-item quantity excluding CANCELED / SUSPECTED_FRAUD / PAYMENT_REVIEW, summed to category × week |
| Splits | Train 2015-01-05 → 2017-01-23 · Validation 2017-01-30 → 2017-06-26 · Test 2017-07-03 → 2017-09-25 (24 warm-start categories) |
| Baselines | Naive lag-1; Random Forest on lags, rolling stats, calendar features, one-hot category |
| LLM enhancement | Frozen Qwen2.5-1.5B-Instruct and Phi-3.5-mini-instruct (both fp16) → mean-pooled hidden states → PCA(4) → appended to RF features |
| Robustness | 5 seeds, random-vector control, no-category-ID ablation, week-block bootstrap CI |
| Inventory | Reorder points + base-stock back-test against realised demand |
| Routing | DBSCAN clusters → nearest neighbour vs 2-opt vs OR-Tools on 20-stop samples |

## Main outputs
`test_model_comparison.csv`, `validation_model_comparison.csv`, `zero_forecast_diagnostic.csv`,
`robustness_summary.csv`, `robustness_bootstrap_mae.csv`, `inventory_backtest.csv`,
`route_comparison.csv`, `final_experiment_summary.json`.

## Reproducibility notes
Global seed 42; robustness seeds 42, 7, 123, 2024, 31337. Tested on Google Colab (Tesla T4, Python 3.13, PyTorch 2.x).
Inventory and routing are scenario-based: DataCo has no stock-on-hand, supplier lead times or road network.
