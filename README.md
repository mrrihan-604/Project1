# House Price Prediction

## Business Objective

Predict the selling price of a house from property-related information for a real-estate company.

## Dataset Understanding

- 1,000 records and 9 columns.
- Target: `Price`.
- Numeric features: `Square_Feet`, `Bedrooms`, `Bathrooms`, `Floors`, and `Year_Built`.
- Categorical features: `Location` and `Condition`.
- `House_ID` is an identifier and is excluded from model training.
- The current dataset has no missing values or duplicate rows.

## Project Deliverables

| Task | Deliverable | Implementation |
| --- | --- | --- |
| TA-001 to TA-005 | Requirements, dataset understanding, feature/target identification, and data-quality checks | `README.md`, `EDA.py` |
| TA-007 | Data preparation and cleaning | `Arranging data.py` |
| TA-008 | Exploratory data analysis | `EDA.py` |
| TA-009 | Visualization results | `data_visualization.py`, `visualization_results/` |
| TA-010 | ML-ready dataset | `data_visualization.py`, `ml_ready_dataset.csv` |
| TA-011 to TA-014 | Train/test split, linear regression, prediction, and evaluation | `model_training.py`, `model_metrics.csv`, `test_predictions.csv` |
| TA-015 | Project documentation | `README.md` |

Team allocation, review, and presentation evidence for TA-006, TA-016, and TA-017 should be added by the team separately.

## Model Workflow

`model_training.py` performs an 80/20 train/test split with `random_state=42`, imputes missing values, scales numeric features, one-hot encodes categorical features, and trains a Linear Regression model. It saves the complete preprocessing-and-model pipeline to `trained_house_price_model.joblib`.

The latest held-out test results are:

| Metric | Value |
| --- | ---: |
| MAE | 23,574.82 |
| RMSE | 30,269.35 |
| R² | 0.948477 |

These metrics describe this dataset and split; they are not a guarantee of performance on new markets or future prices.

## Running the Project

Activate the project environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Train and evaluate the model:

```powershell
python model_training.py
```

Launch the desktop prediction interface:

```powershell
python app.py
```

Generate the visualization files:

```powershell
python data_visualization.py
```
