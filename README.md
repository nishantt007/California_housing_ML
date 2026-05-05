# California Housing Price Predictor

> End-to-end machine learning pipeline for predicting California district median house values using regression models and scikit-learn.

---

## Overview

This project builds a supervised regression model to predict **median house values** for California census districts. It follows a complete ML workflow: exploratory data analysis, stratified train/test splitting, feature engineering, preprocessing pipelines, model training, cross-validation, hyperparameter tuning, and final test evaluation.

**Problem it solves:** Given census-level features of a California district (location, age of housing stock, room counts, population, income), predict the district's median house value.

**Key capabilities:**
- Stratified sampling to preserve income-category distribution across splits
- Custom scikit-learn transformer for derived features
- Unified preprocessing pipeline (imputation → feature engineering → scaling/encoding)
- Comparison of Linear Regression, Decision Tree, and Random Forest regressors
- Grid search hyperparameter tuning with cross-validation
- Feature importance analysis on the best model

---

## Visualizations


<img width="834" height="607" alt="image" src="https://github.com/user-attachments/assets/8119e181-fcd0-4f59-a10e-7cc0d70ca00b" />

Above is the geographical scatter plot representing district locations colored by price and sized by population


<img width="1021" height="707" alt="image" src="https://github.com/user-attachments/assets/b205785d-f3f1-4aa3-9e64-3f8bcb13ba92" />

Above is the scatter matrix repreesnting  pairwise correlations between key attributes

---

## Tech Stack

| Category | Tool / Library |
|----------|---------------|
| Language | Python 3.x |
| Data manipulation | pandas, NumPy |
| Visualization | matplotlib |
| Machine learning | scikit-learn |
| Notebook environment | Jupyter Notebook |
| Dataset format | CSV |

---

## Project Structure

```
.
├── california_housing.csv        # Raw dataset (20,640 districts, 10 features)
└── california_housing_ML.ipynb   # Main notebook: EDA → preprocessing → modeling
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Jupyter Notebook or JupyterLab

### Dependency Installation

```bash
pip install pandas numpy matplotlib scikit-learn notebook
```

Or with conda:

```bash
conda install pandas numpy matplotlib scikit-learn notebook
```

### Environment Setup (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install pandas numpy matplotlib scikit-learn notebook
```

---

## Usage

1. Clone or download the repository.
2. Ensure `california_housing.csv` and `california_housing_ML.ipynb` are in the same directory.
3. Launch Jupyter:

```bash
jupyter notebook california_housing_ML.ipynb
```

---

## Configuration

No external configuration files are required. Key parameters are set inline in the notebook:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `test_size` | Train/test split cell | `0.2` | Fraction of data reserved for testing |
| `random_state` | Split cells | `42` | Random seed for reproducibility |
| `cv` | Cross-validation cells | `10` | Number of K-fold splits |
| `n_estimators` | Grid search param grid | `[3, 10, 30]` | Number of trees in Random Forest |
| `max_features` | Grid search param grid | `[2, 4, 6, 8]` | Features considered per split |
| `add_bedrooms_per_room` | `CombinedAttributesAdder` | `True` | Toggle derived bedroom ratio feature |

---

## Dataset

**File:** `california_housing.csv`  

| Column | Type | Description |
|--------|------|-------------|
| `longitude` | float | District longitude |
| `latitude` | float | District latitude |
| `housing_median_age` | float | Median age of houses in district |
| `total_rooms` | float | Total rooms across all households |
| `total_bedrooms` | float | Total bedrooms (has missing values) |
| `population` | float | District population |
| `households` | float | Number of households |
| `median_income` | float | Median income (in ~$10,000 units) |
| `median_house_value` | float | **Target variable** — median house value (USD) |
| `ocean_proximity` | string | Categorical proximity to ocean |

- **Rows:** 20,640
- **Missing values:** `total_bedrooms` column has sparse nulls (handled by median imputation)
- **Target range:** ~$15,000 – $500,001 (values above $500,001 appear capped)

---

## Workflow

```
Raw CSV
   │
   ├─ EDA: histograms, geo-plots, correlation analysis
   │
   ├─ Stratified split → strat_train_set / strat_test_set
   │
   ├─ Feature engineering on training copy
   │      rooms_per_household, bedrooms_per_room, population_per_household
   │
   ├─ Preprocessing pipeline
   │      Numerical: SimpleImputer → CombinedAttributesAdder → StandardScaler
   │      Categorical: OneHotEncoder (ocean_proximity)
   │      Combined via ColumnTransformer → housing_prepared
   │
   ├─ Model training & CV evaluation
   │      LinearRegression   → underfits
   │      DecisionTreeRegressor → overfits (RMSE = 0 on train)
   │      RandomForestRegressor → best CV performance
   │
   ├─ GridSearchCV on RandomForest → best_estimator_
   │
   ├─ Feature importance ranking
   │
   └─ Final evaluation on strat_test_set → final_rmse
```
