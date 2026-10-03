# 🏠 Housing Prices Prediction — Decision Tree · Random Forest · XGBoost

A machine learning project predicting house sale prices from the [Kaggle House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.  
Three regression models are progressively built, tuned, and compared: **Decision Tree**, **Random Forest**, and **XGBoost**.

![Kaggle Housing Prices Competition](captures/pic_problem.png)

---

## 📁 Project Structure

```
housing_prices/
│
├── data/
│   ├── train.csv                           # Training dataset (1,460 rows)
│   └── test.csv                            # Test dataset (1,459 rows)
│
├── DecisionTreeRegressor.py                # Model 1 — Decision Tree
├── RandomForestRegressor.py                # Model 2 — Random Forest
├── xgboostregressor.py                     # Model 3 — XGBoost
│
├── submissionDecisionTreeRegressor.csv     # Kaggle submission (DT)
├── submissionRandomForestRegressor.csv     # Kaggle submission (RF)
├── submissionXGBRegressor.csv              # Kaggle submission (XGB)
│
├── web/                                    # HouseIQ — interactive web app
│   └── index.html
├── api/                                    # REST API serving the RF model
├── train_and_save_model.py                 # Serializes the trained RF model
│
└── README.md
```

---

## 📦 Dependencies

```bash
pip install pandas scikit-learn xgboost
```

---

## 🔧 Shared Preprocessing Pipeline

All three models share the **exact same preprocessing pipeline**, making the comparison fair and reproducible.

### 1. Feature Engineering (`create_features`)

Before any model sees the data, four **derived features** are computed from existing columns:

| New Feature | Formula | Why it helps |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Single number capturing total living space |
| `TotalBath` | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | Weighted count that respects half-bath utility |
| `HouseAge` | `YrSold − YearBuilt` | Age at time of sale — newer houses sell higher |
| `TotalPorchSF` | Sum of all porch/deck columns | Outdoor space is a real selling point |

```python
def create_features(df):
    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']    = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['HouseAge']     = df['YrSold'] - df['YearBuilt']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    return df
```

### 2. Sklearn Pipeline

The pipeline chains two steps so that the **same transformations** are applied consistently during training, cross-validation, and inference — **no data leakage**:

```
Raw data → [Preprocessor] → [Model] → Predictions
```

| Step | Numerical columns | Categorical columns |
|---|---|---|
| **Imputation** | `SimpleImputer(strategy='constant')` fills NaN with 0 | `SimpleImputer(strategy='most_frequent')` fills NaN with mode |
| **Encoding** | — | `OneHotEncoder(handle_unknown='ignore')` |

```python
preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='constant'), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

my_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
```

### 3. Train / Validation Split

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```

80 % of the 1,460 training rows → model fitting.  
20 % → held-out validation to compute MAE before submitting.

---

## 🌿 Model 1 — Decision Tree Regressor

**File:** `DecisionTreeRegressor.py`

### How it works

A Decision Tree recursively splits the dataset on feature thresholds, building a binary tree.  
Each leaf node outputs the average sale price of all training samples that fall into it.  
The key hyperparameter is `max_leaf_nodes` — it controls the tree's complexity.

### What I tested and why

> *"My first instinct was to find the optimal tree depth. I ran a loop over candidate `max_leaf_nodes` values and used 5-fold cross-validation to pick the one that generalises best — not just the one that scores best on a single split."*

```python
def get_cv_mae(max_leaf_nodes, X, y, preprocessor):
    test_pipeline = Pipeline([('preprocessor', preprocessor),
                               ('model', DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0))])
    scores = -1 * cross_val_score(test_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

# Final best: 46 leaf nodes
leaf_nodes = 46
avg_mae = get_cv_mae(leaf_nodes, X, y, preprocessor)
```

I tested values from 30 → 50. Too few leaves → underfitting (tree too simple). Too many → overfitting.  
`max_leaf_nodes=46` gave the best cross-validation average, but the final submission pipeline uses `max_leaf_nodes=42`.

### Results

| Metric | Value |
|---|---|
| **Validation MAE** (hold-out set) | **$23,903.84** |
| **Average CV MAE** (5-fold, `max_leaf_nodes=46`) | **$24,698.28** |
| **Kaggle Score** (RMSLE) | **0.19498** |

> **Interpretation:** Off by ~$24k on average. Not bad for a single tree — but the tree is fundamentally limited because it must sacrifice depth (accuracy) to avoid memorising the training data.

---

## 🌲🌲 Model 2 — Random Forest Regressor

**File:** `RandomForestRegressor.py`

### How it works

A Random Forest builds **many Decision Trees** in parallel, each trained on a random subset of the data (bootstrapping) and using only a random subset of features at each split.  
The final prediction is the **average** of all trees — this cancels out individual errors and drastically reduces variance.

### What I tested and why

> *"After getting the Decision Tree baseline, I wanted to see how much an ensemble could improve things. I started with `n_estimators=100` as a default, then experimented with more trees and with `max_depth` to control the depth of each individual tree in the forest."*

```python
def get_cv_mae(n_estimators, max_depth, X, y, preprocessor):
    test_pipeline = Pipeline([('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=n_estimators,
                                                               max_depth=max_depth,
                                                               random_state=0))])
    scores = -1 * cross_val_score(test_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

# Best combination found:
avg_mae = get_cv_mae(510, 30, X, y, preprocessor)
```

I tested several combinations of `(n_estimators, max_depth)`. More trees helped up to a point; unlimited depth was fine because bagging already handles variance.  
Best combo found: **510 trees, max_depth=30**.  
The final submission model uses `n_estimators=100` (faster, slightly higher MAE but still excellent).

### Results

| Metric | Value |
|---|---|
| **Validation MAE** (hold-out set) | **$17,414.45** |
| **Average CV MAE** (5-fold, 510 trees, depth 30) | **$17,628.64** |
| **Kaggle Score** (RMSLE) | **0.14831** |

> **Interpretation:** ~27% improvement over the Decision Tree. The ensemble averaging is doing exactly what it promises — each tree's noise cancels out.

---

## ⚡ Model 3 — XGBoost Regressor

**File:** `xgboostregressor.py`

### How it works

XGBoost (**Extreme Gradient Boosting**) builds trees **sequentially**, not in parallel like Random Forest.  
Each new tree is trained to correct the **residual errors** of all previous trees.  
This makes it much more sample-efficient: it focuses on the hard-to-predict cases.

Key hyperparameters:
- `n_estimators` — how many boosting rounds (trees) to run
- `learning_rate=0.05` — how much each tree contributes (smaller = more conservative, needs more trees)
- `n_jobs=4` — use 4 CPU cores for faster training

### What I tested and why

> *"With XGBoost I experimented with the number of boosting rounds (estimators). Too few → underfitting. Too many → the model starts overfitting since boosting is more aggressive than bagging. I used 5-fold CV to find the sweet spot."*

```python
model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, random_state=0)

def get_cv_mae(n_estimators, X, y, preprocessor):
    test_pipeline = Pipeline([('preprocessor', preprocessor),
                               ('model', XGBRegressor(n_estimators=n_estimators,
                                                      learning_rate=0.05,
                                                      n_jobs=4,
                                                      random_state=0))])
    scores = -1 * cross_val_score(test_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

# Best found: 194 estimators
estimators = 194
avg_mae = get_cv_mae(estimators, X, y, preprocessor)
```

I started with 500 estimators and noticed cross-validation MAE stopped improving after ~194.  
`learning_rate=0.05` is deliberately small — it makes the model learn slowly but generalise better.

### Results

| Metric | Value |
|---|---|
| **Validation MAE** (hold-out set) | **$18,019.72** |
| **Average CV MAE** (5-fold, 194 estimators) | **$16,895.20** |
| **Kaggle Score** (RMSLE) | **0.14785** |

> **Interpretation:** The CV MAE ($16,895) is the best of all three models. The Kaggle public score (0.14785) edges out Random Forest (0.14831) — confirming XGBoost generalises best on unseen data.

---

## ⚔️ Model Comparison

### 📊 Final Results Table

| Model | Val MAE | Avg CV MAE | Kaggle RMSLE |
|---|---|---|---|
| 🌿 Decision Tree (`max_leaf_nodes=42`) | $23,903.84 | $24,698.28 | 0.19498 |
| 🌲 Random Forest (`n_estimators=100`) | $17,414.45 | $17,628.64 | 0.14831 |
| ⚡ XGBoost (`n_estimators=500, lr=0.05`) | $18,019.72 | $16,895.20 | **0.14785** |

> 🏆 **XGBoost wins** on both CV MAE and Kaggle score.  
> 🥈 **Random Forest** wins on hold-out validation MAE, and powers the HouseIQ web app.

### Kaggle Submission Scores

![Kaggle Submission Scores](captures/Capture%20d'écran%202026-10-03%20224522.png)

---

### 🧠 Why Each Model Behaves Differently

#### Decision Tree — Underfits at useful depths
A single tree has to choose: be simple (low variance, high bias) or be complex (high variance, low bias).  
With `max_leaf_nodes=42` it sits in a compromise zone — it is not overfitting, but it is also not capturing all patterns.

#### Random Forest — Parallel ensemble, great baseline
By averaging 100–510 independent trees, variance drops massively.  
The trees don't "talk to each other" — they are trained independently and then averaged.  
This makes Random Forest very robust but it can plateau because it doesn't learn from its own mistakes.

#### XGBoost — Sequential boosting, learns from errors
Each new tree specifically targets the samples the previous trees got wrong.  
This means XGBoost squeezes more signal from the same data.  
The risk is overfitting — which is why controlling `learning_rate` and `n_estimators` via CV is crucial.

### 📐 Mathematical Intuition

**Random Forest** — variance reduction by averaging:
```
Variance(ensemble) ≈ Variance(single tree) / n_trees
```

**XGBoost** — additive residual correction:
```
F_m(x) = F_{m-1}(x) + η × h_m(x)
```
where `η` is the learning rate and `h_m` is the new tree fitted on the residuals of `F_{m-1}`.

---

## 🌐 HouseIQ — Interactive Web App

Beyond the ML scripts, a full **HouseIQ** web application was built to let anyone interact with the trained Random Forest model in real time.

### Predict Price Tab

Enter house details (quality, size, location, year built…) and get an instant price estimate powered by the trained model.

![HouseIQ Predict Price](captures/Capture%20d'écran%202026-10-03%20223553.png)

### Find by Budget Tab

Enter a budget and the app scans the 1,460-house dataset to show you houses in that price range, along with average statistics (quality, area, year built, bathrooms, etc.).

![HouseIQ Find by Budget](captures/Capture%20d'écran%202026-10-03%20223802.png)

### How the web app connects to the model

1. `train_and_save_model.py` trains the Random Forest pipeline and serialises it to disk (pickle).
2. The `api/` folder contains a lightweight server that loads the model and exposes a `/predict` endpoint.
3. The `web/index.html` frontend sends house feature values to the API and displays the result.

---

## 🚀 How to Run

### Run Decision Tree model
```bash
python DecisionTreeRegressor.py
# → prints Validation MAE and CV MAE
# → outputs: submissionDecisionTreeRegressor.csv
```

### Run Random Forest model
```bash
python RandomForestRegressor.py
# → prints Validation MAE and CV MAE
# → outputs: submissionRandomForestRegressor.csv
```

### Run XGBoost model
```bash
python xgboostregressor.py
# → prints Validation MAE and CV MAE
# → outputs: submissionXGBRegressor.csv
```

### Launch the HouseIQ Web App
```bash
python train_and_save_model.py   # train & save model
cd api && python app.py          # start API server
# open web/index.html in your browser
```

---

## 📈 Conclusion

> *"I started with a single Decision Tree to understand the data and establish a baseline (~$24k MAE). Then I moved to Random Forest, which immediately cut the error by ~27% just by averaging 100 trees. Finally, I brought in XGBoost, which leverages gradient boosting to squeeze out the best cross-validated performance at $16,895 CV MAE and a Kaggle score of 0.14785 — the best of all three. The key engineering decisions that helped across all models were: feature engineering (TotalSF, HouseAge, etc.), using Sklearn Pipelines to prevent data leakage, and relying on 5-fold cross-validation rather than a single train/val split to pick hyperparameters."*

| What improved results the most |
|---|
| ✅ Feature engineering (`TotalSF`, `HouseAge`, `TotalBath`, `TotalPorchSF`) |
| ✅ Pipeline-based preprocessing — no data leakage |
| ✅ 5-fold cross-validation for hyperparameter selection |
| ✅ Progressively switching: Decision Tree → Random Forest → XGBoost |
