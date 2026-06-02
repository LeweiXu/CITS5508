# Chapter 2 — End-to-End Machine Learning Project

> **Study booklet — CITS5508.** This chapter is less heavily examined than the others, but
> several of its concepts are *core exam skills*: RMSE vs MAE (and norms), data snooping &
> stratified sampling, preprocessing without leakage, pipelines, cross-validation, and
> grid/randomised search. The sample exam's coding question (Q5b: scale → grid-search Ridge
> → MSE) is lifted straight from this workflow. Learn the **concepts and the API patterns**;
> you don't need the full California-housing case study by heart.

**Running example:** predict the **median house value** of a California district from census
features. Framed as **supervised**, **multiple regression** (many features), **univariate**
(one output), **batch** learning.

---

## 1. The 8-step project checklist

1. Look at the big picture (frame the problem).
2. Get the data.
3. Explore & visualise to gain insight.
4. Prepare the data for ML algorithms.
5. Select & train a model.
6. Fine-tune the model.
7. Present the solution.
8. Launch, monitor & maintain.

---

## 2. Frame the problem & pick a performance measure

- Ask **what the business objective is** and **how the output will be used** — this
  determines framing, algorithms, the metric, and the effort.
- Identify the **current/baseline solution** and whether the task is
  supervised/unsupervised, classification/regression, batch/online.

### 🔍 Deep dive — RMSE vs MAE (and the idea of a norm)
Both measure the distance between the prediction vector and the target vector. The choice
encodes *how you want to penalise errors*.

- **RMSE (Root Mean Square Error)** — the **ℓ₂ (Euclidean) norm**:
  
  `RMSE = sqrt( (1/m) · Σᵢ ( h(xⁱ) − yⁱ )² )`
  
  Squaring means **large errors dominate** → RMSE is **sensitive to outliers** but ideal
  when large errors are especially bad and noise is roughly Gaussian. **Default for
  regression.**

- **MAE (Mean Absolute Error)** — the **ℓ₁ (Manhattan) norm**:
  
  `MAE = (1/m) · Σᵢ | h(xⁱ) − yⁱ |`
  
  Errors weighted linearly → **more robust to outliers** than RMSE.

> **General principle:** a higher-order norm ℓₖ focuses more on large values and ignores
> small ones, so it's more outlier-sensitive. RMSE (ℓ₂) > MAE (ℓ₁) in outlier sensitivity.

**Notation:** *m* = number of instances; *xⁱ* = feature vector of instance *i*; *yⁱ* = its
label; **X** = matrix of all feature vectors (one row each); *h* = the prediction function
(**hypothesis**); *ŷ = h(x)* = a prediction.

**Check the assumptions** — confirm the downstream really needs numeric prices (regression)
and not price *categories* (which would make it classification).

---

## 3. Create the test set *first* (and why)

Set aside ~20% **before** exploring, to avoid **data-snooping bias** (your brain overfits to
patterns you noticed in the test data → over-optimistic estimates).

- **Random sampling** is fine for large datasets but risks **sampling bias** on smaller ones.
- Use a **stable** split so the test set is consistent across runs (hash each instance's ID,
  or fix `random_state`).
- **Stratified sampling** — when an important feature (e.g. median income) exists and the
  dataset isn't huge, split so the test set keeps the same **proportions of strata** as the
  full data (bin income into categories, sample within each). Avoids a test set that
  misrepresents key subgroups.

```python
from sklearn.model_selection import train_test_split
# simple random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# stratified split on an important categorical column
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=income_cat, random_state=42)
```

---

## 4. Explore & visualise

- Scatter plots (e.g. latitude/longitude coloured by price) reveal geographic structure.
- **Pearson correlation coefficient** *r* ∈ [−1, 1]: +1 strong positive linear, −1 strong
  negative, 0 no *linear* relation. **Caveat:** captures only **linear** association — it
  can miss strong nonlinear relationships and says nothing about slope.
- **Attribute combinations** — engineered ratios (rooms-per-house, bedrooms-ratio,
  people-per-house) often correlate with the target far better than the raw counts. Feature
  engineering frequently beats model tweaking.

---

## 5. Prepare the data (always via functions / pipelines)

Transform data with **reusable transformers**, never by hand — so you can reapply identically
to the training set, test set, and live data, and treat the choices as hyperparameters.

### Cleaning missing values
1. Drop those **instances** (`dropna`).
2. Drop the whole **attribute** (`drop`).
3. **Impute** — fill with a value, usually the **median** (`SimpleImputer(strategy="median")`).
   Compute it on the **training set only**, store it, apply to all sets and to live data.

### Text & categorical attributes
- **OrdinalEncoder** — categories → integers 0,1,2…; implies an **order** the model will
  believe (nearby numbers = similar) — wrong for nominal categories.
- **OneHotEncoder** — one binary feature per category (one "hot"). Preferred for nominal
  categories; outputs a **sparse matrix**. Beware the **curse of dimensionality** with very
  high-cardinality categories (consider a meaningful numeric feature or a learned embedding).

### 🔍 Deep dive — feature scaling without data leakage (exam-critical)
Most algorithms (gradient descent, SVMs, k-NN, regularised linear models) behave badly when
features have very different scales.

- **Min-max scaling (normalisation)** → range [0,1]: `(x − min)/(max − min)`. `MinMaxScaler`.
  Sensitive to outliers.
- **Standardisation** → zero mean, unit variance: `(x − μ)/σ`. `StandardScaler`. Less
  outlier-sensitive; no bounded range.

**The leakage rule (this is exactly the sample-exam Q5b pattern):** **fit the scaler on the
training set only**, then `transform` train, test and live data with those *same* learned
statistics. Fitting on the test set leaks information about it into training.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn μ, σ from TRAIN only
X_test_scaled  = scaler.transform(X_test)        # apply the SAME μ, σ
```

Also: log-transform heavily **skewed** features before scaling; consider bucketizing; for
multimodal features add **RBF similarity** features to key modes; sometimes scale the
**target** and inverse-transform predictions.

### scikit-learn API design (good for "explain the design" questions)
- **Estimators** — have `fit()`; learned attributes end with `_` (e.g. `imputer.statistics_`).
  Hyperparameters are constructor arguments.
- **Transformers** — also have `transform()` / `fit_transform()` (imputers, scalers, encoders).
- **Predictors** — have `predict()` and `score()` (e.g. `LinearRegression`).

### Pipelines
- **`Pipeline`** chains transformers + a final estimator; `fit` runs each step's
  `fit_transform` in order. **`ColumnTransformer`** applies different pipelines to different
  columns (numeric vs categorical) — bundling *all* preprocessing into one reusable object.

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

---

## 6. Select & train a model, then evaluate honestly

- Train a few quick models (`LinearRegression`, `DecisionTreeRegressor`,
  `RandomForestRegressor`) and compare.
- A `DecisionTreeRegressor` giving **0 training RMSE** is a red flag for **overfitting**, not
  success — only held-out evaluation reveals it.

### k-fold cross-validation
Split the training set into **k folds**; train k times, each holding out one fold for
evaluation → k scores giving a **mean ± standard deviation** (the std tells you how *precise*
the estimate is). scikit-learn CV utilities expect a *utility* (higher = better), so cost
functions are passed negated (`scoring="neg_root_mean_squared_error"`).

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10,
                         scoring="neg_root_mean_squared_error")
rmse = -scores            # flip sign back
print(rmse.mean(), rmse.std())
```

---

## 7. Fine-tune

- **`GridSearchCV`** — exhaustively tries every combination in a grid, via CV. `best_params_`,
  `best_estimator_`. By default **refits** the winner on the whole training set
  (`refit=True`). Best when the search space is small.
- **`RandomizedSearchCV`** — samples a fixed number of random combinations (works with
  continuous distributions). Preferred for **large** search spaces — you control the compute
  budget and explore more values per hyperparameter.
- **Ensemble methods** — combining the best models often beats any single one (→ Ch. 7).
- **Analyse the best model** — `feature_importances_` shows which features matter (drop
  useless ones); inspect errors for systematic problems.

```python
# Sample-exam Q5b pattern, fully worked
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

grid = GridSearchCV(Ridge(),
                    {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},  # five values
                    cv=3, scoring="neg_mean_squared_error")
grid.fit(X_train_scaled, y_train)          # refit=True → best model on full train set
print("best alpha:", grid.best_params_["alpha"])
y_pred = grid.predict(X_test_scaled)       # predict with the refit best model
print("test MSE:", mean_squared_error(y_test, y_pred))
```

---

## 8. Evaluate on the test set, then launch

- Run the final tuned model on the test set **once** for the generalisation estimate. **Do
  not** tune afterwards (that overfits the test set).
- A **95% confidence interval** (e.g. via `scipy.stats.t`) communicates uncertainty honestly.
- **Launch / monitor / maintain:** save the whole pipeline+model (`joblib`), deploy, monitor
  live performance (models rot under data drift), alert on drops, retrain on fresh data, keep
  backups to roll back.

---

## 9. Concept checklist (reusable for the exam)

- **RMSE = ℓ₂** (outlier-sensitive, default); **MAE = ℓ₁** (robust).
- **Data-snooping bias** → set the test set aside first.
- **Stratified sampling** preserves subgroup proportions across train/test.
- **Pearson correlation** = linear association only.
- Missing values → drop rows / drop column / **impute (median)**; fit the imputer on **train**.
- **OrdinalEncoder vs OneHotEncoder**; one-hot for nominal categories.
- **Standardisation vs min-max**; **fit on train only, transform the rest** (no leakage).
- **Pipeline / ColumnTransformer** to bundle preprocessing reproducibly.
- **k-fold CV** for honest evaluation (mean ± std).
- **GridSearchCV vs RandomizedSearchCV** (small vs large/continuous search spaces).
- scikit-learn API: **estimator / transformer / predictor**; learned attributes end in `_`.
