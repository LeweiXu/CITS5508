# Chapter 4 — Training Models

> **Study booklet — CITS5508.** The most exam-dense chapter. The sample exam draws Q1a
> (logistic-regression "how many hours" computation), Q1b (bias/variance + ridge α), Q1c
> (polynomial features), and Q5b (ridge grid-search) directly from here. Deep dives below on
> **gradient descent**, **reading learning curves**, **bias/variance**, **ridge vs lasso**,
> and **logistic/softmax regression**, with worked exam-style calculations.

---

## 1. Linear Regression

A linear model predicts a **weighted sum of features + a bias (intercept) term**:

`ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ = θ·x = θᵀx`

- *ŷ* prediction; *n* features; *θⱼ* parameters (θ₀ = bias, θ₁…θₙ = feature weights);
  by convention *x₀ = 1* so the bias folds into the dot product.

**Training = find θ minimising a cost function.** We use **MSE** (simpler than RMSE, same
minimiser since √ is monotonic):

`MSE(θ) = (1/m) · Σᵢ ( θᵀxⁱ − yⁱ )²`

> **Training loss vs evaluation metric:** the loss you *optimise* during training often
> differs from the metric you *report*. The training loss should be easy to optimise and
> correlate with the metric. E.g. classifiers are trained on **log loss** but evaluated on
> precision/recall — log loss is easy to minimise and doing so usually improves P/R.

### The Normal Equation (closed-form solution)
A direct formula for the optimal θ:

`θ̂ = (XᵀX)⁻¹ Xᵀ y`

- **Pros:** exact; no hyperparameters; no feature scaling needed.
- **Cons:** computing `(XᵀX)⁻¹` is **O(n²·⁴) to O(n³)** in the number of features *n* → very
  slow when *n* is large; also fails if `XᵀX` is singular (e.g. *m < n* or redundant
  features).
- scikit-learn's `LinearRegression` actually uses the **SVD pseudoinverse** `θ̂ = X⁺y`, which
  is **O(n²)**, always defined (handles singular cases), and more numerically stable.
- Both are **O(m)** in the number of instances → handle large *m* well (if they fit in
  memory). Prediction is fast: linear in #instances and #features.

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_       # bias term, feature weights (kept separate)
```

---

## 2. 🔍 Deep dive — Gradient Descent (GD)

A **generic iterative optimisation** algorithm: tweak parameters step by step to minimise a
cost function. Analogy: lost in fog, walk downhill in the direction of steepest descent until
the slope (gradient) is zero.

**Algorithm:** start with **random initialisation** of θ; repeatedly compute the gradient
∇θ (which points *uphill*) and step in the *opposite* direction:

`θ_next = θ − η · ∇θ MSE(θ)`

### The learning rate η (a hyperparameter)
- **Too small** → many tiny steps → very slow convergence.
- **Too large** → overshoots, may **diverge** (cost grows, jumps across the valley).
- Find a good value with grid search (limit #epochs so slow ones get pruned).

### Convexity, local minima, and feature scaling
- The MSE cost for linear regression is **convex** (one global minimum, no local minima) and
  smooth → GD is **guaranteed** to approach the global minimum (given small-enough η and
  enough time).
- Non-convex cost surfaces have **local minima**, **plateaus**, **ridges** — GD can get
  stuck or stall.
- **Feature scaling matters:** with unscaled features the cost "bowl" is **elongated**, so GD
  zig-zags and converges slowly; with scaling it heads straight to the minimum. **Always
  scale before GD** (`StandardScaler`).

### The three GD variants (know the trade-offs cold)

| Variant | Data used per step | Pros | Cons |
|---|---|---|---|
| **Batch GD** | The **whole** training set | Stable, straight path; stops *at* the minimum | Very slow on large *m*; needs all data in memory |
| **Stochastic GD (SGD)** | **One random** instance | Fast steps; out-of-core; randomness can **escape local minima** | Noisy/bouncy; never settles exactly; needs a learning schedule |
| **Mini-batch GD** | A **small random batch** | Hardware/GPU efficient; less erratic than SGD | Can get stuck in local minima; tuning batch size |

- **Epoch** = one pass over the training set.
- **SGD bounces** around the minimum forever → use a **learning schedule** (gradually shrink
  η, like *simulated annealing*) so it settles. Too-fast decay → stuck; too-slow → keeps
  bouncing.
- **SGD requires IID data** — shuffle instances (or pick randomly) each epoch; if data is
  sorted by label, SGD optimises one label then the next and won't converge well.
- All variants converge to essentially the **same model** for linear regression and predict
  identically. Batch GD & SGD scale **well with #features** (unlike the Normal equation).

```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None,
                       eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())     # fit() expects 1D y
```

---

## 3. Polynomial Regression

Fit **nonlinear** data with a **linear** model by adding **powers of features** as new
features, then running linear regression on the extended set.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)     # adds x², (and cross-terms for multiple features)
LinearRegression().fit(X_poly, y)
```

- With **multiple features**, `PolynomialFeatures` also adds **interaction/cross terms** —
  this lets the model capture relationships *between* features that plain linear regression
  can't. For features *a, b* with `degree=3`: a², a³, b², b³, **ab, a²b, ab²**.
- `PolynomialFeatures(degree=d)` turns *n* features into **(n+d)! / (d!·n!)** features —
  beware the **combinatorial explosion**.

### 🔍 Worked example — sample-exam Q1c (cubic features, 2 inputs)
"All cubic polynomial features for x₁ and x₂" (degree 3, two features): every monomial of
total degree 1–3:
`x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³` (+ the bias term 1 if `include_bias=True`).

---

## 4. 🔍 Deep dive — Learning curves (reading train-vs-validation plots)

A **learning curve** plots the model's **training error** and **validation error** as a
function of **training-set size** (or training iterations). It's the key visual tool for
diagnosing under/overfitting — a core exam skill.

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
```

**How to read the two curves:**

| Pattern | Diagnosis | What it looks like |
|---|---|---|
| Both errors high, curves **close together** on a plateau | **Underfitting (high bias)** | Train error rises from ~0 and plateaus high; validation error falls to meet it just above |
| Train error **low**, large **gap** to a higher validation error | **Overfitting (high variance)** | Train curve stays low; validation curve well above; gap shrinks if you add data |
| Both errors low and close | **Good fit** | Both converge low |

**Why the curves behave this way (the *why*):**
- *Training error:* with 1–2 instances the model fits perfectly (error ≈ 0). As instances are
  added, the model can no longer fit them all (noise + model limits), so training error
  **rises** and plateaus.
- *Validation error:* with few training instances the model can't generalise → validation
  error is **high**; as it sees more data it improves → validation error **falls**.

**Actionable rules:**
- **Underfitting** → adding more data **won't help**. Use a more powerful model or better
  features.
- **Overfitting** → adding more data **does help** (the gap closes); or regularise / simplify
  the model.

---

## 5. 🔍 Deep dive — The Bias/Variance trade-off

Generalisation error decomposes into **three** parts:

- **Bias** — error from **wrong assumptions** (e.g. assuming linear when it's quadratic).
  High bias → **underfitting**. *(Distinct from the bias *term* θ₀.)*
- **Variance** — error from **excessive sensitivity** to small variations in the training
  data. A model with many degrees of freedom (high-degree polynomial) → high variance →
  **overfitting**.
- **Irreducible error** — from data **noise** itself; only reducible by cleaning the data.

**The trade-off:** increasing model complexity → **variance ↑, bias ↓**; decreasing
complexity → **bias ↑, variance ↓**. You tune complexity/regularisation to balance them.

### 🔍 Worked example — sample-exam Q1b
*"Ridge regression; training and validation errors are almost equal and fairly high. High
bias or high variance? Increase or decrease α?"*
- Errors **close together + both high** → the curves of a model that **underfits** → **high
  bias**.
- α controls regularisation strength; **more α = simpler = more bias**. To *reduce* bias you
  need a **less constrained** model → **decrease α**. ✅

---

## 6. Regularized Linear Models

Constraining (regularising) the weights reduces overfitting. **Always scale features first**
(regularised models are scale-sensitive). The penalty is added to the cost **during training
only** — evaluate with the plain (R)MSE.

### Ridge regression (ℓ₂ penalty)
`J(θ) = MSE(θ) + α · Σᵢ₌₁ⁿ θᵢ²`  (the bias θ₀ is **not** penalised)

- Adds α times the squared **ℓ₂ norm** of the weights → shrinks weights toward (but not to)
  zero.
- **α = 0** → plain linear regression; **α very large** → all weights ≈ 0 → flat line at the
  data mean (high bias).
- Has a closed-form solution and a GD form. `RidgeCV` auto-tunes α via CV (much faster than
  `GridSearchCV`).

```python
from sklearn.linear_model import Ridge
Ridge(alpha=0.1, solver="cholesky").fit(X, y)
```

### Lasso regression (ℓ₁ penalty)
`J(θ) = MSE(θ) + 2α · Σᵢ₌₁ⁿ |θᵢ|`

- Uses the **ℓ₁ norm**. **Key property:** lasso drives the weights of the **least important
  features exactly to zero** → automatic **feature selection** → a **sparse model**.
- *Why ℓ₁ zeroes weights and ℓ₂ doesn't:* the ℓ₁ gradient is ±1 regardless of magnitude, so
  it keeps pushing small weights all the way to 0 (and bounces there); the ℓ₂ gradient
  shrinks as weights approach 0, so it slows and never fully eliminates them.

```python
from sklearn.linear_model import Lasso
Lasso(alpha=0.1).fit(X, y)
```

### Elastic Net (mix of ℓ₁ and ℓ₂)
Weighted sum of ridge + lasso penalties, mix ratio *r* (`l1_ratio`): r=0 → ridge, r=1 → lasso.

### Which to use? (exam answer)
- **Almost always use *some* regularisation** → avoid plain linear regression.
- **Ridge** is a good default.
- If you suspect **only a few features matter** → **lasso** or **elastic net** (they zero out
  useless features).
- Prefer **elastic net over lasso** when features **outnumber instances** or are **strongly
  correlated** (lasso behaves erratically there).

### Early stopping ("a beautiful free lunch")
For iterative learners (GD), **stop training as soon as the validation error reaches its
minimum** (then starts rising = overfitting begins). For noisy SGD/mini-batch curves, stop
only after validation has stayed above the minimum for a while, then **roll back** to the
best-recorded parameters.

```python
from copy import deepcopy
best_val = float("inf")
for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)      # incremental
    val = root_mean_squared_error(y_valid, sgd_reg.predict(X_valid_prep))
    if val < best_val:
        best_val, best_model = val, deepcopy(sgd_reg)   # copy params too
```

---

## 7. 🔍 Deep dive — Logistic Regression

Used for **binary classification**: estimate the probability that an instance is in the
positive class, then threshold (default 0.5).

**Model:** compute a weighted sum (the **logit** *t = θᵀx*), then squash through the
**sigmoid / logistic function**:

`p̂ = hθ(x) = σ(θᵀx)`,  where  `σ(t) = 1 / (1 + exp(−t))`

- σ is **S-shaped**, outputs (0,1); σ(t) = 0.5 exactly at **t = 0**, < 0.5 for t < 0, ≥ 0.5
  for t ≥ 0.
- **Prediction:** ŷ = 1 if p̂ ≥ 0.5 (i.e. if θᵀx ≥ 0), else 0.
- The **logit** *t* = `logit(p) = log(p / (1−p))` = the **log-odds** (log of the
  positive/negative probability ratio); logit is the inverse of the logistic.

### 🔍 Worked example — sample-exam Q1a (must be able to do this)
Model: `p̂ = σ(θ₀ + θ₁·hours + θ₂·GPA)` with θ₀ = −6, θ₁ = 0.05, θ₂ = 1, GPA = 3.5.
A **50% chance** means p̂ = 0.5 ⇔ the logit **t = 0**:

`−6 + 0.05·hours + 1·(3.5) = 0`
`0.05·hours = 6 − 3.5 = 2.5`
`hours = 2.5 / 0.05 = `**`50 hours`**. ✅

> Key trick: "50% probability" ⇒ set the linear part (logit) to **0** and solve. (For some
> other probability p, use t = logit(p) = ln(p/(1−p)) on the right-hand side instead of 0.)

### Training & cost function (log loss)
Train θ to give **high p̂ for positives, low p̂ for negatives**. Cost for one instance:
`−log(p̂)` if y=1, `−log(1−p̂)` if y=0 — huge cost when confidently wrong, ~0 when confidently
right. Averaged over the set = the **log loss**:

`J(θ) = −(1/m) Σᵢ [ yⁱ·log(p̂ⁱ) + (1−yⁱ)·log(1−p̂ⁱ) ]`

- **No closed-form solution**, but the log loss is **convex** → GD finds the global minimum.
- (Minimising log loss = maximum-likelihood under a Gaussian-per-class assumption; wrong
  assumptions → bias, just as MSE assumes linear data + Gaussian noise.)

### Decision boundary & regularisation
- The decision boundary is where p̂ = 0.5, i.e. **θᵀx = 0** — a **linear** boundary (a
  point/line/hyperplane). For the iris petal-width example it's ≈ 1.6 cm.
- scikit-learn's `LogisticRegression` regularisation strength is **`C`** (the **inverse** of
  α): **higher C = less regularisation**. ℓ₂ penalty by default.

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=1.0, random_state=42)   # ℓ2 by default
log_reg.fit(X_train, y_train)
log_reg.predict_proba(X_new)          # probabilities
```

---

## 8. Softmax Regression (multinomial logistic regression)

Generalises logistic regression to **multiple classes directly** (no OvR/OvO needed).

**How it works:** for instance **x**, compute a score `sₖ(x) = θ⁽ᵏ⁾ᵀx` per class *k*, then
apply the **softmax** to turn scores into probabilities:

`p̂ₖ = σ(s(x))ₖ = exp(sₖ(x)) / Σⱼ exp(sⱼ(x))`

- Predict the class with the **highest probability** (= highest score): `ŷ = argmaxₖ p̂ₖ`.
- **Multiclass, not multioutput** → only for **mutually exclusive** classes (iris species),
  **not** for recognising multiple people in one image.
- Trained by minimising **cross-entropy** (penalises low probability on the true class);
  for K = 2 it **reduces to the logistic log loss**.
- The decision boundaries between any two classes are **linear**; the model can predict a
  class with estimated probability **below 50%** (e.g. where three classes each ≈ 33%).

```python
# LogisticRegression does softmax automatically for >2 classes (solver="lbfgs")
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)
softmax_reg.predict_proba([[5, 2]]).round(2)    # e.g. [[0.00, 0.04, 0.96]]
```

> **Exam tip (sample-exam-style):** "classify pictures as outdoor/indoor **and**
> daytime/nighttime" → these are **two independent binary** outputs → use **two logistic
> regression** classifiers, **not** one softmax (softmax is for mutually-exclusive classes).

---

## 9. Quick-reference summary

- **Linear regression**: `ŷ = θᵀx`; train by minimising **MSE** via the **Normal
  equation/SVD** (closed-form; slow in *n*, no scaling needed) **or gradient descent**
  (iterative; scales to large *n*; needs scaling).
- **GD variants:** Batch (stable, slow), **Stochastic** (fast, noisy, escapes local minima,
  needs a learning schedule), Mini-batch (GPU-efficient middle ground). η too small = slow,
  too large = diverges.
- **Polynomial regression**: add powers + cross terms; (n+d)!/(d!n!) features → explosion.
- **Learning curves**: both-high-and-close = **underfit**; low-train-with-gap = **overfit**.
  More data helps overfitting, not underfitting.
- **Bias/variance**: complexity ↑ → variance ↑, bias ↓. High bias = underfit; high variance
  = overfit.
- **Regularisation**: **Ridge (ℓ₂)** shrinks weights; **Lasso (ℓ₁)** zeroes weights (feature
  selection, sparse); **Elastic Net** mixes them. **α ↑ = more regularisation = more bias.**
- **Logistic regression**: `σ(θᵀx)`; 50% prob ⇔ logit = 0; log-loss cost (convex); linear
  decision boundary; regularised by **C** (inverse of α).
- **Softmax**: multiclass, mutually-exclusive; softmax of per-class scores; cross-entropy
  loss; reduces to log loss for 2 classes.
