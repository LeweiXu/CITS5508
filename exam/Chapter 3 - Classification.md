# Chapter 3 — Classification (+ k-Nearest Neighbours)

> **Study booklet — CITS5508.** One of the most heavily examined chapters. Expect
> confusion-matrix metrics, the precision/recall trade-off, **ROC/AUC** (a known
> "confusing" topic — given its own deep dive below), multiclass vs multilabel, OvO/OvR, and
> a manual **k-NN distance computation** (sample-exam Q2). A full **k-NN** section is added
> (course content beyond the textbook) with a worked bag-of-words example.

**Running example — MNIST:** 70,000 handwritten-digit images, each **28×28 = 784 features**
(pixel intensities 0 = white … 255 = black), label = the digit. Pre-split 60,000 train /
10,000 test, and **pre-shuffled** so CV folds are similar and order-sensitive algorithms
behave (**don't** shuffle time-series).

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target          # X.shape == (70000, 784)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

---

## 1. Binary classification & why accuracy misleads

A **binary classifier** distinguishes two classes (e.g. "5" vs "not-5"). The book uses
`SGDClassifier` — a **linear** model that scales to large data and handles instances one at a
time (so it suits online learning).

```python
from sklearn.linear_model import SGDClassifier
y_train_5 = (y_train == '5')             # binary target
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

### 🔍 Deep dive — the accuracy trap on skewed data
The 5-detector gets ~95% cross-val accuracy. But a `DummyClassifier` that **always predicts
"not-5"** scores ~**90%**, simply because only ~10% of digits are 5s.

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")   # ~0.95
# DummyClassifier (always "not-5")                                       # ~0.90
```

**Lesson:** on **skewed / imbalanced** datasets, accuracy is misleading — a useless model
can score high. Use the **confusion matrix** and its derived metrics instead.

---

## 2. The confusion matrix

Get **out-of-sample** predictions for the training set with `cross_val_predict` (k-fold,
returns a clean prediction per instance — predictions on data the model didn't train on),
then compute the matrix.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# array([[53892,   687],     <- actual negatives: 53892 TN, 687 FP
#        [ 1891,  3530]])    <- actual positives: 1891 FN, 3530 TP
```

**Rows = actual class, columns = predicted class.**

```
                 Predicted Neg     Predicted Pos
Actual Neg          TN                FP   (Type I error — false alarm)
Actual Pos          FN  (Type II)     TP
```

- **TN/TP** correct; **FP** = predicted positive, actually negative (**Type I**);
  **FN** = predicted negative, actually positive (**Type II**).
- A **perfect** classifier has nonzero entries **only on the main diagonal**.

---

## 3. Precision, recall, F₁

- **Precision** = accuracy of positive predictions: **`Precision = TP / (TP + FP)`**
  — "of those I flagged positive, what fraction really are?" *Trivially gamed:* one single
  confident correct positive prediction → precision = 1/1 = 100%, but useless.

- **Recall** (= **sensitivity** = **true positive rate, TPR**): **`Recall = TP / (TP + FN)`**
  — "of all real positives, what fraction did I catch?"

- **F₁ score** = **harmonic mean** of precision & recall:
  
  **`F₁ = 2 · (P · R) / (P + R)`**
  
  The harmonic mean **punishes low values**, so F₁ is high **only if both** P and R are
  high. Good single number for comparing classifiers — but it favours *similar* P and R,
  which isn't always what you want.

```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_train_5, y_train_pred)   # 0.837  = 3530 / (687 + 3530)
recall_score(y_train_5, y_train_pred)      # 0.651  = 3530 / (1891 + 3530)
f1_score(y_train_5, y_train_pred)          # 0.733
```

### 🔍 Deep dive — the precision/recall trade-off
A classifier computes a **score** via a **decision function**; if score > **threshold** →
positive, else negative.

- **Raise** the threshold → fewer positives → **precision ↑ (generally), recall ↓.**
- **Lower** the threshold → more positives → **recall ↑, precision ↓.**
- You **cannot maximise both** — increasing one tends to decrease the other.
- **Recall only ever decreases** as the threshold rises (smooth curve). **Precision can
  occasionally dip** when raising the threshold (bumpy curve) — adding/removing one instance
  can shift the TP/FP ratio either way.

**Which to prioritise depends on the cost of each error type:**
- *Kid-safe video filter* → favour **high precision** (reject many safe videos, never show
  a bad one), accept low recall.
- *Shoplifter / cancer / fraud detection* → favour **high recall** (catch almost all
  positives), accept low precision (false alarms tolerable).
- Slogan: *"99% precision — at what recall?"* Always quote them together.

```python
# choose a threshold yourself instead of using predict()
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# e.g. lowest threshold achieving >= 90% precision:
idx = (precisions >= 0.90).argmax()
threshold_90 = thresholds[idx]
y_pred_90 = (y_scores >= threshold_90)
```

---

## 4. 🔍 Deep dive — the ROC curve and AUC (the topic to nail)

The **ROC (Receiver Operating Characteristic)** curve plots **TPR (recall) on the y-axis**
against **FPR on the x-axis**, sweeping over **all possible thresholds**.

**The two rates:**
- **TPR (recall, sensitivity)** = `TP / (TP + FN)` = fraction of **positives** caught.
- **FPR (fall-out)** = `FP / (FP + TN)` = fraction of **negatives** wrongly flagged positive
  = **1 − specificity**.
- **TNR (specificity)** = `TN / (TN + FP)` = fraction of negatives correctly identified.
- So the ROC curve plots **sensitivity vs (1 − specificity)**.

**How to read the plot (exam skill):**
- Each point corresponds to one **threshold**. The **top-left corner (TPR=1, FPR=0)** is the
  perfect classifier — catches all positives with zero false alarms.
- The **diagonal dotted line** (TPR = FPR) is a **purely random** classifier.
- A good classifier's curve **bows toward the top-left**, staying as far above the diagonal
  as possible.
- Moving **along** the curve = changing the threshold: lowering it moves you up-and-right
  (more TP **and** more FP); raising it moves you down-and-left.
- The **trade-off**: higher TPR (recall) always costs more FPR (false alarms).

**AUC (Area Under the ROC Curve):**
- **AUC = 1.0** → perfect; **AUC = 0.5** → random (the diagonal); below 0.5 → worse than
  random (predictions inverted).
- Interpretation: AUC = the probability that the classifier ranks a **random positive**
  higher than a **random negative**. It is **threshold-independent**, so it summarises
  ranking quality across all thresholds in one number — handy for comparing models.

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores)        # 0.96
```

### ROC vs Precision/Recall (PR) curve — which to use, and why
- Prefer the **PR curve** when the **positive class is rare**, or when you **care more about
  false positives** than false negatives.
- Otherwise use the **ROC curve**.
- **Why it matters:** on skewed data the ROC/AUC can look *deceptively good*, because there
  are huge numbers of easy negatives (large TN) that keep FPR low even with many FP. The PR
  curve has **no TN term** (precision = TP/(TP+FP), recall = TP/(TP+FN)), so it exposes the
  remaining weakness. Rule of thumb: rare positive → look at the PR curve.

```python
# RandomForestClassifier has no decision_function but has predict_proba
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42)
y_probas = cross_val_predict(forest, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas[:, 1]          # P(positive) as the score
```
The random forest beats the SGD classifier on both the PR curve and AUC (~0.998). (These are
*estimated* probabilities — `sklearn.calibration` can calibrate them toward true ones.)

> **scikit-learn rule:** every classifier has either `decision_function()` or
> `predict_proba()` (sometimes both). Use whichever exists as the score for ROC/PR curves.

---

## 5. Multiclass classification (> 2 classes)

Some classifiers are **natively multiclass** (`LogisticRegression`, `RandomForestClassifier`,
`GaussianNB`, naive Bayes, SGD). Strictly binary classifiers (`SVC`) are extended via:

- **One-versus-the-rest (OvR / OvA)** — train **N** binary classifiers, one per class
  (0-detector, 1-detector, …). Predict the class whose classifier gives the **highest
  score**. *Default for most algorithms.*
- **One-versus-one (OvO)** — train one classifier per **pair**: **N·(N−1)/2** classifiers
  (45 for MNIST's 10 classes). Predict the class winning the most **duels**. Each classifier
  trains only on its two classes' data → **small training sets**.

### 🔍 Deep dive — choosing OvR vs OvO (sample-exam Q5c)
- For algorithms that **scale poorly with training-set size** (notably **SVMs**, whose
  training is ~O(m² to m³)), **OvO is preferred**: many classifiers on small subsets is
  faster than a few classifiers on the huge full set.
- For most other algorithms, **OvR is preferred** (fewer classifiers to train and store).
- scikit-learn auto-selects (OvO for `SVC`, OvR for SGD); force either with
  `OneVsOneClassifier` / `OneVsRestClassifier`.

Simply **scaling inputs** lifts SGD's MNIST accuracy from ~85.8% to ~89.1%:
```python
from sklearn.preprocessing import StandardScaler
X_train_scaled = StandardScaler().fit_transform(X_train.astype("float64"))
```

---

## 6. Error analysis (reading a confusion-matrix plot)

```python
from sklearn.metrics import ConfusionMatrixDisplay
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
```

How to read it:
- **Normalise by row** (`normalize="true"`) so a class with fewer instances isn't misread as
  "more errors" — each row then shows *what fraction of that true class went where*.
- Confusion matrices are **generally not symmetric** (many digits → misread as 8, but few 8s
  → misread back).
- To make errors pop, zero-weight correct predictions; you can normalise by row **or** column
  (`normalize="pred"`). **Interpret carefully:** with errors-only-by-row, "36% in row 7,
  col 9" means *36% of the errors made on 7s were 9s* — **not** 36% of all 7s.
- Insights → fixes: more data for confusable classes; engineered features (count closed
  loops: 8→2, 6→1, 5→0); preprocessing (centre/de-rotate); **data augmentation** (add
  shifted/rotated copies so the model tolerates such variation). `SGDClassifier` is *linear*
  (a weight per pixel per class), so it confuses 3s and 5s, which differ in only a few pixels.

---

## 7. Multilabel & multioutput classification

- **Multilabel** — output **multiple binary labels** per instance. Face recognition: a photo
  of Alice & Charlie → `[True, False, True]`. `KNeighborsClassifier` supports it natively.
  Evaluate e.g. by averaging F₁ across labels: `average="macro"` (labels equal) or
  `average="weighted"` (each label weighted by its **support** = #instances with that label).
  - For classifiers lacking native support (e.g. `SVC`): train one model per label; to
    capture **label dependencies**, use a **`ClassifierChain`** — each model also receives
    earlier models' predictions as input.
- **Multioutput (multioutput-multiclass)** — generalisation where each label is itself
  **multiclass**. Example: image denoising — input a noisy digit, output a clean 784-pixel
  image where each pixel-label takes 0–255. Here classification and regression blur.

### 🔍 Deep dive — multiclass vs multilabel (sample-exam Q3b)
- **Multiclass:** exactly **one** label per instance, **> 2** mutually-exclusive classes.
  *Example:* digit recognition — features = pixel intensities; response ∈ {0,…,9}.
- **Multilabel:** **several** labels per instance at once (not mutually exclusive).
  *Example:* tagging which people appear in a photo — features = image pixels; response = a
  set of binary tags {Alice?, Bob?, Charlie?}.

---

## 8. k-Nearest Neighbours (k-NN) — instance-based classification

> Taught in this unit beyond Géron. The sample exam (Q2) needs a manual k-NN computation with
> Euclidean distance on a bag-of-words representation; Q3a asks for a k-NN hyperparameter.
> Master both the concept and the arithmetic.

### What it is
k-NN is the canonical **instance-based** (a.k.a. **lazy**, **non-parametric**) algorithm.
There is essentially **no training** — the model just **stores the whole training set**. All
work happens at prediction time:

1. Compute the **distance** from the query **x** to every stored training instance.
2. Find the **k nearest** neighbours.
3. **Classification** → predict the **majority class** among them (a vote).
   **Regression** → predict their **mean** (or distance-weighted mean) target value.

It is the literal embodiment of *learning by heart + a similarity measure* (Ch. 1). The
textbook notes k-NN works well on MNIST — tune `n_neighbors` and `weights` to exceed 97%.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn.fit(X_train, y_train)
knn.predict(X_new)
```

### Distance metrics (the "nearest" in k-NN)
For instances **xᵢ**, **xⱼ** with *n* features:
- **Euclidean (ℓ₂)** — the sample-exam metric:
  `D(xᵢ, xⱼ) = sqrt( Σ_{ℓ=1..n} ( xᵢ[ℓ] − xⱼ[ℓ] )² )`
- **Manhattan (ℓ₁):** `Σ |xᵢ[ℓ] − xⱼ[ℓ]|`.
- **Minkowski (ℓ_p):** `( Σ |xᵢ[ℓ] − xⱼ[ℓ]|^p )^{1/p}` — p=2 Euclidean, p=1 Manhattan.
  scikit-learn default = Minkowski, p=2.

### 🔍 Deep dive — choosing k as a bias/variance trade-off
k is the key **hyperparameter** (the answer to "give a k-NN hyperparameter").
- **Small k (k=1):** very **flexible**, jagged boundary, **low bias / high variance** —
  fits training perfectly (1-NN has 0 training error) but is noise/outlier-sensitive →
  **overfits**.
- **Large k:** **smoother** boundary, **high bias / low variance** — robust to noise but can
  **underfit** and blur real boundaries; at k = N it always predicts the global majority.
- Choose k by **cross-validation**; an **odd** k avoids ties in binary voting.
- **Weighting:** `weights="uniform"` (equal votes) vs `weights="distance"` (closer neighbours
  count more) — distance-weighting reduces the influence of far neighbours and ties.

### Why feature scaling is essential for k-NN
Distance is dominated by **large-range** features. If one feature spans thousands and another
spans [0,1], the big one swamps the distance and the small one is ignored. **Always
standardise/normalise before k-NN** so features contribute comparably.

### Curse of dimensionality
In **high dimensions**, distances between points become nearly uniform — every point is
"far," and "nearest" loses meaning — so k-NN degrades and needs exponentially more data.
Mitigate with **dimensionality reduction** (Ch. 8) or feature selection.

### Pros & cons (exam-ready)
**Pros:** simple; no training cost; naturally **multiclass** and **multilabel**; flexible
non-linear boundary; no model assumptions (non-parametric).
**Cons:** **slow & memory-heavy at prediction** (compares to all stored instances); very
sensitive to **feature scaling**, **irrelevant features**, and the **curse of
dimensionality**; choice of k and metric matters.

### 🔍 Worked example — k-NN bag-of-words (mirrors sample-exam Q2)
**Bag-of-words:** one feature per dictionary word = its count.
Dictionary order: **[Money, Free, For, Gambling, Fun, Machine, Learning]**.

| i | Money | Free | For | Gambling | Fun | Machine | Learning | Spam |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| 2 | 1 | 2 | 1 | 1 | 1 | 0 | 0 | True |
| 3 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | True |
| 4 | 0 | 0 | 0 | 0 | 3 | 1 | 1 | False |
| 5 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | False |

**(a)** Query "machine learning for free" → **q = [0, 1, 1, 0, 0, 1, 1]**.

**(b) k = 1** — squared Euclidean distances (rank by these; report √ at the end):
- #1: 9+1+1+0+0+1+1 = **13** → √13 ≈ 3.61
- #2: 1+1+0+1+1+1+1 = **6** → √6 ≈ 2.45
- #3: 0+1+0+1+1+1+1 = **4** → √4 = **2.00**
- #4: 0+1+1+0+9+0+0 = **11** → √11 ≈ 3.32
- #5: 0+0+1+0+0+0+0 = **1** → √1 = **1.00**

Nearest is **#5 (1.00) → False (genuine / not spam)**.

**(c) k = 3** — three smallest: #5 = 1.00 (False), #3 = 2.00 (True), #2 ≈ 2.45 (True).
Majority vote = **2 True vs 1 False → True (spam)**.

> Exam tips: (1) build the query vector in the **dictionary's column order**; (2) you may
> rank by **squared** distances but show the final √ values; (3) state the vote explicitly;
> (4) note how the prediction **flips from False (k=1) to True (k=3)** — a clean illustration
> of how k changes the decision boundary.

---

## 9. Quick-reference summary

- **Accuracy misleads on skewed data** → use the confusion matrix.
- **Precision = TP/(TP+FP)**, **Recall = TP/(TP+FN)**, **F₁ = harmonic mean** (high only if
  both high). **FPR = FP/(FP+TN) = 1 − specificity**; **TPR = recall**.
- **Threshold** sets the precision/recall trade-off; you can't maximise both.
- **ROC** = TPR vs FPR; bows to top-left; diagonal = random. **AUC**: 1 perfect, 0.5 random
  = P(rank random positive > random negative).
- **PR curve** for rare positives / costly FPs; **ROC** otherwise (ROC flatters skewed data).
- **OvR** (N classifiers) vs **OvO** (N(N−1)/2, small training sets, preferred for SVMs).
- **Multiclass** = one of many; **multilabel** = many binary tags; **multioutput** =
  multilabel where each label is multiclass.
- **k-NN**: instance-based/lazy; stores data; majority vote (classification) or mean
  (regression) over **k nearest** by a distance metric (Euclidean = ℓ₂). **k** is the
  hyperparameter (small = overfit/low-bias-high-variance; large = underfit/high-bias-low-
  variance). **Must scale features**; suffers the **curse of dimensionality**.
