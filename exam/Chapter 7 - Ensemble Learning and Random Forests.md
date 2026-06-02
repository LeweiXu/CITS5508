# Chapter 7 — Ensemble Learning and Random Forests

> **Study booklet — CITS5508.** Directly examined: **bagging vs pasting** (sample-exam Q4d),
> **random forests & extra-trees** (Q4a). Deep dives below on *why ensembles work* (the
> wisdom of the crowd / law of large numbers), the bagging/pasting/OOB machinery, and the
> two boosting families (AdaBoost, gradient boosting). The unifying theme: **combine many
> predictors to beat the best single one**, mostly by **reducing variance**.

---

## 1. Why ensembles work — wisdom of the crowd

An **ensemble** = a group of predictors; **ensemble learning** = aggregating their
predictions (e.g. majority vote). Aggregating often beats the best individual predictor.

### 🔍 Deep dive — the law of large numbers
Even a **weak learner** (slightly better than random) can combine into a **strong learner**
(high accuracy), *if* there are enough of them and they are **sufficiently diverse**.

*Biased-coin analogy:* a coin with 51% heads, tossed 1,000 times, gives a heads majority
~75% of the time (≈97% at 10,000 tosses) — the **law of large numbers** drives the head ratio
toward 51%. Likewise 1,000 classifiers each 51% accurate → ~75% ensemble accuracy via majority
vote.

**The crucial caveat:** this assumes the classifiers make **independent, uncorrelated
errors**. In reality they train on the same data and make correlated errors, lowering the
gain. ⇒ **Ensembles work best when predictors are as independent/diverse as possible** — e.g.
train them with **very different algorithms**.

---

## 2. Voting classifiers

- **Hard voting** — each classifier votes; predict the **majority** class. Often beats the
  best member.
- **Soft voting** — average the predicted **class probabilities** and predict the argmax.
  Usually **better than hard voting** because it weights confident votes more. Requires every
  classifier to have `predict_proba()` (for `SVC`, set `probability=True`).

```python
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
], voting='soft')      # 'hard' for majority vote
voting_clf.fit(X_train, y_train)
```

---

## 3. 🔍 Deep dive — Bagging vs Pasting (sample-exam Q4d)

Another route to diversity: use the **same algorithm** but train each predictor on a
**different random subset** of the training instances.

- **Bagging** (**b**ootstrap **agg**regat**ing**) — sampling **with replacement**. The same
  instance can be picked multiple times for the *same* predictor.
- **Pasting** — sampling **without replacement**. An instance is used at most once per
  predictor.
- Both let an instance appear across *different* predictors; only **bagging** allows
  repeats *within* one predictor.

**Aggregation:** statistical **mode** (majority vote) for classification, **average** for
regression.

**Why it works (bias/variance):** each predictor has **higher bias** than one trained on the
full set, but **aggregation reduces variance**. Net result: **similar bias, lower variance**
than a single predictor → better generalisation.

**Bagging vs pasting trade-off:** bagging injects **more diversity** (bootstrap subsets are
more different) → slightly **higher bias** but **less correlated** predictors → **lower
variance**. Bagging is **generally preferred**; cross-validate both if you have time.

**Parallelism:** predictors train and predict **independently** → bagging/pasting **scale
very well** across CPU cores / servers (`n_jobs=-1`).

```python
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=500, max_samples=100,
                            bootstrap=True,         # False → pasting
                            n_jobs=-1, random_state=42)
```

### Out-of-bag (OOB) evaluation
With bagging (sampling *m* with replacement), each predictor sees on average **~63%** of the
instances; the unseen **~37%** are its **out-of-bag** instances. Since each instance is OOB
for several predictors, those predictors can evaluate it → a **free validation set, no
holdout needed**. Set `oob_score=True` → read `oob_score_`.

### Random patches & random subspaces
`BaggingClassifier` can also sample **features** (`max_features`, `bootstrap_features`):
- **Random patches** — sample **both** instances and features (useful for high-dim inputs
  like images; speeds up training).
- **Random subspaces** — keep all instances, sample only **features**.
- Feature sampling adds diversity → trades a bit more bias for lower variance.

---

## 4. 🔍 Deep dive — Random Forests & Extra-Trees (sample-exam Q4a)

**Random forest** = an **ensemble of decision trees**, trained via **bagging** (usually
`max_samples` = full training set size). Use the optimised `RandomForestClassifier` /
`RandomForestRegressor`.

**The extra randomness (what makes it a *forest*, not just bagged trees):** when splitting a
node, instead of searching for the best feature among **all** features, it searches only
within a **random subset** of features (default √n). → more tree diversity → **higher bias,
lower variance** → usually a better model overall.

```python
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
```

**Extra-Trees (Extremely Randomized Trees):** push randomness further — for each candidate
feature, use a **random threshold** rather than searching for the best one.
- Trades **more bias for lower variance**.
- **Much faster to train** (finding the best threshold per feature is the costly step).
- Can't say in advance whether `ExtraTreesClassifier` beats `RandomForestClassifier` →
  cross-validate both. (`ExtraTrees` defaults to `bootstrap=False`.)

> **Exam answer (Q4a):** *"A random forest is an ensemble of decision trees trained by bagging,
> where each tree also considers only a random subset of features at each split. Extra-trees
> add even more randomness by also choosing split thresholds at random instead of optimising
> them — this increases bias slightly but reduces variance and makes training much faster."*

### Feature importance
Random forests measure a feature's importance by **how much it reduces impurity on average**
across all trees (weighted by samples reaching each node), scaled to sum to 1. Read
`feature_importances_`. Handy for quick feature selection (e.g. iris: petal length 44%, petal
width 42% dominate).

---

## 5. Boosting — train predictors sequentially, each fixing the last

**Boosting** combines weak learners into a strong learner by training predictors
**sequentially**, each correcting its predecessor. **Cannot be parallelised** (each needs the
previous one) → scales worse than bagging.

### AdaBoost (adaptive boosting)
- Train a base predictor; **increase the weights of the instances it misclassified**; train
  the next predictor on the reweighted data; repeat. New predictors focus more on the **hard
  cases**.
- Final prediction = **weighted vote**, where each predictor's weight depends on its accuracy.
- Analogy to gradient descent, but instead of tweaking one model's parameters, AdaBoost
  **adds predictors** to the ensemble.
- **If overfitting** → reduce `n_estimators` or regularise the base estimator more.
- Default base = a **decision stump** (`max_depth=1`).

```python
from sklearn.ensemble import AdaBoostClassifier
AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                   n_estimators=30, learning_rate=0.5, random_state=42)
```

### Gradient boosting
- Instead of reweighting instances, each new predictor is fit to the **residual errors** of
  the previous one. Prediction = **sum** of all predictors' outputs.
- **Gradient Boosted Regression Trees (GBRT)** use decision trees as base learners.
- **`learning_rate`** scales each tree's contribution = **shrinkage** (a regularisation): a
  **low** rate needs **more trees** but generalises better.
- **Too few trees → underfit; too many → overfit.** Find the count via **early stopping**
  (`n_iter_no_change`) or CV. `subsample < 1.0` → **stochastic gradient boosting** (more bias,
  less variance, faster).

```python
from sklearn.ensemble import GradientBoostingRegressor
GradientBoostingRegressor(max_depth=2, learning_rate=0.05,
                          n_estimators=500, n_iter_no_change=10, random_state=42)
```

- **Histogram-based gradient boosting (HGB)** — bins features into integers (`max_bins≤255`)
  → **O(b·m)** instead of O(n·m·log m) → hundreds of times faster on big data; supports
  categorical features and missing values. (See also **XGBoost, LightGBM, CatBoost**.)

---

## 6. Stacking (stacked generalization)

Instead of a fixed aggregation rule (hard/soft voting), **train a model to do the
aggregation** — the **blender** (meta-learner). The base predictors' outputs become the
blender's input features.

- **Training:** use `cross_val_predict` to get **out-of-sample** predictions from each base
  predictor; these form the blending training set (one feature per base predictor); the
  blender is trained on them, then base predictors are retrained on the full set.
- Can have **multiple layers** of blenders.

```python
from sklearn.ensemble import StackingClassifier
StackingClassifier(estimators=[...], final_estimator=RandomForestClassifier(), cv=5)
```

---

## 7. Quick-reference summary

- Ensembles beat the best member when learners are **diverse** and make **uncorrelated
  errors** (law of large numbers).
- **Hard voting** (majority) vs **soft voting** (average probabilities — usually better).
- **Bagging** = sample **with** replacement (repeats allowed per predictor); **Pasting** =
  **without** replacement. Both reduce **variance** via aggregation; bagging usually wins and
  gives free **OOB** evaluation (~37% unseen per predictor).
- **Random forest** = bagged trees + **random feature subset per split**. **Extra-trees** add
  **random thresholds** → faster, more bias, less variance. Both give `feature_importances_`.
- **Boosting** = sequential, each predictor fixes the last (**can't parallelise**).
  **AdaBoost** reweights misclassified instances; **gradient boosting** fits residuals
  (`learning_rate` = shrinkage; tune #trees by early stopping).
- **Stacking** trains a **blender** to aggregate base predictions.
- Bagging/RF/boosting mainly cut **variance** and shine on **tabular** data with little
  preprocessing.
