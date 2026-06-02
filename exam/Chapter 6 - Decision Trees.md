# Chapter 6 — Decision Trees

> **Study booklet — CITS5508.** Decision trees do classification, regression, and
> multioutput tasks, and are the building block of **random forests** (Ch. 7). Exam-relevant:
> the CART algorithm, **Gini vs entropy** (with computations), regularisation
> hyperparameters (Decision-Tree hyperparameter is asked in sample-exam Q3a), white-box
> interpretability, and the **high-variance** limitation that motivates ensembles.

---

## 1. Training, visualising & making predictions

```python
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)            # features: petal length, petal width
```

**How a tree predicts** — start at the **root node** (depth 0) and answer yes/no questions,
walking down until you hit a **leaf node**:
- Root: "petal length ≤ 2.45 cm?" → if yes, go left to a leaf → predict *Iris setosa*.
- If no, go right to a **split node**: "petal width ≤ 1.75 cm?" → yes → *versicolor*; no →
  *virginica*.

**Node attributes** (shown in the visualisation):
- `samples` — how many training instances reach this node.
- `value` — count of those instances per class, e.g. `[0, 1, 45]`.
- `gini` — the node's **Gini impurity** (0 = pure = all one class).

> **Big advantage:** trees need **almost no data preparation** — **no feature scaling or
> centring** required. (Contrast with SVMs / k-NN / GD, which all need scaling.)

scikit-learn uses **CART**, which produces **binary trees** (every split has exactly two
children). Other algorithms (e.g. ID3) allow more than two children per node.

---

## 2. 🔍 Deep dive — Gini impurity vs Entropy

Both measure a node's **impurity** (how mixed its classes are); the tree splits to reduce it.

**Gini impurity** of node *i*:  `Gᵢ = 1 − Σₖ pᵢ,ₖ²`  where pᵢ,ₖ = ratio of class *k* in node
*i*.
- Worked: a node with class counts `[0, 49, 5]` (total 54):
  `G = 1 − (0/54)² − (49/54)² − (5/54)² ≈ 1 − 0 − 0.823 − 0.0086 ≈ 0.168`.
- A **pure** node (all one class) has G = 0.

**Entropy** of node *i*:  `Hᵢ = − Σₖ (pᵢ,ₖ ≠ 0) pᵢ,ₖ · log₂(pᵢ,ₖ)`.
- Same node `[0, 49, 5]`: `H = −(49/54)log₂(49/54) − (5/54)log₂(5/54) ≈ 0.445`.
- Entropy = 0 when the node contains a single class (from information theory: zero "disorder"
  / zero average information).

**Which to use?**
- Usually **little difference** → similar trees. **Gini is slightly faster** (no logarithm) →
  the **default**.
- When they differ: **Gini** tends to **isolate the most frequent class** in its own branch;
  **entropy** tends to produce **slightly more balanced** trees.

---

## 3. White box vs black box; class probabilities

- Decision trees are **white-box** models — intuitive, decisions easy to interpret, rules
  can even be applied by hand. Random forests and neural networks are **black-box** — great
  predictions, but hard to explain *why* (which feature drove a prediction). **Interpretable
  ML** matters for fairness, debugging, trust.
- **Class probabilities:** traverse to the leaf, return the **ratio of each class** among the
  leaf's training instances. E.g. leaf `[0, 49, 5]` → `[0%, 90.7%, 9.3%]` → predict the
  argmax (versicolor). **Caveat:** the probability is **identical for every point in that
  leaf's region** — a tree can't distinguish positions within a leaf.

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)   # array([[0.0, 0.907, 0.093]])
```

---

## 4. The CART training algorithm

**CART (Classification And Regression Tree)** is **greedy** and recursive:
1. Search for the single feature *k* and threshold *tₖ* (e.g. "petal length ≤ 2.45") that
   splits the node into the **two purest subsets, weighted by their size**.
2. Recurse on each subset, then sub-subsets, …
3. Stop at `max_depth`, when no split reduces impurity, or when other stopping
   hyperparameters trigger.

**Classification cost minimised at each split:**
`J(k, tₖ) = (m_left/m)·G_left + (m_right/m)·G_right`  (impurity of children, size-weighted).

- **Greedy** → optimal at the top, then locally optimal below; doesn't look ahead → "good but
  not guaranteed optimal." Finding the *optimal* tree is **NP-complete** (O(exp(m))), so we
  settle for the greedy solution.

**Computational complexity:**
- **Prediction:** traverse root→leaf ≈ **O(log₂ m)** nodes, checking one feature each →
  fast, **independent of #features**.
- **Training:** compare all features on all samples at each node → **O(n · m · log₂ m)**.

---

## 5. 🔍 Deep dive — Regularisation (parametric vs nonparametric)

A decision tree is a **nonparametric model**: the number of parameters is **not fixed before
training**, so the structure can stick arbitrarily close to the data → **prone to
overfitting** if unconstrained. (A **parametric** model like linear regression has a fixed
parameter count → limited freedom → less overfitting, more underfitting risk.)

**Regularise by restricting the tree's freedom** — the Decision-Tree hyperparameters
(sample-exam Q3a answer = `max_depth`, among others):

| Hyperparameter | Effect | Direction to regularise |
|---|---|---|
| `max_depth` | Max tree depth (default None = unlimited) | **decrease** |
| `max_leaf_nodes` | Max number of leaves | decrease |
| `max_features` | Features evaluated per split | decrease |
| `min_samples_split` | Min samples a node needs to split | increase |
| `min_samples_leaf` | Min samples in a leaf | increase |
| `min_weight_fraction_leaf` | Same as above, as a weighted fraction | increase |

> **Rule:** increasing `min_*` or decreasing `max_*` **regularises** (reduces overfitting).

**Pruning** (alternative approach): grow the full tree, then delete nodes whose purity gain
isn't **statistically significant** (e.g. a **χ² test**: if the *p-value* that the gain is due
to chance exceeds a threshold like 5%, prune the children).

```python
# Unregularised vs regularised on the moons dataset
DecisionTreeClassifier(random_state=42)                       # overfits
DecisionTreeClassifier(min_samples_leaf=5, random_state=42)   # generalises better
# test accuracy: ~0.898 vs ~0.92
```

---

## 6. Regression trees

Same CART idea, but each **leaf predicts a value** = the **average target** of its training
instances; splits **minimise MSE** instead of impurity:
`J(k, tₖ) = (m_left/m)·MSE_left + (m_right/m)·MSE_right`.

```python
from sklearn.tree import DecisionTreeRegressor
DecisionTreeRegressor(max_depth=2, random_state=42).fit(X_quad, y_quad)
```
Predictions are **piecewise-constant** (a step per region). Regression trees also overfit
without regularisation — e.g. `min_samples_leaf=10` gives a much more reasonable fit.

---

## 7. Limitations (why we need random forests)

### Sensitivity to axis orientation
All CART splits are **orthogonal** (perpendicular to an axis). So a simple linearly-separable
dataset is split cleanly when axis-aligned, but after a **45° rotation** the boundary becomes
needlessly convoluted and **generalises worse**. Mitigation: **scale + PCA** to rotate the
data toward axis-aligned structure (Ch. 8).

### 🔍 High variance (the key motivation for ensembles)
Decision trees have **high variance**: small changes in hyperparameters or data → very
different trees. Because scikit-learn's training is **stochastic** (it randomly selects
features to evaluate per node), even **retraining on the exact same data** can yield a very
different tree (unless `random_state` is fixed).

**Fix:** **average predictions over many trees** → variance drops sharply. An ensemble of
trees = a **random forest** (Ch. 7), one of the most powerful models available.

---

## 8. Quick-reference summary

- **White-box**, interpretable; **no feature scaling needed**; CART → **binary** trees.
- **Predict** by walking root→leaf; class probability = class ratios in the leaf (constant
  across the leaf's region).
- **Impurity**: **Gini** `1 − Σpₖ²` (faster, default, isolates frequent class) vs **Entropy**
  `−Σpₖlog₂pₖ` (more balanced trees); usually similar.
- **CART** = greedy split minimising size-weighted child impurity (MSE for regression);
  finding the optimal tree is NP-complete.
- **Complexity**: predict **O(log₂ m)**; train **O(n·m·log₂ m)**.
- **Nonparametric** → overfits unless regularised; regularise via `max_depth`↓,
  `min_samples_leaf`↑, etc., or **pruning** (χ² test).
- **Limitations**: orthogonal boundaries (rotation-sensitive, fix with PCA) and **high
  variance** → averaging many trees gives a **random forest**.
