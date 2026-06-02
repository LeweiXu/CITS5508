# Chapter 5 — Support Vector Machines (SVMs)

> **Study booklet — CITS5508.** Sample-exam Q5a asks you to **describe the SVM training
> procedure** (margin, support vectors, which instances define the boundary, decision rule),
> and Q4c asks you to **explain the kernel trick** — both get dedicated deep dives below.
> SVMs are powerful for **small-to-medium nonlinear** datasets but **don't scale** to very
> large ones.

---

## 1. Linear SVM Classification — large margin classification

**The core idea:** instead of just *any* separating line, an SVM fits the **widest possible
"street"** between the two classes. The decision boundary is the centre line; the street's
edges are the **margins**. This is **large margin classification**.

- A boundary that merely separates the classes but passes close to instances will likely
  generalise poorly; the **widest** street stays as far from the nearest instances as
  possible → better generalisation.
- **Support vectors:** the instances sitting **on the edge of the street** (the closest
  points). They alone **"support" / determine** the boundary. **Adding more instances off the
  street does not change the boundary at all** — only support vectors matter.
- **SVMs are sensitive to feature scales** — an unscaled large-range feature makes the street
  nearly flat along that axis. **Always `StandardScaler` first.**

### 🔍 Deep dive — hard vs soft margin (the C hyperparameter)
- **Hard margin:** require **all** instances off the street, on the correct side. Two
  problems: (1) only works if data is **linearly separable**; (2) very **sensitive to
  outliers** (one outlier can make it impossible or wreck the boundary).
- **Soft margin:** allow some **margin violations** (instances inside the street or on the
  wrong side) to get a wider, more robust street. Balance "wide street" vs "few violations".
- The **`C`** hyperparameter sets that balance:
  - **Low C** → more regularisation → **wider** street, **more** violations, **more** support
    vectors → less overfitting (but too low → underfitting).
  - **High C** → narrower street, fewer violations → can overfit.
  - **If overfitting, reduce C.**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)
svm_clf.decision_function(X_new)   # signed distance to boundary (no predict_proba on LinearSVC)
```

> `LinearSVC` has **no `predict_proba`**. `SVC(probability=True)` can produce probabilities,
> but it fits an extra model via 5-fold CV → slow.

---

## 2. Nonlinear SVM Classification

Many datasets aren't linearly separable. Two strategies to handle nonlinearity:

### (a) Add features (polynomial / similarity)
- **Polynomial features:** e.g. a 1-D set that's not separable becomes separable after
  adding `x₂ = x₁²`. Implement with `PolynomialFeatures` → `StandardScaler` → `LinearSVC`.
  But low degree can't fit complex data, and **high degree explodes the feature count** →
  slow.
- **Similarity features:** add features measuring resemblance to **landmarks** via a
  similarity function (e.g. **Gaussian RBF**, a bell curve = 1 at the landmark → 0 far away).
  Put a landmark at **every instance** → very likely separable, but turns *m×n* into *m×m*
  features → huge for large *m*.

### (b) 🔍 Deep dive — the kernel trick (sample-exam Q4c)
**The problem:** adding many (or infinitely many) features to achieve separability is
computationally expensive — sometimes impossible.

**The trick:** certain **kernel functions** `K(a, b)` compute the **dot product of the
transformed vectors** `φ(a)ᵀφ(b)` **directly from the original vectors a, b** — *without ever
computing (or even knowing) the transformation φ*. Because the SVM's training (dual form) and
predictions depend on the data **only through dot products**, you can substitute the kernel
and get **exactly the same result as if you had mapped to the high-dimensional space**, at a
fraction of the cost and with **no combinatorial explosion**.

*Worked intuition:* for the 2-D degree-2 mapping φ, one can show `φ(a)ᵀφ(b) = (aᵀb)²`. So the
"polynomial kernel" `K(a,b) = (aᵀb)²` gives the transformed dot product by just squaring the
original dot product.

**Common kernels:**
| Kernel | Formula |
|---|---|
| Linear | `K(a,b) = aᵀb` |
| Polynomial | `K(a,b) = (γ·aᵀb + r)^d` |
| Gaussian RBF | `K(a,b) = exp(−γ·‖a − b‖²)` |
| Sigmoid | `K(a,b) = tanh(γ·aᵀb + r)` |

- **Mercer's theorem:** if K satisfies Mercer's conditions (continuous, symmetric:
  K(a,b)=K(b,a), …) then a corresponding φ is **guaranteed to exist**, so you can use K even
  without knowing φ. For the **RBF kernel, φ maps to an infinite-dimensional space** — which
  is exactly why you must use the trick rather than the explicit mapping.

**Kernel hyperparameters:**
- **Polynomial `degree`** ↑ if underfitting, ↓ if overfitting; `coef0` controls the
  influence of high- vs low-degree terms.
- **RBF `gamma` (γ)** acts like a **regularisation** knob: **high γ** → narrow bell → small
  influence radius → **wiggly, irregular** boundary (overfit); **low γ** → wide bell →
  smoother boundary (underfit). So: **overfitting → decrease γ (and/or C); underfitting →
  increase γ (and/or C).**

```python
from sklearn.svm import SVC
SVC(kernel="poly", degree=3, coef0=1, C=5)        # polynomial kernel
SVC(kernel="rbf", gamma=5, C=0.001)               # Gaussian RBF kernel
```

**Which kernel?** Try **linear first** (`LinearSVC` is much faster than `SVC(kernel="linear")`
on large data); if the set isn't too large, also try the **Gaussian RBF** (often excellent).

---

## 3. SVM classes & computational complexity

| Class | Time complexity | Out-of-core | Scaling | Kernel trick |
|---|---|---|---|---|
| `LinearSVC` | O(m·n) | No | Yes | No |
| `SVC` | O(m²·n) to O(m³·n) | No | Yes | **Yes** |
| `SGDClassifier` | O(m·n) | **Yes** | Yes | No |

- **`LinearSVC`** (liblinear): fast, scales ~linearly; no kernel trick.
- **`SVC`** (libsvm): supports kernels but **O(m²–m³)** → "dreadfully slow" for large *m* →
  best for small/medium nonlinear sets; scales well with #features (esp. sparse).
- **`SGDClassifier`** does large-margin classification via SGD → out-of-core, low memory,
  scales to data that doesn't fit in RAM.

---

## 4. SVM Regression

Flip the objective: instead of fitting the widest street **between** classes, SVM regression
fits as many instances as possible **inside** a street of width controlled by **ε**, while
limiting points outside it.

- Instances **within** the margin don't affect predictions → the model is **ε-insensitive**.
- **Reducing ε → wider... no:** smaller ε = **narrower** street → **more** support vectors →
  **more regularisation**.
- `LinearSVR` (scales linearly) and `SVR` (kernelised, slow on large *m*) mirror `LinearSVC`
  / `SVC`.

```python
from sklearn.svm import SVR
make_pipeline(StandardScaler(), SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
```

---

## 5. 🔍 Deep dive — how a linear SVM works (sample-exam Q5a answer)

**Decision function & prediction.** A linear SVM computes `wᵀx + b`. Predict the **positive
class if `wᵀx + b ≥ 0`**, else the negative class. The **decision boundary** is the
hyperplane `wᵀx + b = 0`; the **margins** are `wᵀx + b = ±1`.

**Margin width depends on ‖w‖.** The street's width is `2/‖w‖` → to make the margin **wider**
we **minimise ‖w‖** (equivalently ½‖w‖²). The bias *b* only shifts the street, not its width.

**Training objective.**
- **Hard margin:** minimise ½‖w‖² subject to `t⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1` for every instance
  (where t = +1 for positives, −1 for negatives) — i.e. every instance off the street, on its
  correct side.
- **Soft margin:** add a **slack variable ζ⁽ⁱ⁾ ≥ 0** per instance (how much it may violate the
  margin) and minimise `½‖w‖² + C·Σ ζ⁽ⁱ⁾`. **C** trades margin width against violations.
- Both are **convex quadratic programming (QP)** problems → a global optimum. (Alternatively,
  minimise the **hinge loss** `max(0, 1 − t·s)` via gradient descent; `LinearSVC` defaults to
  *squared* hinge, `SGDClassifier` to hinge.)

**Putting it together for the exam:** *"The SVM finds the hyperplane that maximises the margin
(the widest street) between the two classes. The margin is defined by the parallel hyperplanes
`wᵀx + b = ±1`; maximising it means minimising ‖w‖, subject to instances lying on the correct
side (soft margin allows some violations, balanced by C). The only training instances that
matter are the **support vectors** — those on the margin edges (and violators); all others
could be removed without changing the boundary. To classify a new instance, compute
`wᵀx + b` and assign the positive class if it's ≥ 0 (the sign of the decision function), i.e.
which side of the boundary it falls on."*

### The dual problem & why the kernel trick lives there
The **primal** QP can be rewritten as an equivalent **dual** problem (same solution under the
SVM's convexity conditions). The dual is faster when **#instances < #features**, and crucially
it expresses everything via **dot products between instances** — which is **what makes the
kernel trick possible** (the primal does not). Predictions then need dot products of the new
point with **only the support vectors** (since the dual coefficients α⁽ⁱ⁾ are nonzero only for
support vectors).

---

## 6. Quick-reference summary

- SVM = **large-margin** classifier: the **widest street** between classes; boundary fixed
  only by **support vectors** (closest instances).
- **Hard margin** (no violations; needs linearly separable, outlier-sensitive) vs **soft
  margin** (allow violations). **`C`**: low = wider street/more violations/less overfit;
  high = narrower/can overfit. **Overfit → lower C.**
- **Always scale features** (SVMs are scale-sensitive).
- **Kernel trick**: compute `φ(a)ᵀφ(b)` via `K(a,b)` without computing φ → nonlinear SVM with
  no feature explosion. RBF → infinite-dimensional φ. **RBF γ**: high = wiggly/overfit, low =
  smooth/underfit.
- `LinearSVC` (fast, no kernel) vs `SVC` (kernel, O(m²–m³), small/medium data) vs
  `SGDClassifier` (out-of-core).
- **SVM regression**: fit points *inside* an ε-width street; **ε-insensitive**.
- **Training**: maximise margin = minimise ½‖w‖² subject to `t⁽ⁱ⁾(wᵀx⁽ⁱ⁾+b) ≥ 1` (soft margin
  adds slack + C·Σζ); a convex QP; predict by the **sign of `wᵀx + b`**.
