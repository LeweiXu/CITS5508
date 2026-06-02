# Chapter 8 — Dimensionality Reduction

> **Study booklet — CITS5508.** Directly examined: **manifold & manifold learning**
> (sample-exam Q4b). Deep dives below on the **curse of dimensionality**, the two main
> approaches (**projection vs manifold learning**), and **PCA** (the workhorse — how it
> chooses axes, explained variance, choosing dimensions, compression). Also covers random
> projection and LLE.

---

## 1. Why reduce dimensionality?

**Motivations:**
- **Speed up training** (fewer features).
- **Data visualization** — reduce to 2D/3D to plot and spot patterns/clusters, and to
  communicate to non-experts.
- Sometimes **filters noise** → better performance (but usually it just speeds things up).

**Drawbacks:**
- **Information loss** (like JPEG compression) → may slightly hurt performance.
- Adds **pipeline complexity**.
- ⇒ **Try the original data first**; reduce only if needed.

---

## 2. 🔍 Deep dive — The curse of dimensionality

High-dimensional space behaves nothing like our 3-D intuition:
- In a unit square, a random point has ~0.4% chance of being within 0.001 of a border; in a
  10,000-D hypercube, **>99.999999%** of points are near a border.
- Average distance between two random points: ~0.52 (2-D), ~0.66 (3-D), but **~408** in a
  1,000,000-D unit hypercube. **High-dimensional data is extremely sparse** — points are far
  apart, new instances are far from any training instance, so predictions rely on large
  **extrapolations** → **more dimensions ⇒ higher overfitting risk**.
- Reaching a given data density needs instances **exponential** in the dimension count
  (100 features uniformly spread → more instances than atoms in the universe). So "just get
  more data" is infeasible.

---

## 3. 🔍 Deep dive — Two approaches: Projection vs Manifold learning

### Projection
Real data isn't spread uniformly — many features are near-constant or correlated, so
instances lie close to a **lower-dimensional subspace**. **Project** them (perpendicularly)
onto that subspace → new features = coordinates in the subspace (e.g. 3-D data lying near a
plane → project to 2-D).

**Limitation:** projection fails when the subspace **twists** — e.g. the **Swiss roll**.
Dropping a dimension would **squash** different layers together; you really want to **unroll**
it.

### Manifold learning (sample-exam Q4b)
- **Manifold:** a *d*-dimensional shape that is **bent/twisted** inside a higher *n*-D space
  (*d < n*), and **locally resembles a *d*-D hyperplane**. The Swiss roll is a **2-D manifold**
  rolled in 3-D: locally it looks like a 2-D plane.
- **Manifold learning:** dimensionality-reduction techniques that **model the manifold** the
  data lies on, relying on the **manifold assumption (hypothesis)** — *most real-world
  high-dimensional data lies close to a much lower-dimensional manifold* (empirically very
  often true).
  - *Intuition (MNIST):* digit images have heavy constraints (connected lines, white borders,
    roughly centred) — random pixels almost never look like a digit, so the data occupies a
    tiny low-dimensional manifold of the 784-D pixel space.
- **Companion assumption (not always true):** the task becomes **simpler** in the unrolled
  manifold space. Often yes (the Swiss roll's two classes become linearly separable when
  unrolled), but sometimes a boundary that's simple in the original space (e.g. `x₁ = 5`, a
  flat plane) looks **more complex** once unrolled.

> **Exam answer (Q4b):** *"A manifold is a lower-dimensional surface (d-D) that is bent and
> twisted within a higher-dimensional (n-D) space but locally looks like a d-D plane. Manifold
> learning is a family of dimensionality-reduction techniques that try to model this manifold
> — relying on the manifold assumption that real high-dimensional data tends to lie close to a
> much lower-dimensional manifold — so the data can be 'unrolled' into that low-dimensional
> space (e.g. LLE unrolling a Swiss roll)."*

---

## 4. 🔍 Deep dive — PCA (Principal Component Analysis)

The most popular technique. It finds the hyperplane closest to the data and **projects** onto
it.

### Preserving variance
Among candidate axes to project onto, PCA picks the one that **preserves the maximum
variance** — equivalently, the axis that **minimises the mean squared distance** between the
data and its projection (loses the least information).

### Principal components
- **1st PC** = axis of largest variance; **2nd PC** = orthogonal axis capturing the most
  *remaining* variance; and so on (one per dimension). The *i*-th axis is the *i*-th
  **principal component**.
- Found via **Singular Value Decomposition (SVD)**: `X = U Σ Vᵀ`, where **V**'s columns are
  the principal-component unit vectors (in order). **Centre the data first** (scikit-learn's
  `PCA` does this automatically). The PC *directions* (signs) aren't stable, but the **axes**
  are.

### Projecting to d dimensions
Multiply the centred data by the first *d* columns of **V**: `X_d-proj = X · W_d`.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)        # centres the data automatically
pca.components_                   # rows = principal components
```

### Explained variance ratio & choosing the number of dimensions
- `explained_variance_ratio_` = the fraction of variance along each PC (e.g. `[0.76, 0.15]`
  → 76% on PC1, 15% on PC2, ~9% lost if you drop PC3).
- **Choose *d* to preserve a target variance** (e.g. 95%) — set `n_components` to a **float**
  between 0 and 1:

```python
pca = PCA(n_components=0.95)       # keep 95% of variance
X_reduced = pca.fit_transform(X_train)
pca.n_components_                  # e.g. 154 for MNIST (down from 784)
```
- Or plot **cumulative explained variance vs #dimensions** and pick the **elbow**.
- For visualization, force `n_components = 2` or `3`.
- As a **preprocessing step**, tune `n_components` like any hyperparameter (powerful models
  like random forests need fewer dimensions; linear models need more).

### PCA for compression & reconstruction error
- MNIST → 154 features keeps 95% variance at <20% of the original size.
- **Decompress** with `inverse_transform()` (`X_recovered = X_d-proj · W_dᵀ`) — close to the
  original but not exact. The mean squared distance between original and reconstructed data is
  the **reconstruction error** (also used for anomaly detection — Ch. 9).

### PCA variants (when to use which)
- **Randomized PCA** (`svd_solver="randomized"`) — approximate, **O(m·d² + d³)** ≪ full SVD's
  O(m·n² + n³); much faster when *d ≪ n*. (`"auto"` uses it by default for large data.)
- **Incremental PCA** (`IncrementalPCA` + `partial_fit`) — feed **mini-batches**; for data
  that doesn't fit in memory or arrives online.

---

## 5. Random projection

Project onto a **random** lower-dimensional subspace. Surprisingly, the **Johnson–Lindenstrauss
lemma** guarantees distances are preserved fairly well, so similar/different instances stay
similar/different. The minimum safe target dimension *d* depends only on **m** and the
tolerance **ε** (not on **n**) — `johnson_lindenstrauss_min_dim()`. Simple, fast, no training
(only the data's shape is used). `SparseRandomProjection` is usually preferable (less memory,
faster). Good for **very high-dimensional** data where PCA is too slow.

---

## 6. LLE (Locally Linear Embedding)

A **nonlinear, manifold-learning** technique that does **not** use projection. Two steps:
1. For each instance, find its *k* nearest neighbours and the **linear weights** that best
   reconstruct it from them (capturing local relationships).
2. Find a low-dimensional placement of all instances that **preserves those local
   relationships** as well as possible.

Great at **unrolling twisted manifolds** (e.g. the Swiss roll) with low noise. Scales poorly
(an `m²` term) to very large datasets.

**Other techniques (one-liners):** **MDS** preserves pairwise distances; **Isomap** preserves
**geodesic** (graph) distances; **t-SNE** keeps similar instances close / dissimilar apart
(great for **visualizing clusters**); **LDA** is a supervised technique that projects onto the
most class-discriminative axes (good preprocessing before classification).

---

## 7. Quick-reference summary

- Reduce dimensionality to **speed up training** and **visualize**; costs **information loss**
  + complexity → try original data first.
- **Curse of dimensionality:** high-D space is sparse, distances huge, overfitting risk up;
  density needs exponentially more data.
- **Projection** (data near a flat subspace) vs **manifold learning** (data on a twisted
  lower-D **manifold**; relies on the **manifold assumption**).
- **PCA:** project onto axes of **maximum variance** (= min reconstruction error), found via
  **SVD**; pick *d* via **explained variance ratio** (e.g. 95%) or the elbow; supports
  compression (`inverse_transform`), **randomized** and **incremental** variants.
- **Random projection** (JL lemma): fast, distance-preserving, for very high-D data.
- **LLE / Isomap / MDS / t-SNE / LDA**: manifold/embedding techniques; t-SNE for cluster
  visualization, LDA is supervised.
