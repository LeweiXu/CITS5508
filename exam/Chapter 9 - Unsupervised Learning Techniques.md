# Chapter 9 — Unsupervised Learning Techniques

> **Study booklet — CITS5508.** Unsupervised learning works on **unlabelled** data (we have
> X, not y) — and most real-world data is unlabelled. Exam-relevant: definition/applications
> of clustering, **choosing k** (elbow & silhouette — both exam-skill plots), **anomaly vs
> novelty detection**, k-means vs DBSCAN vs Gaussian mixtures. Deep dives below on the
> **k-means algorithm**, **silhouette analysis**, **DBSCAN**, and **GMMs/EM**.

---

## 1. The three unsupervised tasks

- **Clustering** — group similar instances into **clusters**. Uses: customer segmentation,
  data analysis, dimensionality reduction (via cluster affinities), feature engineering,
  anomaly detection, semi-supervised learning, search engines, image segmentation.
- **Anomaly / outlier detection** — learn what **normal** looks like, then flag abnormal
  instances (**anomalies/outliers**; normal ones = **inliers**). Uses: fraud, defective
  products, cleaning datasets.
- **Density estimation** — estimate the **probability density function (PDF)** of the process
  that generated the data; low-density points are likely anomalies.

> **No universal definition of a "cluster"** — different algorithms capture different kinds:
> some look for instances around a **centroid**, some for **dense regions** of any shape, some
> are **hierarchical**.

---

## 2. 🔍 Deep dive — k-means

Clusters by finding *k* cluster centres (**centroids**) and assigning each instance to the
nearest one. You must **specify k in advance**.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X)        # cluster index per instance (== kmeans.labels_)
kmeans.cluster_centers_               # the k centroids
kmeans.predict(X_new)                 # assign new instances to nearest centroid
```

**The algorithm (Lloyd's):**
1. Initialise *k* centroids (e.g. random instances).
2. **Assign** each instance to the nearest centroid.
3. **Update** each centroid = mean of its assigned instances.
4. Repeat 2–3 until centroids stop moving.

**Guaranteed to converge** (the mean squared distance to centroids can only decrease), but
**possibly to a local optimum** depending on initialisation. Complexity ~**linear** in *m, k,
n* (when the data is clusterable) → k-means is one of the **fastest** clustering algorithms.

- **Hard clustering** = one cluster per instance; **soft clustering** = a score per cluster
  (e.g. `transform()` returns the distance to each centroid → also a nonlinear
  **dimensionality-reduction** trick: turn *n*-D data into *k*-D affinity vectors).
- The decision boundaries form a **Voronoi tessellation**.

**Centroid initialisation:**
- **inertia** = sum of squared distances from instances to their centroids = k-means' cost.
  scikit-learn runs `n_init` random inits and keeps the **lowest inertia** one.
- **k-means++** (default init) seeds centroids **far apart** → much less likely to hit a bad
  local optimum.
- **MiniBatchKMeans** uses mini-batches → ~3–4× faster, slightly worse inertia; for huge data.

### 🔍 Deep dive — choosing the number of clusters k
- **Inertia is NOT a good selector** — it keeps **decreasing** as *k* rises (more centroids ⇒
  everything closer to one). Don't just minimise it.
- **Elbow method:** plot **inertia vs k**; pick the **elbow** where the drop sharply slows
  (e.g. k=4) — coarse but quick.
- **Silhouette score** (more precise): the mean **silhouette coefficient** over all instances:
  
  **`silhouette coeff = (b − a) / max(a, b)`**
  
  where *a* = mean distance to **other instances in the same cluster** (intra-cluster), *b* =
  mean distance to the **nearest other cluster**. Ranges **−1 to +1**: **+1** = deep inside its
  own cluster; **0** = near a boundary; **−1** = probably in the wrong cluster. Plot
  silhouette score vs k and pick the peak (`silhouette_score(X, kmeans.labels_)`).
- **Silhouette diagram:** one "knife" shape per cluster (height = cluster size, width = sorted
  coefficients). Clusters whose bulk falls **left of** the mean-score dashed line are poor.
  Helps prefer clusters of **similar size** (e.g. choose k=5 over k=4 even if k=4's score is
  marginally higher).

### Limits of k-means
- Must **specify k**; must **run several times** (local optima).
- Performs **poorly on clusters of varying sizes, densities, or non-spherical shapes** (it
  only uses distance to centroid). For elliptical clusters, **Gaussian mixtures** are better.
- **Scale features first** (else stretched clusters).

**Applications shown:** **image (color) segmentation** (cluster pixel colours, replace each
with its cluster mean) and **semi-supervised learning** (below).

### Clustering for semi-supervised learning (label propagation)
With few labels: cluster the data, label the **representative** instance (closest to each
centroid), then **propagate** that label to the whole cluster. On digits, labelling 50
representatives + propagation (dropping the 1% farthest from centroids) reached ~91% accuracy
— matching the fully-labelled model with only ~5 labels/class. **Active learning**
(uncertainty sampling) goes further: a human labels the instances the model is **least
certain** about.

---

## 3. 🔍 Deep dive — DBSCAN (density-based clustering)

Defines clusters as **continuous regions of high density**:
- For each instance, count neighbours within distance **ε** (its **ε-neighbourhood**).
- An instance with ≥ `min_samples` neighbours (incl. itself) is a **core instance** (in a
  dense region).
- All instances in a core instance's neighbourhood join its cluster; chains of neighbouring
  core instances form one cluster.
- An instance that is neither a core instance nor in one's neighbourhood = an **anomaly**
  (label `−1`).

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)
dbscan.labels_                 # -1 = anomaly
```

- **Pros:** finds **any number** of clusters of **arbitrary shape**; **robust to outliers**;
  only two hyperparameters (`eps`, `min_samples`).
- **Cons:** struggles when **densities vary** a lot between clusters; **O(m²·n)** → poor
  scaling. No `predict()` (only `fit_predict`) — to classify new points, train e.g. a
  `KNeighborsClassifier` on the core instances. (See also **HDBSCAN**.)

**Other algorithms (one-liners):** **agglomerative** (bottom-up hierarchy), **BIRCH** (large
data, few features), **mean-shift** (climbs to density maxima; O(m²n)), **affinity
propagation** (instances elect exemplars; no preset k; O(m²)), **spectral clustering**
(embed a similarity matrix, then cluster; captures complex structure).

---

## 4. 🔍 Deep dive — Gaussian Mixture Models (GMMs)

A **probabilistic, generative** model assuming the data was generated from a **mixture of k
Gaussian distributions** with unknown parameters. Each Gaussian = one (typically **ellipsoidal**)
cluster with its own mean, covariance (shape/size/orientation), and **weight**.

```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, n_init=10)   # default n_init=1 — set higher!
gm.fit(X)
gm.weights_, gm.means_, gm.covariances_
gm.predict(X)            # hard clustering
gm.predict_proba(X)      # soft clustering (responsibilities)
gm.sample(6)             # generative: draw new instances
gm.score_samples(X)      # log PDF (density) at each point
```

**Trained by Expectation–Maximization (EM)** — like k-means but with **soft assignments**:
1. **Expectation step:** estimate the probability (**responsibility**) that each instance
   belongs to each cluster, given current parameters.
2. **Maximization step:** update each cluster's weight/mean/covariance using **all** instances,
   weighted by those responsibilities.
Repeat to convergence. Like k-means, EM can hit **local optima** → use `n_init > 1`. EM
generalises k-means: it also learns cluster **size, shape, orientation, and weight**.

- **`covariance_type`** constrains cluster shapes to ease learning: `"spherical"`,
  `"diag"`, `"tied"` (shared covariance), `"full"` (default, unconstrained).
- **Anomaly detection with a GMM:** flag instances in **low-density** regions — pick a density
  **threshold** matching the expected anomaly rate (e.g. flag the bottom 2% of `score_samples`).
  Tuning the threshold is the **precision/recall trade-off**.
- **Choosing k for a GMM:** inertia/silhouette **don't apply** (clusters aren't spherical).
  Instead minimise a **theoretical information criterion** — **BIC** or **AIC** (both penalise
  #parameters and reward fit; BIC penalises complexity more). Or use
  **`BayesianGaussianMixture`**, which sets unnecessary clusters' weights to ~0 automatically
  (just set `n_components` higher than you need).
- **Limitation:** GMMs assume **ellipsoidal** clusters → fail on shapes like the moons (they
  fit ellipsoids, not crescents).

---

## 5. 🔍 Deep dive — Anomaly vs Novelty detection (sample-exam-relevant)

- **Anomaly (outlier) detection:** the training set **may be contaminated** with outliers; the
  goal is to detect (and often remove) them. The algorithm must be **robust** to outliers in
  training.
- **Novelty detection:** the training set is assumed **clean** (no outliers); the goal is to
  detect **new** instances that differ from everything seen in training.

> Key difference: **what you assume about the training set.** Anomaly detection tolerates
> outliers in training (often used to *clean* data); novelty detection assumes a pristine
> training set and flags anything novel afterward.

**Dedicated algorithms:** **Isolation Forest** (random trees isolate outliers in fewer
splits; great in high-D), **Local Outlier Factor (LOF)** (compares an instance's local
density to its neighbours'), **One-class SVM** (separates data from the origin in
feature space; good for **novelty** detection, high-D, but doesn't scale), **EllipticEnvelope**
(Fast-MCD; assumes a single Gaussian of inliers), and **PCA reconstruction error** (anomalies
reconstruct poorly).

---

## 6. Quick-reference summary

- **Clustering** groups unlabelled instances; an instance's "label" = its **cluster index**
  (not a class).
- **k-means:** assign-to-nearest-centroid ↔ update-centroids; needs **k**; **inertia** =
  cost; **k-means++** init; fast; bad on non-spherical/varying clusters; **scale first**.
- **Choosing k:** **elbow** of inertia (coarse) or **silhouette score** `(b−a)/max(a,b)` ∈
  [−1,1] + silhouette diagram (prefers balanced clusters).
- **DBSCAN:** density-based; **core instances** + ε-neighbourhoods; finds arbitrary shapes,
  robust to outliers (labels them `−1`); no `predict`; O(m²n).
- **GMM:** mixture of *k* Gaussians, trained by **EM** (soft assignments / responsibilities);
  generative; great for **ellipsoidal** clusters and **density-based anomaly detection**;
  choose k via **BIC/AIC** or `BayesianGaussianMixture`.
- **Anomaly vs novelty:** anomaly detection allows a **contaminated** training set; novelty
  detection assumes a **clean** one.
