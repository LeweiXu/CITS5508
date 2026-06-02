# CITS5508 Practice Exam — Chapter 9 — ANSWER KEY

---

## Question 1.

**(a)** *(3 marks)* **Clustering** = the unsupervised task of grouping **similar instances**
into clusters. In **classification** a label is the **true target class** (supplied during
training); in **clustering** an instance's "label" is just the **index of the cluster** the
algorithm assigned it to (discovered, not a ground-truth class).

**(b)** *(4 marks)* Any four: customer segmentation; data analysis; dimensionality reduction
(cluster affinities); feature engineering; anomaly detection; semi-supervised learning; search
engines (image similarity); image segmentation.

**(c)** *(5 marks)* **k-means loop:** (1) **assign** each instance to the nearest **centroid**;
(2) **update** each centroid to the **mean** of its assigned instances. Repeat until centroids
stop moving. It is **guaranteed to converge** because the mean squared distance to centroids
**can only decrease** each step and is bounded below by 0. It can converge to a **suboptimal
(local) optimum** depending on the **random centroid initialisation** (mitigated by
`n_init`/k-means++).

---

## Question 2.

**(a)** *(3 marks)* **Inertia** (sum of squared distances to the nearest centroid) **always
decreases as k increases** — more centroids mean every instance is closer to one — so minimising
inertia would just pick the largest k. It cannot, by itself, indicate the right number of
clusters.

**(b)** *(4 marks)*
- **Elbow method:** plot **inertia vs k** and pick the **elbow** — the k after which inertia
  drops only slowly (diminishing returns). Coarse but quick.
- **Silhouette score:** the mean **silhouette coefficient** over all instances; plot it vs k
  and pick the **maximum**. More precise (but more expensive).

**(c)** *(3 marks)* `silhouette = (b − a) / max(a, b) = (2.0 − 0.5) / max(0.5, 2.0) = 1.5 / 2.0 =
`**`0.75`**. A value close to **+1** means the instance is **well inside its own cluster and far
from the nearest other cluster** (a good assignment). The coefficient ranges from **−1 to +1**
(≈0 = on a boundary; <0 = likely in the wrong cluster).

**(d)** *(2 marks)* Any two: clusters with **different sizes**, **different densities**, or
**non-spherical/elliptical shapes**; also if features are **not scaled**.

---

## Question 3.

**(a)** *(5 marks)* **DBSCAN** defines clusters as **continuous regions of high density**.
- **ε-neighbourhood:** the set of instances within distance **ε** of a given instance.
- **Core instance:** an instance with at least `min_samples` instances in its ε-neighbourhood
  (i.e. located in a dense region).
- Instances in a core instance's neighbourhood join its cluster; chains of neighbouring core
  instances form one cluster.
- **Anomaly:** an instance that is **neither a core instance nor in the neighbourhood of one**
  (labelled −1).

**(b)** *(4 marks)* Advantages over k-means (any two): finds clusters of **arbitrary shape**;
**does not require specifying k**; **robust to outliers** (labels them as anomalies). Limitation
(one): struggles when cluster **densities vary** a lot; and/or **scales poorly** (≈ O(m²·n)).

**(c)** *(3 marks)* Train a separate classifier (e.g. a **`KNeighborsClassifier`**) on the
DBSCAN **core instances** and their cluster labels, then use it to predict the cluster of a new
instance (optionally setting a maximum distance so far-away points are labelled anomalies).

---

## Question 4.

**(a)** *(3 marks)* A **GMM** assumes the data was generated from a **mixture of k Gaussian
distributions** with unknown parameters: each instance is drawn from one of the k Gaussians
(chosen by cluster weight), and instances from one Gaussian form an (typically **ellipsoidal**)
cluster with its own mean, covariance (shape/size/orientation), and weight.

**(b)** *(5 marks)* **EM steps:** (1) **Expectation:** estimate the **probability
(responsibility)** that each instance belongs to each cluster, given current parameters;
(2) **Maximization:** update each cluster's weight/mean/covariance using **all** instances,
each weighted by its responsibility. Repeat to convergence. Differences from k-means: EM uses
**soft assignments** (probabilities) rather than hard ones, and it learns not just the cluster
**centres** but also their **size, shape, orientation, and weights** (covariance matrices).

**(c)** *(4 marks)* Use a **theoretical information criterion** — **BIC** and **AIC** (both
penalise the number of parameters and reward fit; minimise them over k). A variant that
auto-selects clusters: **`BayesianGaussianMixture`** (sets unnecessary clusters' weights to ≈0).

---

## Question 5.

**(a)** *(4 marks)* **Anomaly detection** assumes the training set **may be contaminated** with
outliers and aims to detect (and often remove) them — so the method must be **robust to outliers
in training**. **Novelty detection** assumes the training set is **clean** (no outliers) and
aims to flag **new** instances that differ from everything seen during training. The key
difference is the **assumption about the training set's cleanliness**.

**(b)** *(4 marks)* A trained GMM gives a **density** (PDF / `score_samples`) at any point;
instances in **low-density regions** are flagged as anomalies. Choose the **density threshold**
to match the **expected anomaly rate** — e.g. if ~2% of products are defective, set the
threshold at the **2nd percentile** of densities. Too many **false positives** → lower the
threshold; too many **false negatives** → raise it. This tuning is exactly the
**precision/recall trade-off**.

**(c)** *(4 marks)* Any two, e.g.:
- **Isolation Forest** — random trees split the space randomly; anomalies get **isolated in
  fewer splits** (shorter average path) because they're far from other instances.
- **Local Outlier Factor (LOF)** — compares an instance's **local density** to that of its
  neighbours; an anomaly is **much less dense** than its neighbours.
- **One-class SVM** — separates the data from the origin in feature space; points outside the
  learned region are anomalies (good for novelty detection).
- **PCA reconstruction error** — anomalies reconstruct **poorly** (large error).
