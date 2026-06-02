# Chapter 1 — The Machine Learning Landscape

> **Study booklet — CITS5508.** This is the conceptual foundation of the entire unit. The
> exam mines it heavily for "explain/define/compare" questions (supervised vs unsupervised,
> parameter vs hyperparameter, overfit/underfit, bias/variance, test/validation/train-dev
> sets). Every definition here is potential free marks, and the ideas reappear in every
> later chapter. Read for *understanding the why*, not just memorising.

---

## 1. What machine learning is (and isn't)

**Machine learning (ML)** = the science and art of programming computers so they can
**learn from data** instead of following explicitly hand-coded rules.

Two canonical definitions:

- **Arthur Samuel (1959):** the field of study that gives computers the ability to learn
  without being explicitly programmed.
- **Tom Mitchell (1997) — the engineering definition (memorise this):** *a computer program
  is said to learn from experience **E** with respect to some task **T** and some performance
  measure **P**, if its performance on **T**, as measured by **P**, improves with experience
  **E**.*

**Worked framing — spam filter:** **T** = flag new emails as spam/ham; **E** = the training
data (example emails users have labelled); **P** = e.g. **accuracy** = ratio of correctly
classified emails.

**Why the Mitchell framing matters:** it forces you to name three things before building
anything. If you can't state T, E and P you cannot evaluate or improve a system. Most real
ML failures are really "we optimised the wrong P" or "E didn't match the deployment data."

**Counter-example (the test of understanding):** downloading all of Wikipedia gives a
computer *more data* but is **not** learning — its performance on any task **T** has not
improved. **Data ≠ learning.**

### Core vocabulary (used unit-wide)
| Term | Meaning |
|---|---|
| Training set | The examples the system learns from |
| Training instance / sample | One example in the training set |
| Model | The part that learns patterns & predicts (neural net, random forest, linear model…) |
| Feature / attribute / predictor | An input variable (mileage, GPA…) |
| Label / target | The desired output. *Label* → classification; *target* → regression |
| Accuracy | Ratio of correct predictions (common classification metric) |

---

## 2. Why use ML? (four situations where it wins)

1. **Problems needing long lists of hand-tuned rules** (e.g. spam) — ML shortens code and is
   usually more accurate.
2. **Complex problems with no good algorithmic solution** (e.g. speech recognition) — learn
   from many examples instead.
3. **Fluctuating environments** — an ML system can be *retrained* and stay current.
   *Adaptation example:* spammers switch "4U" → "For U"; an ML filter notices the new
   frequent phrase and adapts **automatically**; a rule-based filter needs a human to write
   a new rule every time.
4. **Getting insight from large/complex data** — inspecting a trained model reveals which
   patterns matter (**data mining**).

---

## 3. The three classification axes of ML systems

Systems are categorised on **three independent axes** — you can freely combine them. A
modern spam filter can be *online + model-based + supervised* simultaneously.

### Axis A — Supervision during training

| Type | Training data | Typical tasks |
|---|---|---|
| **Supervised** | Labelled (inputs **+** desired outputs) | Classification, regression |
| **Unsupervised** | Unlabelled | Clustering, dimensionality reduction, anomaly detection, association rule learning |
| **Semi-supervised** | Mostly unlabelled + a few labels | Combination of the two (Google Photos) |
| **Self-supervised** | Unlabelled → *generates* its own labels | Same tasks as supervised, after pretraining |
| **Reinforcement** | Agent + rewards/penalties | Learn a *policy* maximising long-term reward |

**Supervised learning.** Training set includes the desired solutions (labels).
- **Classification** → predict a discrete class (spam/ham).
- **Regression** → predict a numeric **target** from features (car price from mileage, age…).
- **Key subtlety:** *logistic regression* is used for **classification** — it outputs a
  *probability* of belonging to a class (e.g. 20% spam), despite "regression" in its name.

**Unsupervised learning.** No labels; finds structure by itself.
- **Clustering** — group similar instances (blog-visitor segments). *Hierarchical
  clustering* further subdivides groups into sub-groups.
- **Visualization** — output a 2D/3D map of complex data that preserves structure
  (e.g. **t-SNE** keeps distinct clusters from overlapping) so you can plot and inspect it.
- **Dimensionality reduction** — simplify data without losing much information, often by
  merging correlated features (**feature extraction**, e.g. merge a car's mileage & age into
  a single "wear & tear" feature). Doing this *before* a supervised algorithm makes it run
  faster, use less memory, and sometimes perform better.
- **Anomaly detection** — trained mostly on normal instances; flags abnormal new instances
  (credit-card fraud, manufacturing defects, outlier removal).
- **Novelty detection** — detect instances *unseen* in a very clean training set.
  *Distinction from anomaly detection:* anomaly detection tolerates a few outliers in
  training and may flag rare-but-present cases (e.g. 1% Chihuahuas) as anomalies; novelty
  detection needs a training set with **none** of the target thing, so it would **not** flag
  a new Chihuahua as novel.
- **Association rule learning** — discover relations between attributes (buyers of barbecue
  sauce + chips also buy steak → place them together).

**Semi-supervised learning.** Labelling is costly → many unlabelled + few labelled
instances. Usually = unsupervised step (cluster) + supervised step (propagate labels, then
train). **Google Photos:** clusters the same face across photos (unsupervised); you supply
one label per person (supervised); it then names everyone.

**Self-supervised learning.** Generate a fully labelled dataset *from* an unlabelled one,
then apply supervised learning. E.g. mask part of each image and train a model to
reconstruct it (input = masked image, label = original). The pretrained model is usually
then **transfer-learned / fine-tuned** onto the real task. It uses generated labels
(≈ supervised) but starts unlabelled — treat it as its own category.

**Reinforcement learning (RL).** An **agent** observes an **environment**, takes
**actions**, receives **rewards/penalties**, and learns a **policy** (situation → action)
maximising long-term reward. *AlphaGo* learned its policy from millions of games + self-play;
during the championship, learning was **off** — it just applied the fixed policy
(**offline** use).

### Axis B — Batch vs online learning

**Batch (offline) learning.** Trained on **all** data at once, then deployed *without*
further learning. To add new data → retrain a new version from scratch on the **full**
dataset (old + new) and swap it in.
- *Downside:* deployed accuracy slowly decays as the world changes — **model rot** / **data
  drift**. Fix: retrain regularly. Full retraining is slow, resource-hungry (CPU/RAM/disk/IO),
  costly if daily, and infeasible for huge data or resource-limited devices (phone, Mars
  rover).

**Online (incremental) learning.** Trained incrementally on instances fed sequentially,
individually or in **mini-batches**. Each step is fast & cheap.
- Good for: fast-changing systems (stock market), limited compute, and **out-of-core
  learning** (data too big for memory — load a chunk, train, repeat).
- **Learning rate** = how fast it adapts to new data. *High* → adapts fast but forgets old
  data and is noise-sensitive; *low* → more inertia, learns slowly but is robust to noise.
- *Risk:* bad/garbage data degrades a live system fast → monitor, switch learning off on a
  drop, screen inputs (e.g. anomaly detection).
- ("Online" is a confusing name — out-of-core is often done offline; think **incremental
  learning**.)

### Axis C — Instance-based vs model-based learning (how they generalise)

The whole point of learning is to **generalise** to instances never seen in training. Good
training performance is necessary but **not sufficient**.

- **Instance-based learning** — "learn by heart." Store the training examples; classify a
  new instance using a **similarity measure** to stored examples. E.g. flag an email as
  spam if it shares many words with known spam. **k-Nearest Neighbours (k-NN)** is the
  canonical instance-based algorithm (covered in Ch. 3).
- **Model-based learning** — build a *model* (with parameters) of the examples, then predict
  with it. E.g. `life_satisfaction = θ₀ + θ₁ × GDP_per_capita`.

**The model-based workflow (spine of the unit — memorise):**
1. Study the data.
2. **Model selection** = choose the model *type* **and** fully specify its architecture.
3. **Train** — the learning algorithm searches the **parameter space** for the parameter
   values that minimise a **cost function** (or maximise a **utility/fitness function**).
   For linear regression the cost measures distance between predictions and targets.
4. Apply to new cases (**inference**), hoping it generalises.

```python
# Model-based (linear regression)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.predict(X_new)

# Swap to instance-based (k-NN) — note how little changes
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)
model.predict(X_new)
```

> "Model" is overloaded: a *type* (linear regression), a *fully-specified architecture*
> (linear regression, 1 input → 1 output), or the *final trained model* with specific
> parameter values. Be precise in exam answers.

---

## 4. Main challenges — "bad data" and "bad model"

### Bad data
- **Insufficient quantity.** Most algorithms need thousands of examples; image/speech need
  millions. *The Unreasonable Effectiveness of Data* (Banko & Brill 2001; Norvig 2009):
  for complex problems, *very different* algorithms perform almost identically **once given
  enough data** — data can matter more than the algorithm. (Caveat: small/medium datasets
  are still common; extra data is often costly — don't abandon good algorithms.)
- **Nonrepresentative training data.** To generalise well, training data must represent the
  production cases. Two error sources:
  - **Sampling noise** — small samples are nonrepresentative by chance.
  - **Sampling bias** — even large samples are nonrepresentative if the sampling *method* is
    flawed. *Literary Digest* 1936: predicted Landon beats Roosevelt (he lost) because the
    address lists favoured wealthier Republican-leaning voters, plus **nonresponse bias**
    (only ~25% replied).
- **Poor-quality data.** Errors, outliers, noise. Cleaning is worth it. For missing
  features: drop the attribute, drop the instances, fill in (e.g. median), or train with and
  without the feature.
- **Irrelevant features** ("garbage in, garbage out"). Success needs good **feature
  engineering**: *feature selection* (pick useful existing features), *feature extraction*
  (combine into better ones, e.g. dimensionality reduction), and creating new features by
  gathering data.

### Bad model — overfitting & underfitting
- **Overfitting** — great on training data, poor generalisation. Happens when the model is
  **too complex relative to the amount and noisiness** of the data → it fits patterns in the
  *noise*. *Example:* a complex model "learns" that every training country with a *w* in its
  name has satisfaction > 7 (New Zealand, Norway, Sweden, Switzerland) — pure chance, won't
  generalise to Rwanda/Zimbabwe. **Fixes:** simplify the model (fewer parameters/features or
  **constrain** it), gather more data, reduce noise.
- **Underfitting** — model too simple to capture the structure → inaccurate even on training
  data. **Fixes:** more powerful model, better features, fewer constraints (less
  regularisation).

### 🔍 Deep dive — regularisation and degrees of freedom
**Regularisation** = constraining a model to make it simpler and cut overfitting risk.
Intuition on `θ₀ + θ₁x`:
- 2 free parameters → **2 degrees of freedom** (tweak intercept *and* slope).
- Force `θ₁ = 0` → 1 degree of freedom → can only shift up/down → ends near the mean (very
  simple, likely underfits).
- Allow `θ₁` but **keep it small** → effectively *between* 1 and 2 degrees of freedom — a
  middle ground that balances fitting vs generalising.

The amount of regularisation is set by a **hyperparameter** (§6). Too large → an almost-flat
model (slope ≈ 0) that won't overfit but won't find a good solution either.

---

## 5. Testing and validating (high-value exam material)

You can't *hope* a model generalises — you must measure it.

- **Train/test split.** Split into **training set** and **test set** (commonly 80/20; for
  huge data even 1% test can suffice). Train on training, estimate the **generalization
  error** (= **out-of-sample error**) on the test set.
- **Diagnosis:** *low training error but high test error → overfitting.*

### 🔍 Deep dive — why you need a validation set (the test-set trap)
If you tune hyperparameters by repeatedly measuring on the **test set** and picking what's
best *for that set*, you implicitly **fit to the test set**: e.g. measured 5% error but 15%
in production. **The test set must be touched only once, at the very end.**

**Holdout validation.** Carve a **validation set** (a.k.a. *dev set*) out of the training
set. Train candidate models (different hyperparameters) on the reduced training set, pick
the best on the validation set, **retrain the winner on the full training set** (incl.
validation), then evaluate **once** on the test set.
- Validation set too **small** → noisy model selection (you may pick a suboptimal model);
  too **large** → reduced training set is much smaller than the full one ("like picking the
  fastest sprinter to run a marathon").

**Cross-validation (CV).** Use many small validation folds: each candidate is evaluated once
per fold (trained on the rest) and results averaged → a far more reliable estimate.
Trade-off: training time multiplies by the number of folds.

### 🔍 Deep dive — data mismatch and the train-dev set
Sometimes abundant data isn't quite representative of production (web flower photos vs
in-app photos). **Rule:** validation & test sets **must** be as representative of production
as possible (so build them only from real, production-like data).
- Problem: poor validation performance — is it **overfitting** or **data mismatch**?
- **Andrew Ng's train-dev set:** hold out some *training-distribution* data as a train-dev
  set. Train on the training set only, then:
  - poor on **train-dev** → **overfitting** (simplify/regularise, get more/cleaner data);
  - good on train-dev but poor on **dev** → **data mismatch** (preprocess training data to
    look like production, retrain);
  - good on both → finally evaluate on the **test set**.

### No Free Lunch (NFL) theorem
Choosing a model type implicitly makes **assumptions** about the data (a linear model
assumes the data is fundamentally linear + noise). **Wolpert (1996):** with *no* assumptions,
no model is *a priori* better than another. The only way to be sure is to evaluate them all
(impossible) → in practice make reasonable assumptions and try a few reasonable models.

---

## 6. Parameter vs hyperparameter — and the three "spaces" (frequent exam question)

- **Model parameter** — *learned from data* during training (θ₀, θ₁; neural-net weights). It
  defines the trained model and is used to predict.
- **Hyperparameter** — a parameter of the **learning algorithm**, **not** the model. **Set
  before training**, **constant during training**; the algorithm does *not* adjust it.
  Examples: regularisation strength α, k in k-NN, tree `max_depth`, learning rate.

**The "spaces" (sample-exam Q3a):**
- **Feature (input) space** — the space of all possible input feature vectors **x** (one
  axis per feature). A 2-feature problem → a 2D feature space.
- **Parameter space** — the space of all possible parameter vectors **θ**. Training =
  searching parameter space for the **θ** minimising the cost function. More parameters →
  more dimensions → harder search ("a needle in a 300-D haystack").
- **Hyperparameters** — settings of the learning algorithm, fixed before training. *One per
  technique:* Decision Tree → `max_depth`; k-NN → number of neighbours `k`; Lasso →
  regularisation strength α.

---

## 7. Quick-reference summary

- ML = improving at task **T** (measured by **P**) by learning from experience **E**.
- **Three axes:** supervision (supervised/unsupervised/semi/self/RL) × batch-vs-online ×
  instance-based-vs-model-based.
- **Generalisation** is the goal; estimate it with a held-out **test set** (used once).
- **Validation set / cross-validation** for model & hyperparameter selection — never tune
  on the test set.
- **Overfitting** = too complex for the data (→ high variance); **underfitting** = too
  simple (→ high bias). **Regularisation** trades a little training fit for better
  generalisation, controlled by a hyperparameter.
- **Parameter** = learned; **hyperparameter** = set before training.
- **No Free Lunch:** no universally best model — try a few.

### Exam self-test
- Define ML via Mitchell's T/E/P; name four application areas.
- Supervised vs unsupervised, with an example of each *(sample exam Q4e)* → Supervised: spam
  classification (features = word counts, response = spam/ham). Unsupervised: customer
  segmentation by clustering (features = purchase history, **no** response variable).
- Two common supervised tasks (classification, regression); four unsupervised tasks
  (clustering, dimensionality reduction, anomaly detection, association rule learning).
- Online learning? Out-of-core learning? Which type relies on a similarity measure?
  (instance-based, e.g. k-NN).
- Model parameter vs hyperparameter; what model-based algorithms search for (parameter
  values minimising a cost function) and how they predict.
- Name four ML challenges; "great on training, poor on new data" → overfitting + three fixes.
- Purpose of test / validation / train-dev set; what goes wrong if you tune on the test set.
