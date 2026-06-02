# CITS5508 Practice Exam — Chapter 3 — ANSWER KEY

---

## Question 1.

**(a)** *(2 marks)* TP = 40, TN = 80, FP = 20, FN = 10.

**(b)** *(6 marks)*
- **Accuracy** = (TP + TN) / total = (40 + 80) / 150 = 120/150 = **0.80 (80%)**.
- **Precision** = TP / (TP + FP) = 40 / (40 + 20) = 40/60 = **0.667 (66.7%)**.
- **Recall** = TP / (TP + FN) = 40 / (40 + 10) = 40/50 = **0.80 (80%)**.
- **F₁** = 2·(P·R)/(P+R) = 2·(0.667·0.80)/(0.667+0.80) = 2·0.5333/1.4667 = **0.727**.
*(1½ marks each.)*

**(c)** *(4 marks)* On a **skewed/imbalanced** dataset, a model that simply predicts the
**majority (negative) class** can score high accuracy without detecting any positives — so
accuracy hides poor positive-class performance. Report metrics derived from the **confusion
matrix**: **precision**, **recall**, **F₁**, and look at the **PR curve** (positive class is
rare). *(Here recall = 80% is fine, but precision = 67% shows real weakness that accuracy
masks.)*

---

## Question 2.

**(a)** *(4 marks)* The classifier scores each instance via a **decision function** and
predicts positive if the score exceeds a **threshold**. **Raising** the threshold makes fewer
positive predictions → **precision tends to increase** (the remaining positives are more
confidently correct) and **recall decreases** (more true positives are missed). You cannot
maximise both at once — improving one generally worsens the other.

**(b)** *(4 marks)*
1. Kid-safe video filter → **high precision** (better to wrongly reject some safe videos than
   to let one unsafe video through; a false positive "safe" label is costly).
2. Tumour screening → **high recall** (must catch nearly all true tumours; missing one
   (false negative) is dangerous, while false alarms can be re-checked).
*(2 marks each: correct choice + justification.)*

**(c)** *(4 marks)* As the threshold rises one instance at a time, the **recall** denominator
(total actual positives) is fixed and TP can only drop or stay → recall is **monotonically
non-increasing** (smooth). **Precision** = TP/(TP+FP); removing a positive prediction can
remove either a TP or an FP, which can move the TP/(TP+FP) ratio **up or down** depending on
which it was — so precision is **bumpy / not strictly monotonic**.

---

## Question 3.

**(a)** *(4 marks)* The ROC curve plots, over **all thresholds**:
- **y-axis: True Positive Rate (TPR = recall) = TP/(TP+FN)** — fraction of positives caught.
- **x-axis: False Positive Rate (FPR) = FP/(FP+TN) = 1 − specificity** — fraction of negatives
  wrongly flagged positive.
Each point corresponds to one threshold.

**(b)** *(4 marks)* **AUC** = area under the ROC curve = the probability that the classifier
ranks a random **positive** above a random **negative** (a threshold-independent measure of
ranking quality). **Perfect classifier AUC = 1.0**; **random classifier AUC = 0.5** (the
diagonal). A good classifier's curve **bows toward the top-left corner** (high TPR at low FPR);
the further from the diagonal, the better.

**(c)** *(4 marks)* With rare positives there are **many easy negatives** (large TN), which
keeps FPR low even when there are many false positives — so the ROC/AUC can look **deceptively
good**. The **PR curve** uses precision = TP/(TP+FP) and recall = TP/(TP+FN), **neither of which
involves TN**, so it directly exposes how many of the positive predictions are actually wrong —
more informative when the positive class is rare or false positives are costly.

---

## Question 4.

**(a)** *(2 marks)* "free offer" → Money 0, Free 1, Win 0, Offer 1, Meeting 0, Report 0
→ **q = [0, 1, 0, 1, 0, 0]**.

**(b)** *(5 marks)* Squared Euclidean distances (then √):
- to #1 [2,1,1,0,0,0]: (0−2)²+(1−1)²+(0−1)²+(1−0)²+0+0 = 4+0+1+1 = **6** → √6 ≈ **2.449**
- to #2 [1,1,0,1,0,0]: (0−1)²+0+0+0+0+0 = **1** → √1 = **1.000**
- to #3 [0,0,0,0,1,1]: 0+1+0+1+1+1 = **4** → √4 = **2.000**
- to #4 [0,0,0,1,1,1]: 0+1+0+0+1+1 = **3** → √3 ≈ **1.732**

**(c)** *(3 marks)*
- **k = 1:** nearest is **#2** (distance 1.000) → predict **True (spam)**.
- **k = 3:** three nearest are #2 = 1.000 (True), #4 = 1.732 (False), #3 = 2.000 (False).
  Votes = 1 True, 2 False → predict **False (not spam)**.
*(Note the prediction flips from spam at k=1 to not-spam at k=3.)*

**(d)** *(2 marks)* Feature scaling matters because (any two): k-NN uses **distances**, so a
feature with a **large numeric range dominates** the distance and swamps small-range features;
without scaling, distances (and therefore neighbours) reflect units rather than relevance.
The bias/variance hyperparameter is **k** (the number of neighbours): small k = low bias/high
variance (overfit), large k = high bias/low variance (underfit).

---

## Question 5.

**(a)** *(5 marks)*
- **Multiclass:** exactly **one** label per instance, chosen from **> 2 mutually exclusive**
  classes. *Example:* handwritten-digit recognition — features = pixel intensities; response ∈
  {0,…,9}.
- **Multilabel:** **multiple** (independent) binary labels per instance simultaneously.
  *Example:* tagging which people appear in a photo — features = image pixels; response =
  {Alice?, Bob?, Charlie?} (a set of binary tags).
*(2 marks distinction, 1½ per correctly described example.)*

**(b)** *(4 marks)*
- **OvR:** train **N** binary classifiers (one class vs all the rest); predict the class whose
  classifier gives the highest score.
- **OvO:** train one classifier per **pair** of classes → **N·(N−1)/2** classifiers; predict
  the class winning the most duels.
- OvO is preferred for **SVMs** because SVM training **scales poorly with dataset size**
  (≈ O(m²–m³)); OvO trains each classifier on only the **two relevant classes' data** (small
  subsets), which is faster than a few classifiers on the full large set.

**(c)** *(3 marks)*
```python
y_scores = forest.predict_proba(X)[:, 1]   # probability of the positive class
```
`predict_proba` returns estimated **class probabilities**; column 1 is the estimated
probability of the **positive class**, used as the ranking score for `precision_recall_curve`.
