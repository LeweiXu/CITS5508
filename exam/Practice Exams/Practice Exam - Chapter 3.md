# CITS5508 Practice Exam — Chapter 3

**Classification (+ k-Nearest Neighbours)**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 3 (Answer Key).md*.

---

## Question 1. (12 marks)

A binary classifier was evaluated, producing the following confusion matrix (rows = actual,
columns = predicted):

|                | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 80 | 20 |
| **Actual Positive** | 10 | 40 |

**(a)** Identify the number of true positives, true negatives, false positives, and false
negatives. *(2 marks)*

**(b)** Compute the **accuracy**, **precision**, **recall**, and **F₁ score**. Show your
working. *(6 marks)*

**(c)** A colleague says "the classifier has 80% accuracy, so it is good." The positive class
makes up only a small fraction of a much larger dataset. Explain why accuracy can be misleading
here and which metrics you would report instead. *(4 marks)*

---

## Question 2. (12 marks)

**(a)** Explain the **precision/recall trade-off** in terms of the classifier's **decision
threshold**. What happens to precision and recall as the threshold is raised? *(4 marks)*

**(b)** For each scenario, state whether you would favour **high precision** or **high recall**,
and justify: *(4 marks)*
  1. A classifier that flags videos as "safe for children."
  2. A classifier that screens medical scans for a dangerous tumour.

**(c)** Recall only ever decreases as the threshold is raised, but precision can sometimes
*dip*. Explain why precision is not strictly monotonic in the threshold. *(4 marks)*

---

## Question 3. (12 marks)

**(a)** Describe how the **ROC curve** is constructed: name the quantity on each axis and define
it. *(4 marks)*

**(b)** What does the **AUC** measure? State the AUC of a perfect classifier and of a purely
random one, and explain how to read the curve to judge classifier quality. *(4 marks)*

**(c)** A dataset is highly imbalanced (positives are rare). Explain why the **precision/recall
curve** may be more informative than the ROC curve in this case. *(4 marks)*

---

## Question 4. (12 marks)

Consider the email bag-of-words training set below. The dictionary order is
**[Money, Free, Win, Offer, Meeting, Report]**.

| i | Money | Free | Win | Offer | Meeting | Report | Spam |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 1 | 0 | 0 | 0 | True |
| 2 | 1 | 1 | 0 | 1 | 0 | 0 | True |
| 3 | 0 | 0 | 0 | 0 | 1 | 1 | False |
| 4 | 0 | 0 | 0 | 1 | 1 | 1 | False |

A new query email is **"free offer"**.

**(a)** Write the bag-of-words feature vector for the query. *(2 marks)*

**(b)** Using **Euclidean distance**, compute the distance from the query to each of the four
training emails. Show your working. *(5 marks)*

**(c)** What label does a **k = 1** nearest-neighbour classifier predict? What does a **k = 3**
classifier predict? Show the votes. *(3 marks)*

**(d)** Give **two** reasons why feature scaling matters for k-NN, and name the hyperparameter
that controls the bias/variance trade-off in k-NN. *(2 marks)*

---

## Question 5. (12 marks)

**(a)** Explain the difference between **multiclass** and **multilabel** classification. Give a
real-world example of each, describing the features and the response. *(5 marks)*

**(b)** Explain the **OvR (one-versus-the-rest)** and **OvO (one-versus-one)** strategies. For
N classes, how many binary classifiers does each require, and why is OvO often preferred for
SVMs? *(4 marks)*

**(c)** A `RandomForestClassifier` does not expose a `decision_function()`. Write one line of
code showing how you would obtain per-instance scores suitable for plotting a precision/recall
curve, and explain what those scores represent. *(3 marks)*
