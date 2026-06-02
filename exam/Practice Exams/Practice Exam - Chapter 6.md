# CITS5508 Practice Exam — Chapter 6

**Decision Trees**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 6 (Answer Key).md*.

---

## Question 1. (12 marks)

A decision-tree node applies to **50** training instances with class counts: **0** of class A,
**40** of class B, **10** of class C.

**(a)** Compute the node's **Gini impurity**. Show your working. *(3 marks)*

**(b)** Compute the node's **entropy** (use log₂). Show your working. *(4 marks)*

**(c)** What would the Gini impurity be if all 50 instances belonged to a single class? What
do we call such a node? *(2 marks)*

**(d)** State one practical difference between using **Gini impurity** and **entropy** as the
split criterion, and say which is the scikit-learn default and why. *(3 marks)*

---

## Question 2. (12 marks)

The **CART** algorithm is considering a split of a node containing **100** instances. The
candidate split produces:
- **Left** child: 40 instances — 30 of class A, 10 of class B.
- **Right** child: 60 instances — 10 of class A, 50 of class B.

**(a)** Compute the Gini impurity of the **left** child and of the **right** child.
*(4 marks)*

**(b)** Compute the **CART cost** for this split (the size-weighted average impurity of the
children). *(3 marks)*

**(c)** Explain why CART is described as a **greedy** algorithm, and why finding the optimal
tree is generally intractable. *(3 marks)*

**(d)** State the time complexity of **prediction** and of **training** for a decision tree
(in terms of m instances and n features). *(2 marks)*

---

## Question 3. (12 marks)

**(a)** Decision trees are called **nonparametric** models. Explain what this means and why it
makes them prone to **overfitting**. *(4 marks)*

**(b)** List **four** hyperparameters you can use to **regularise** a `DecisionTreeClassifier`,
and for each state whether you increase or decrease it to reduce overfitting. *(4 marks)*

**(c)** Briefly describe **pruning** as an alternative regularisation approach, including the
role of a statistical test such as the χ² test. *(4 marks)*

---

## Question 4. (12 marks)

**(a)** Decision trees are described as **white-box** models, whereas random forests and neural
networks are **black-box**. Explain the distinction and why it matters. *(4 marks)*

**(b)** Explain how a decision tree **estimates class probabilities** for a new instance. Why
does it give the **same** probability estimate for every instance that falls in the same leaf?
*(4 marks)*

**(c)** Decision trees require **very little data preparation**. Name one preprocessing step
that is **not** required for decision trees but **is** required for SVMs / k-NN, and explain
why. *(4 marks)*

---

## Question 5. (12 marks)

**(a)** Explain the **sensitivity to axis orientation** limitation of decision trees, and
describe one technique that mitigates it. *(4 marks)*

**(b)** Decision trees are said to have **high variance**. Explain what this means and how an
**ensemble** (random forest) addresses it. *(4 marks)*

**(c)** Write a short scikit-learn snippet that trains a `DecisionTreeClassifier` regularised
with `max_depth=3` and `min_samples_leaf=10`, and prints the trained tree's feature
importances. *(4 marks)*
