# CITS5508 Practice Exam — Chapter 8

**Dimensionality Reduction**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 8 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** State **two** main motivations for reducing a dataset's dimensionality and **two** main
drawbacks. *(4 marks)*

**(b)** Explain the **curse of dimensionality**. Why does high-dimensional data tend to be
**sparse**, and why does this increase the risk of **overfitting**? *(4 marks)*

**(c)** Why is "just collect more data to fill the space" usually not a practical solution to
the curse of dimensionality? *(2 marks)*

**(d)** Once a dataset's dimensionality has been reduced (e.g. by PCA), can the operation be
reversed exactly? Explain. *(2 marks)*

---

## Question 2. (12 marks)

**(a)** Explain the difference between the two main approaches to dimensionality reduction:
**projection** and **manifold learning**. *(4 marks)*

**(b)** Briefly explain what a **manifold** and **manifold learning** mean, and state the
**manifold assumption (hypothesis)**. *(4 marks)*

**(c)** Using the Swiss roll as an example, explain why simple projection (e.g. dropping a
dimension) can fail, and what manifold learning does instead. *(4 marks)*

---

## Question 3. (12 marks)

**(a)** Explain how **PCA** chooses the axis (1st principal component) onto which it projects.
Give the **two** equivalent justifications for this choice. *(4 marks)*

**(b)** What is the **explained variance ratio**, and how would you use it to choose the number
of dimensions to keep? *(4 marks)*

**(c)** Write a scikit-learn snippet that applies PCA to keep **95%** of the variance, and a
second line that prints how many components were actually retained. *(4 marks)*

---

## Question 4. (12 marks)

**(a)** A 1,000-dimensional dataset is reduced with PCA, keeping an explained-variance ratio of
95%. How many dimensions will the resulting dataset have? Explain your reasoning. *(3 marks)*

**(b)** Define **reconstruction error** in the context of PCA, and explain how it can be used
for **anomaly detection**. *(4 marks)*

**(c)** State when you would use each of: **regular PCA**, **incremental PCA**, **randomized
PCA**, and **random projection**. *(5 marks)*

---

## Question 5. (12 marks)

**(a)** Explain how **random projection** reduces dimensionality, and why it can preserve
distances reasonably well (refer to the Johnson–Lindenstrauss lemma). Does the required output
dimensionality depend on the original number of features n? *(5 marks)*

**(b)** **LLE** is a manifold-learning technique. Outline its two main steps at a high level.
*(4 marks)*

**(c)** Name one dimensionality-reduction technique especially suited to **visualising
clusters** in 2-D, and one **supervised** technique that projects onto the most
class-discriminative axes. *(3 marks)*
