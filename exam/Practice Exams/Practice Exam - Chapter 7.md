# CITS5508 Practice Exam — Chapter 7

**Ensemble Learning and Random Forests**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 7 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** Explain, using the idea of the **law of large numbers** (or the "wisdom of the crowd"),
why an ensemble of many **weak learners** can form a **strong learner**. *(4 marks)*

**(b)** This benefit relies on a critical assumption about the individual predictors. State it,
and explain one practical way to make an ensemble's predictors satisfy it better. *(4 marks)*

**(c)** Explain the difference between a **hard voting** and a **soft voting** classifier, and
state which tends to perform better and why. *(4 marks)*

---

## Question 2. (12 marks)

**(a)** Explain the difference between **bagging** and **pasting**. *(3 marks)*

**(b)** Both methods generally give an ensemble with **similar bias but lower variance** than a
single predictor. Explain why aggregation reduces variance. *(3 marks)*

**(c)** What is **out-of-bag (OOB)** evaluation? Approximately what fraction of the training
set is OOB for a given predictor under bagging, and why is OOB evaluation useful? *(4 marks)*

**(d)** Why do bagging and pasting **scale well** across multiple CPU cores or servers, whereas
boosting does not? *(2 marks)*

---

## Question 3. (12 marks)

**(a)** Describe what a **random forest** is and the **extra source of randomness** it
introduces compared with a plain bagging ensemble of decision trees. *(4 marks)*

**(b)** Describe **extra-trees (extremely randomized trees)**. What additional randomness do
they add, and what are the consequences for **bias/variance** and **training speed**?
*(4 marks)*

**(c)** Explain how a random forest computes **feature importance**, and give one practical use
of it. *(4 marks)*

---

## Question 4. (12 marks)

**(a)** Explain the general principle of **boosting**. How does it differ fundamentally from
bagging? *(3 marks)*

**(b)** Describe how **AdaBoost** trains its sequence of predictors. If an AdaBoost ensemble is
**overfitting**, what could you change? *(4 marks)*

**(c)** Describe how **gradient boosting** works (e.g. GBRT). What does the **`learning_rate`**
hyperparameter control, and what is the relationship between the learning rate and the number
of trees needed (shrinkage)? *(5 marks)*

---

## Question 5. (12 marks)

**(a)** Explain how **stacking** differs from a simple voting classifier, and define the
**blender** (meta-learner). *(4 marks)*

**(b)** When training a stacking ensemble, why are the blender's training features generated
using **cross-validated (out-of-sample) predictions** from the base predictors rather than
their predictions on the data they were trained on? *(4 marks)*

**(c)** Write a short scikit-learn snippet that builds a **soft-voting** classifier combining a
`LogisticRegression`, a `RandomForestClassifier`, and an `SVC`. Note any change needed to the
`SVC` for soft voting to work. *(4 marks)*
