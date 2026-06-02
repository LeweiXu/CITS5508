# CITS5508 Practice Exam — Chapters 1–2

**The ML Landscape & End-to-End ML Projects**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapters 1-2 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** Tom Mitchell defined machine learning in terms of a task **T**, a performance
measure **P**, and experience **E**. For a system that learns to recommend films to a
streaming-service user, state what T, E and P could each be. *(3 marks)*

**(b)** Explain the difference between **supervised** and **unsupervised** learning. Give one
real-world application of each, and in each case describe the **features** and (if any) the
**response variable**. *(4 marks)*

**(c)** For each of the following, state whether it is best framed as supervised,
unsupervised, semi-supervised, self-supervised, or reinforcement learning, and justify in one
sentence: *(3 marks)*
  1. Training a robot to walk over unknown terrain by trial and reward.
  2. Grouping a website's visitors into segments with no predefined categories.
  3. Pre-training an image model by masking patches and predicting the missing pixels.

**(d)** What is the difference between **instance-based** and **model-based** learning? Name
one algorithm of each type. *(2 marks)*

---

## Question 2. (12 marks)

**(a)** Define a model **parameter** and a model **hyperparameter**, and clearly state what
distinguishes them. *(3 marks)*

**(b)** Explain what **overfitting** and **underfitting** mean. For each, give **two** possible
remedies. *(5 marks)*

**(c)** Explain what **regularisation** is and how it relates to a model's **degrees of
freedom**. Why does setting a regularisation hyperparameter to a very large value risk
underfitting? *(4 marks)*

---

## Question 3. (12 marks)

**(a)** You are choosing between three candidate models. Explain why you should **not** select
the best one by repeatedly evaluating all three on the **test set**, and describe how a
**validation set** (or cross-validation) avoids this problem. *(4 marks)*

**(b)** Describe **k-fold cross-validation**, and state one advantage and one disadvantage
compared with a single holdout validation set. *(4 marks)*

**(c)** A team trains a flower-classifier on millions of images scraped from the web, but the
app will run on photos taken with phone cameras. After training, validation accuracy is poor.
Explain how a **train-dev set** would let them determine whether the problem is **overfitting**
or **data mismatch**. *(4 marks)*

---

## Question 4. (12 marks)

**(a)** A regression model produced the following predictions on five test instances:

| Instance | Actual yᵢ | Predicted ŷᵢ |
|---|---|---|
| 1 | 10 | 12 |
| 2 | 20 | 18 |
| 3 | 30 | 35 |
| 4 | 40 | 39 |
| 5 | 50 | 38 |

Compute the **MAE** and the **RMSE** of these predictions. Show your working. *(5 marks)*

**(b)** Which of the two metrics is **more sensitive to outliers**, and why? With which norm
(ℓ₁ or ℓ₂) is each associated? *(3 marks)*

**(c)** Explain **data-snooping bias** and the practice that prevents it. *(2 marks)*

**(d)** Explain what **stratified sampling** is and why it can give a better train/test split
than purely random sampling. *(2 marks)*

---

## Question 5. (12 marks)

**(a)** The following code prepares a dataset:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Explain why `fit_transform` is called on the training set but only `transform` on the test
set. What problem would arise if you called `fit_transform` on the test set as well?
*(4 marks)*

**(b)** A dataset has a numeric feature with some missing values and a nominal categorical
feature. Describe an appropriate preprocessing step for **each**, and explain why an
`OrdinalEncoder` would be a poor choice for the categorical feature. *(4 marks)*

**(c)** Write a short scikit-learn snippet that builds a `Pipeline` (or `make_pipeline`) which
median-imputes missing values, standardises the features, and finishes with a
`LinearRegression` estimator. Briefly state one benefit of bundling these steps into a single
pipeline. *(4 marks)*
