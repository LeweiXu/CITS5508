# CITS5508 Practice Exam — Chapter 4

**Training Models**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 4 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** A linear regression model can be trained using the **Normal Equation** or using
**gradient descent**. State **two** situations in which you would prefer gradient descent over
the Normal Equation, and explain why. *(4 marks)*

**(b)** Describe the roles of the three gradient descent variants — **batch**, **stochastic**,
and **mini-batch** GD — and state one advantage of stochastic GD over batch GD. *(4 marks)*

**(c)** Explain the effect of the **learning rate** η on gradient descent if it is set (i) too
small and (ii) too large. Why is feature scaling important before running gradient descent?
*(4 marks)*

---

## Question 2. (12 marks)

**(a)** Explain what a **learning curve** plots. Sketch or describe the shape of the training
and validation curves for (i) a model that is **underfitting** and (ii) a model that is
**overfitting**. *(6 marks)*

**(b)** The generalisation error can be decomposed into **bias**, **variance**, and
**irreducible error**. Define each, and state which one is reduced by increasing model
complexity. *(4 marks)*

**(c)** You train a ridge regression model and observe that the **training error and
validation error are almost equal and both fairly high**. Is the model suffering from high
bias or high variance? Should you **increase** or **decrease** the regularisation
hyperparameter α? Justify. *(2 marks)*

---

## Question 3. (12 marks)

**(a)** Write down (in words or formula) how the cost function of **ridge regression** differs
from that of **lasso regression**. *(3 marks)*

**(b)** Lasso tends to produce a **sparse model** while ridge does not. Explain *why* lasso
drives some weights exactly to zero whereas ridge only shrinks them. *(4 marks)*

**(c)** For each situation, recommend ridge, lasso, elastic net, or plain linear regression,
and justify: *(3 marks)*
  1. You want at least a little regularisation and have no reason to suspect useless features.
  2. You suspect only a few of many features are useful.
  3. You have more features than instances and several features are strongly correlated.

**(d)** Explain **early stopping** as a regularisation technique for iterative learners.
*(2 marks)*

---

## Question 4. (12 marks)

A logistic regression model estimates the probability that a student **passes** a unit:

p̂ = σ(θ₀ + θ₁·x₁ + θ₂·x₂),  where σ(t) = 1 / (1 + e⁻ᵗ),

with x₁ = hours studied, x₂ = prior GPA, and fitted coefficients θ₀ = −4, θ₁ = 0.05, θ₂ = 0.8.

**(a)** A student has a GPA of x₂ = 3.0. How many **hours** must they study to have a **50%**
chance of passing? Show your working. *(4 marks)*

**(b)** For a student with GPA 3.0 who studies **40 hours**, compute the estimated probability
of passing. *(3 marks)*

**(c)** Explain why setting the estimated probability to 50% corresponds to setting the
**logit** (the linear part) to zero, and state what kind of **decision boundary** logistic
regression produces. *(3 marks)*

**(d)** You must classify photos along two independent axes: indoor/outdoor **and**
day/night. Should you use one **softmax** regression classifier or two **logistic** regression
classifiers? Justify. *(2 marks)*

---

## Question 5. (12 marks)

A dataset has been split and scaled as follows:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**(a)** Implement a **ridge regression** model with scikit-learn. **Grid-search** the
regularisation strength α over **five** values using **k-fold cross-validation with k = 3**,
refit the best model on the whole training set, **print the best α**, and compute the **mean
squared error on the test set**. *(6 marks)*

**(b)** In scikit-learn's `LogisticRegression`, the regularisation strength is controlled by
`C`, not `alpha`. State the relationship between `C` and regularisation strength. *(2 marks)*

**(c)** Write down **all** the polynomial features (up to and including degree 3) that
`PolynomialFeatures(degree=3)` would generate for two input features x₁ and x₂ (excluding the
bias term). *(4 marks)*
