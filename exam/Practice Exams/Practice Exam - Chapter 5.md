# CITS5508 Practice Exam — Chapter 5

**Support Vector Machines**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 5 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** Describe the **training procedure** of a linear SVM classifier. Your answer should
explain: how the **margin** is defined, what the **support vectors** are, which training
instances are relevant for defining the **decision boundary**, and what quantity is used as
the **decision rule** for prediction. *(6 marks)*

**(b)** Explain what it means to **maximise the margin**, and why this is expressed as
**minimising ½‖w‖²** rather than maximising the margin directly. *(3 marks)*

**(c)** Why is it important to **scale the input features** before training an SVM? *(3 marks)*

---

## Question 2. (12 marks)

**(a)** Explain the difference between **hard margin** and **soft margin** classification, and
give **two** problems with hard margin classification. *(5 marks)*

**(b)** Describe the effect of the regularisation hyperparameter **C** on a linear SVM. If your
SVM is **overfitting**, should you increase or decrease C? *(4 marks)*

**(c)** Can a `LinearSVC` output a **probability** for its predictions? What about a confidence
**score**? Explain. *(3 marks)*

---

## Question 3. (12 marks)

**(a)** Explain what the **kernel trick** is and why it is useful. *(4 marks)*

**(b)** State the formula (or describe) the **Gaussian RBF kernel**, and explain the effect of
increasing the hyperparameter **γ (gamma)** on the decision boundary. Is γ behaving like a
regularisation parameter — if the model underfits, should you increase or decrease γ?
*(4 marks)*

**(c)** According to **Mercer's theorem**, why can you use a kernel even without knowing the
feature-space transformation φ? What is special about the φ corresponding to the RBF kernel?
*(4 marks)*

---

## Question 4. (12 marks)

**(a)** Compare the scikit-learn classes **`LinearSVC`**, **`SVC`**, and **`SGDClassifier`**
for SVM classification, in terms of (i) computational/time complexity, (ii) support for the
kernel trick, and (iii) suitability for very large or out-of-core datasets. *(6 marks)*

**(b)** You have a **linearly separable** dataset with 2 million instances and want a linear
SVM. Which class would you choose and why? *(3 marks)*

**(c)** You have a small (~1,000 instance) dataset that is clearly **not** linearly separable.
Which class and kernel would you try, and why? *(3 marks)*

---

## Question 5. (12 marks)

**(a)** Explain how SVMs are adapted for **regression** (SVM regression). What does the
hyperparameter **ε** control, and what does it mean for the model to be **ε-insensitive**?
*(5 marks)*

**(b)** Write a short scikit-learn snippet that builds a pipeline scaling the features and then
training an **RBF-kernel SVM classifier** with `C=10` and `gamma=0.5`. *(4 marks)*

**(c)** The SVM optimisation problem can be solved in its **primal** or **dual** form. State
one reason you might solve the **dual** problem, and explain its connection to the kernel
trick. *(3 marks)*
