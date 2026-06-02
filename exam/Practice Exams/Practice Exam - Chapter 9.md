# CITS5508 Practice Exam — Chapter 9

**Unsupervised Learning Techniques**

- This practice paper has **5 questions** for **60 marks**.
- Allow approximately **two minutes per mark** (≈ 2 hours total).
- Only UWA-approved calculators permitted.
- Answers are in the separate file *Practice Exam - Chapter 9 (Answer Key).md*.

---

## Question 1. (12 marks)

**(a)** Define **clustering**, and explain how the meaning of an instance's "label" differs
between clustering and classification. *(3 marks)*

**(b)** List **four** distinct applications of clustering. *(4 marks)*

**(c)** Describe the **k-means algorithm** as a loop of two repeating steps. Why is it
guaranteed to converge, and why might it converge to a **suboptimal** solution? *(5 marks)*

---

## Question 2. (12 marks)

**(a)** Explain why **inertia** is **not** a good metric for choosing the number of clusters k.
*(3 marks)*

**(b)** Describe the **elbow method** and the **silhouette score** as two techniques for
choosing k. *(4 marks)*

**(c)** An instance has a mean intra-cluster distance **a = 0.5** and a mean nearest-cluster
distance **b = 2.0**. Compute its **silhouette coefficient**, and interpret what the value
tells you. State the range of possible values. *(3 marks)*

**(d)** State **two** situations in which k-means performs poorly. *(2 marks)*

---

## Question 3. (12 marks)

**(a)** Describe how **DBSCAN** defines clusters. In your answer define **ε-neighbourhood**,
**core instance**, and how anomalies are identified. *(5 marks)*

**(b)** State **two** advantages of DBSCAN over k-means, and **one** limitation. *(4 marks)*

**(c)** `DBSCAN` has no `predict()` method. Describe how you could classify a new instance into
one of the clusters DBSCAN found. *(3 marks)*

---

## Question 4. (12 marks)

**(a)** Describe what a **Gaussian Mixture Model (GMM)** assumes about how the data was
generated. *(3 marks)*

**(b)** GMMs are trained with the **Expectation–Maximization (EM)** algorithm. Describe the two
steps, and explain how EM differs from k-means (hint: hard vs soft assignment, what extra
properties of clusters EM learns). *(5 marks)*

**(c)** Inertia and the silhouette score are not reliable for choosing the number of clusters
in a GMM. Name **two** criteria you would use instead, and name a model variant that selects
the number of clusters automatically. *(4 marks)*

---

## Question 5. (12 marks)

**(a)** Explain the difference between **anomaly detection** and **novelty detection**,
focusing on what each assumes about the **training set**. *(4 marks)*

**(b)** Explain how a trained **GMM can be used for anomaly detection**, including how you would
choose the density threshold and how it relates to the precision/recall trade-off. *(4 marks)*

**(c)** Name **two** dedicated algorithms (other than GMM) for anomaly/novelty detection, and
in one sentence each describe how they identify anomalies. *(4 marks)*
