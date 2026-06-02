## **CHAPTER 9 Unsupervised Learning Techniques** 

Although most of the applications of machine learning today are based on supervised learning (and as a result, this is where most of the investments go to), the vast majority of the available data is unlabeled: we have the input features **X** , but we do not have the labels **y** . The computer scientist Yann LeCun famously said that “if intelligence was a cake, unsupervised learning would be the cake, supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake.” In other words, there is a huge potential in unsupervised learning that we have only barely started to sink our teeth into. 

Say you want to create a system that will take a few pictures of each item on a manufacturing production line and detect which items are defective. You can fairly easily create a system that will take pictures automatically, and this might give you thousands of pictures every day. You can then build a reasonably large dataset in just a few weeks. But wait, there are no labels! If you want to train a regular binary classi‐ fier that will predict whether an item is defective or not, you will need to label every single picture as “defective” or “normal”. This will generally require human experts to sit down and manually go through all the pictures. This is a long, costly, and tedious task, so it will usually only be done on a small subset of the available pictures. As a result, the labeled dataset will be quite small, and the classifier’s performance will be disappointing. Moreover, every time the company makes any change to its products, the whole process will need to be started over from scratch. Wouldn’t it be great if the algorithm could just exploit the unlabeled data without needing humans to label every picture? Enter unsupervised learning. 

**259** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

In Chapter 8 we looked at the most common unsupervised learning task: dimension‐ ality reduction. In this chapter we will look at a few more unsupervised tasks: 

## _Clustering_ 

The goal is to group similar instances together into _clusters_ . Clustering is a great tool for data analysis, customer segmentation, recommender systems, search engines, image segmentation, semi-supervised learning, dimensionality reduc‐ tion, and more. 

## _Anomaly detection (also called outlier detection)_ 

The objective is to learn what “normal” data looks like, and then use that to detect abnormal instances. These instances are called _anomalies_ , or _outliers_ , while the normal instances are called _inliers_ . Anomaly detection is useful in a wide variety of applications, such as fraud detection, detecting defective products in manufacturing, identifying new trends in time series, or removing outliers from a dataset before training another model, which can significantly improve the performance of the resulting model. 

## _Density estimation_ 

This is the task of estimating the _probability density function_ (PDF) of the random process that generated the dataset. Density estimation is commonly used for anomaly detection: instances located in very low-density regions are likely to be anomalies. It is also useful for data analysis and visualization. 

Ready for some cake? We will start with two clustering algorithms, _k_ -means and DBSCAN, then we’ll discuss Gaussian mixture models and see how they can be used for density estimation, clustering, and anomaly detection. 

## **Clustering Algorithms: k-means and DBSCAN** 

As you enjoy a hike in the mountains, you stumble upon a plant you have never seen before. You look around and you notice a few more. They are not identical, yet they are sufficiently similar for you to know that they most likely belong to the same species (or at least the same genus). You may need a botanist to tell you what species that is, but you certainly don’t need an expert to identify groups of similar-looking objects. This is called _clustering_ : it is the task of identifying similar instances and assigning them to _clusters_ , or groups of similar instances. 

Just like in classification, each instance gets assigned to a group. However, unlike classification, clustering is an unsupervised task. Consider Figure 9-1: on the left is the iris dataset (introduced in Chapter 4), where each instance’s species (i.e., its class) is represented with a different marker. It is a labeled dataset, for which classification algorithms such as logistic regression, SVMs, or random forest classifiers are well suited. On the right is the same dataset, but without the labels, so you cannot use a classification algorithm anymore. This is where clustering algorithms step in: many of 

**260 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

them can easily detect the lower-left cluster. It is also quite easy to see with our own eyes, but it is not so obvious that the upper-right cluster is composed of two distinct subclusters. That said, the dataset has two additional features (sepal length and width) that are not represented here, and clustering algorithms can make good use of all features, so in fact they identify the three clusters fairly well (e.g., using a Gaussian mixture model, only 5 instances out of 150 are assigned to the wrong cluster). 

_Figure 9-1. Classification (left) versus clustering (right)_ 

Clustering is used in a wide variety of applications, including: 

## _Customer segmentation_ 

You can cluster your customers based on their purchases and their activity on your website. This is useful to understand who your customers are and what they need, so you can adapt your products and marketing campaigns to each segment. For example, customer segmentation can be useful in _recommender systems_ to suggest content that other users in the same cluster enjoyed. 

## _Data analysis_ 

When you analyze a new dataset, it can be helpful to run a clustering algorithm, and then analyze each cluster separately. 

## _Dimensionality reduction_ 

Once a dataset has been clustered, it is usually possible to measure each instance’s _affinity_ with each cluster; affinity is any measure of how well an instance fits into a cluster. Each instance’s feature vector **x** can then be replaced with the vector of its cluster affinities. If there are _k_ clusters, then this vector is _k_ -dimensional. The new vector is typically much lower-dimensional than the original feature vector, but it can preserve enough information for further processing. 

**Clustering Algorithms: k-means and DBSCAN | 261** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## _Feature engineering_ 

The cluster affinities can often be useful as extra features. For example, we used _k_ -means in Chapter 2 to add geographic cluster affinity features to the California housing dataset, and they helped us get better performance. 

## _Anomaly detection (also called outlier detection)_ 

Any instance that has a low affinity to all the clusters is likely to be an anomaly. For example, if you have clustered the users of your website based on their behavior, you can detect users with unusual behavior, such as an unusual number of requests per second. 

## _Semi-supervised learning_ 

If you only have a few labels, you could perform clustering and propagate the labels to all the instances in the same cluster. This technique can greatly increase the number of labels available for a subsequent supervised learning algorithm, and thus improve its performance. 

## _Search engines_ 

Some search engines let you search for images that are similar to a reference image. To build such a system, you would first apply a clustering algorithm to all the images in your database; similar images would end up in the same cluster. Then when a user provides a reference image, all you’d need to do is use the trained clustering model to find this image’s cluster, and you could then simply return all the images from this cluster. 

## _Image segmentation_ 

By clustering pixels according to their color, then replacing each pixel’s color with the mean color of its cluster, it is possible to considerably reduce the number of different colors in an image. Image segmentation is used in many object detection and tracking systems, as it makes it easier to detect the contour of each object. 

There is no universal definition of what a cluster is: it really depends on the context, and different algorithms will capture different kinds of clusters. Some algorithms look for instances centered around a particular point, called a _centroid_ . Others look for continuous regions of densely packed instances: these clusters can take on any shape. Some algorithms are hierarchical, looking for clusters of clusters. And the list goes on. 

In this section, we will look at two popular clustering algorithms, _k_ -means and DBSCAN, and explore some of their applications, such as nonlinear dimensionality reduction, semi-supervised learning, and anomaly detection. 

## **262 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## **k-means** 

Consider the unlabeled dataset represented in Figure 9-2: you can clearly see five blobs of instances. The _k_ -means algorithm is a simple algorithm capable of clustering this kind of dataset very quickly and efficiently, often in just a few iterations. It was proposed by Stuart Lloyd at Bell Labs in 1957 as a technique for pulse-code modulation, but it was only published outside of the company in 1982.[1] In 1965, Edward W. Forgy had published virtually the same algorithm, so _k_ -means is some‐ times referred to as the Lloyd–Forgy algorithm. 

_Figure 9-2. An unlabeled dataset composed of five blobs of instances_ 

Let’s train a _k_ -means clusterer on this dataset. It will try to find each blob’s center and assign each instance to the closest blob: 

**`from sklearn.cluster import`** `KMeans` **`from sklearn.datasets import`** `make_blobs` 

`X, y = make_blobs([...])` _`# make the blobs: y contains the cluster IDs, but we # will not use them; that's what we want to predict`_ `k = 5 kmeans = KMeans(n_clusters=k, random_state=42) y_pred = kmeans.fit_predict(X)` 

Note that you have to specify the number of clusters _k_ that the algorithm must find. In this example, it is pretty obvious from looking at the data that _k_ should be set to 5, but in general it is not that easy. We will discuss this shortly. 

Each instance will be assigned to one of the five clusters. In the context of clustering, an instance’s _label_ is the index of the cluster to which the algorithm assigns this instance; this is not to be confused with the class labels in classification, which are used as targets (remember that clustering is an unsupervised learning task). The 

> 1 Stuart P. Lloyd, “Least Squares Quantization in PCM”, _IEEE Transactions on Information Theory_ 28, no. 2 (1982): 129–137. 

**Clustering Algorithms: k-means and DBSCAN | 263** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

`KMeans` instance preserves the predicted labels of the instances it was trained on, available via the `labels_` instance variable: 

**`>>>`** `y_pred array([4, 0, 1, ..., 2, 1, 0], dtype=int32)` **`>>>`** `y_pred` **`is`** `kmeans.labels_ True` 

We can also take a look at the five centroids that the algorithm found: 

**`>>>`** `kmeans.cluster_centers_ array([[-2.80389616,  1.80117999], [ 0.20876306,  2.25551336], [-2.79290307,  2.79641063], [-1.46679593,  2.28585348], [-2.80037642,  1.30082566]])` 

You can easily assign new instances to the cluster whose centroid is closest: 

**`>>> import numpy as np >>>`** `X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])` **`>>>`** `kmeans.predict(X_new) array([1, 1, 2, 2], dtype=int32)` 

If you plot the cluster’s decision boundaries, you get a Voronoi tessellation: see Figure 9-3, where each centroid is represented with an X. 

_Figure 9-3. k-means decision boundaries (Voronoi tessellation)_ 

The vast majority of the instances were clearly assigned to the appropriate cluster, but a few instances were probably mislabeled, especially near the boundary between the top-left cluster and the central cluster. Indeed, the _k_ -means algorithm does not behave very well when the blobs have very different diameters because all it cares about when assigning an instance to a cluster is the distance to the centroid. 

Instead of assigning each instance to a single cluster, which is called _hard clustering_ , it can be useful to give each instance a score per cluster, which is called _soft clustering_ . The score can be the distance between the instance and the centroid or a similarity 

**264 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

score (or affinity), such as the Gaussian radial basis function we used in Chapter 2. In the `KMeans` class, the `transform()` method measures the distance from each instance to every centroid: 

**`>>>`** `kmeans.transform(X_new).round(2) array([[2.81, 0.33, 2.9 , 1.49, 2.89], [5.81, 2.8 , 5.85, 4.48, 5.84], [1.21, 3.29, 0.29, 1.69, 1.71], [0.73, 3.22, 0.36, 1.55, 1.22]])` 

In this example, the first instance in `X_new` is located at a distance of about 2.81 from the first centroid, 0.33 from the second centroid, 2.90 from the third centroid, 1.49 from the fourth centroid, and 2.89 from the fifth centroid. If you have a highdimensional dataset and you transform it this way, you end up with a _k_ -dimensional dataset: this transformation can be a very efficient nonlinear dimensionality reduc‐ tion technique. Alternatively, you can use these distances as extra features to train another model, as in Chapter 2. 

## **The k-means algorithm** 

So, how does the algorithm work? Well, suppose you were given the centroids. You could easily label all the instances in the dataset by assigning each of them to the cluster whose centroid is closest. Conversely, if you were given all the instance labels, you could easily locate each cluster’s centroid by computing the mean of the instances in that cluster. But you are given neither the labels nor the centroids, so how can you proceed? Start by placing the centroids randomly (e.g., by picking _k_ instances at random from the dataset and using their locations as centroids). Then label the instances, update the centroids, label the instances, update the centroids, and so on until the centroids stop moving. The algorithm is guaranteed to converge in a finite number of steps (usually quite small). That’s because the mean square distance between the instances and their closest centroids can only go down at each step, and since it cannot be negative, it’s guaranteed to converge. 

You can see the algorithm in action in Figure 9-4: the centroids are initialized randomly (top left), then the instances are labeled (top right), then the centroids are updated (center left), the instances are relabeled (center right), and so on. As you can see, in just three iterations the algorithm has reached a clustering that seems close to optimal. 

The computational complexity of the algorithm is generally linear with regard to the number of instances _m_ , the number of clusters _k_ , and the number of dimensions _n_ . However, this is only true when the data has a clustering structure. If it does not, then in the worst-case scenario the complexity can increase exponentially with the number of instances. In practice, this rarely happens, and _k_ -means is generally one of the fastest clustering algorithms. 

**Clustering Algorithms: k-means and DBSCAN | 265** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-4. The k-means algorithm_ 

Although the algorithm is guaranteed to converge, it may not converge to the right solution (i.e., it may converge to a local optimum): whether it does or not depends on the centroid initialization. Figure 9-5 shows two suboptimal solutions that the algorithm can converge to if you are not lucky with the random initialization step. 

_Figure 9-5. Suboptimal solutions due to unlucky centroid initializations_ 

Let’s take a look at a few ways you can mitigate this risk by improving the centroid initialization. 

## **266 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## **Centroid initialization methods** 

If you happen to know approximately where the centroids should be (e.g., if you ran another clustering algorithm earlier), then you can set the `init` hyperparameter to a NumPy array containing the list of centroids, and set `n_init` to `1` : 

```
good_init=np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans=KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
```

Another solution is to run the algorithm multiple times with different random initializations and keep the best solution. The number of random initializations is controlled by the `n_init` hyperparameter: by default it is equal to `10` , which means that the whole algorithm described earlier runs 10 times when you call `fit()` , and Scikit-Learn keeps the best solution. But how exactly does it know which solution is the best? It uses a performance metric! That metric is called the model’s _inertia_ , which is the sum of the squared distances between the instances and their closest centroids. It is roughly equal to 219.4 for the model on the left in Figure 9-5, 258.6 for the model on the right in Figure 9-5, and only 211.6 for the model in Figure 9-3. The `KMeans` class runs the algorithm `n_init` times and keeps the model with the lowest inertia. In this example, the model in Figure 9-3 will be selected (unless we are very unlucky with `n_init` consecutive random initializations). If you are curious, a model’s inertia is accessible via the `inertia_` instance variable: 

```
>>> kmeans.inertia_
211.59853725816836
```

The `score()` method returns the negative inertia (it’s negative because a predictor’s `score()` method must always respect Scikit-Learn’s “greater is better” rule: if a predic‐ tor is better than another, its `score()` method should return a greater score): 

```
>>> kmeans.score(X)
-211.5985372581684
```

An important improvement to the _k_ -means algorithm, _k-means++_ , was proposed in a 2006 paper by David Arthur and Sergei Vassilvitskii.[2] They introduced a smarter initialization step that tends to select centroids that are distant from one another, and this improvement makes the _k_ -means algorithm much less likely to converge to a suboptimal solution. The paper showed that the additional computation required for the smarter initialization step is well worth it because it makes it possible to drastically reduce the number of times the algorithm needs to be run to find the optimal solution. The _k_ -means++ initialization algorithm works like this: 

**1.** Take one centroid **c**[(1)] , chosen uniformly at random from the dataset. 

> 2 David Arthur and Sergei Vassilvitskii, “k-Means++: The Advantages of Careful Seeding”, _Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms_ (2007): 1027–1035. 

**Clustering Algorithms: k-means and DBSCAN | 267** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

- i 2 

- **2.** Take a new centroid **c**[(] _[i]_[)] , choosing an instance **x**[(] _[i]_[)] with probability D x / ∑ mj = 1 D x j 2, where D( **x** ( _i_ )) is the distance between the instance **x** ( _i_ ) and the closest centroid that was already chosen. This probability distribution ensures that instances farther away from already chosen centroids are much more likely to be selected as centroids. 

**3.** Repeat the previous step until all _k_ centroids have been chosen. 

The `KMeans` class uses this initialization method by default. 

## **Accelerated k-means and mini-batch k-means** 

Another improvement to the _k_ -means algorithm was proposed in a 2003 paper by Charles Elkan.[3] On some large datasets with many clusters, the algorithm can be accelerated by avoiding many unnecessary distance calculations. Elkan achieved this by exploiting the triangle inequality (i.e., that a straight line is always the shortest dis‐ tance between two points[4] ) and by keeping track of lower and upper bounds for dis‐ tances between instances and centroids. However, Elkan’s algorithm does not always accelerate training, and sometimes it can even slow down training significantly; it depends on the dataset. Still, if you want to give it a try, set `algorithm="elkan"` . 

Yet another important variant of the _k_ -means algorithm was proposed in a 2010 paper by David Sculley.[5] Instead of using the full dataset at each iteration, the algo‐ rithm is capable of using mini-batches, moving the centroids just slightly at each iteration. This speeds up the algorithm (typically by a factor of three to four) and makes it possible to cluster huge datasets that do not fit in memory. Scikit-Learn implements this algorithm in the `MiniBatchKMeans` class, which you can use just like the `KMeans` class: 

```
fromsklearn.clusterimportMiniBatchKMeans
```

```
minibatch_kmeans=MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
```

If the dataset does not fit in memory, the simplest option is to use the `memmap` class, as we did for incremental PCA in Chapter 8. Alternatively, you can pass one mini-batch at a time to the `partial_fit()` method, but this will require much more work, since you will need to perform multiple initializations and select the best one yourself. 

> 3 Charles Elkan, “Using the Triangle Inequality to Accelerate k-Means”, _Proceedings of the 20th International Conference on Machine Learning_ (2003): 147–153. 

> 4 The triangle inequality is AC ≤ AB + BC, where A, B and C are three points and AB, AC, and BC are the distances between these points. 

- 5 David Sculley, “Web-Scale K-Means Clustering”, _Proceedings of the 19th International Conference on World Wide Web_ (2010): 1177–1178. 

## **268 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Although the mini-batch _k_ -means algorithm is much faster than the regular _k_ -means algorithm, its inertia is generally slightly worse. You can see this in Figure 9-6: the plot on the left compares the inertias of mini-batch _k_ -means and regular _k_ -means models trained on the previous five-blobs dataset using various numbers of clusters _k_ . The difference between the two curves is small, but visible. In the plot on the right, you can see that mini-batch _k_ -means is roughly 3.5 times faster than regular _k_ -means on this dataset. 

_Figure 9-6. Mini-batch k-means has a higher inertia than k-means (left) but it is much faster (right), especially as k increases_ 

## **Finding the optimal number of clusters** 

So far, we’ve set the number of clusters _k_ to 5 because it was obvious by looking at the data that this was the correct number of clusters. But in general, it won’t be so easy to know how to set _k_ , and the result might be quite bad if you set it to the wrong value. As you can see in Figure 9-7, for this dataset setting _k_ to 3 or 8 results in fairly bad models. 

You might be thinking that you could just pick the model with the lowest inertia. Unfortunately, it is not that simple. The inertia for _k_ =3 is about 653.2, which is much higher than for _k_ =5 (211.6). But with _k_ =8, the inertia is just 119.1. The inertia is not a good performance metric when trying to choose _k_ because it keeps getting lower as we increase _k_ . Indeed, the more clusters there are, the closer each instance will be to its closest centroid, and therefore the lower the inertia will be. Let’s plot the inertia as a function of _k_ . When we do this, the curve often contains an inflexion point called the _elbow_ (see Figure 9-8). 

**Clustering Algorithms: k-means and DBSCAN | 269** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-7. Bad choices for the number of clusters: when k is too small, separate clusters get merged (left), and when k is too large, some clusters get chopped into multiple pieces (right)_ 

_Figure 9-8. Plotting the inertia as a function of the number of clusters k_ 

As you can see, the inertia drops very quickly as we increase _k_ up to 4, but then it decreases much more slowly as we keep increasing _k_ . This curve has roughly the shape of an arm, and there is an elbow at _k_ = 4. So, if we did not know better, we might think 4 was a good choice: any lower value would be dramatic, while any higher value would not help much, and we might just be splitting perfectly good clusters in half for no good reason. 

This technique for choosing the best value for the number of clusters is rather coarse. A more precise (but also more computationally expensive) approach is to use the _silhouette score_ , which is the mean _silhouette coefficient_ over all the instances. An instance’s silhouette coefficient is equal to ( _b_ – _a_ ) / max( _a_ , _b_ ), where _a_ is the mean distance to the other instances in the same cluster (i.e., the mean intra-cluster distance) and _b_ is the mean nearest-cluster distance (i.e., the mean distance to the instances of the next closest cluster, defined as the one that minimizes _b_ , excluding the instance’s own cluster). The silhouette coefficient can vary between –1 and +1. A coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters, while a coefficient close to 0 means that it is close to a cluster 

## **270 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

boundary; finally, a coefficient close to –1 means that the instance may have been assigned to the wrong cluster. 

To compute the silhouette score, you can use Scikit-Learn’s `silhouette_score()` function, giving it all the instances in the dataset and the labels they were assigned: 

**`>>> from sklearn.metrics import`** `silhouette_score` **`>>>`** `silhouette_score(X, kmeans.labels_) 0.655517642572828` 

Let’s compare the silhouette scores for different numbers of clusters (see Figure 9-9). 

_Figure 9-9. Selecting the number of clusters k using the silhouette score_ 

As you can see, this visualization is much richer than the previous one: although it confirms that _k_ = 4 is a very good choice, it also highlights the fact that _k_ = 5 is quite good as well, and much better than _k_ = 6 or 7. This was not visible when comparing inertias. 

An even more informative visualization is obtained when we plot every instance’s silhouette coefficient, sorted by the clusters they are assigned to and by the value of the coefficient. This is called a _silhouette diagram_ (see Figure 9-10). Each diagram contains one knife shape per cluster. The shape’s height indicates the number of instances in the cluster, and its width represents the sorted silhouette coefficients of the instances in the cluster (wider is better). 

The vertical dashed lines represent the mean silhouette score for each number of clusters. When most of the instances in a cluster have a lower coefficient than this score (i.e., if many of the instances stop short of the dashed line, ending to the left of it), then the cluster is rather bad since this means its instances are much too close to other clusters. Here we can see that when _k_ = 3 or 6, we get bad clusters. But when _k_ = 4 or 5, the clusters look pretty good: most instances extend beyond the dashed line, to the right and closer to 1.0. When _k_ = 4, the cluster at index 1 (the second from the bottom) is rather big. When _k_ = 5, all clusters have similar sizes. So, even though the overall silhouette score from _k_ = 4 is slightly greater than for _k_ = 5, it seems like a good idea to use _k_ = 5 to get clusters of similar sizes. 

**Clustering Algorithms: k-means and DBSCAN | 271** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-10. Analyzing the silhouette diagrams for various values of k_ 

## **Limits of k-means** 

Despite its many merits, most notably being fast and scalable, _k_ -means is not perfect. As we saw, it is necessary to run the algorithm several times to avoid suboptimal solutions, plus you need to specify the number of clusters, which can be quite a hassle. Moreover, _k_ -means does not behave very well when the clusters have varying sizes, different densities, or nonspherical shapes. For example, Figure 9-11 shows how _k_ -means clusters a dataset containing three ellipsoidal clusters of different dimen‐ sions, densities, and orientations. 

As you can see, neither of these solutions is any good. The solution on the left is better, but it still chops off 25% of the middle cluster and assigns it to the cluster on the right. The solution on the right is just terrible, even though its inertia is lower. So, depending on the data, different clustering algorithms may perform better. On these types of elliptical clusters, Gaussian mixture models work great. 

**272 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-11. k-means fails to cluster these ellipsoidal blobs properly_ 

It is important to scale the input features (see Chapter 2) before you run _k_ -means, or the clusters may be very stretched and _k_ -means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally helps _k_ -means. 

Now let’s look at a few ways we can benefit from clustering. We will use _k_ -means, but feel free to experiment with other clustering algorithms. 

## **Using Clustering for Image Segmentation** 

_Image segmentation_ is the task of partitioning an image into multiple segments. There are several variants: 

- In _color segmentation_ , pixels with a similar color get assigned to the same seg‐ ment. This is sufficient in many applications. For example, if you want to analyze satellite images to measure how much total forest area there is in a region, color segmentation may be just fine. 

- In _semantic segmentation_ , all pixels that are part of the same object type get assigned to the same segment. For example, in a self-driving car’s vision system, all pixels that are part of a pedestrian’s image might be assigned to the “pedes‐ trian” segment (there would be one segment containing all the pedestrians). 

- In _instance segmentation_ , all pixels that are part of the same individual object are assigned to the same segment. In this case there would be a different segment for each pedestrian. 

**Clustering Algorithms: k-means and DBSCAN | 273** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

The state of the art in semantic or instance segmentation today is achieved using complex architectures based on convolutional neural networks (see Chapter 14). In this chapter we are going to focus on the (much simpler) color segmentation task, using _k_ -means. 

We’ll start by importing the Pillow package (successor to the Python Imaging Library, PIL), which we’ll then use to load the _ladybug.png_ image (see the upper-left image in Figure 9-12), assuming it’s located at `filepath` : 

```
>>> importPIL
>>> image=np.asarray(PIL.Image.open(filepath))
>>> image.shape
(533, 800, 3)
```

The image is represented as a 3D array. The first dimension’s size is the height; the second is the width; and the third is the number of color channels, in this case red, green, and blue (RGB). In other words, for each pixel there is a 3D vector containing the intensities of red, green, and blue as unsigned 8-bit integers between 0 and 255. Some images may have fewer channels (such as grayscale images, which only have one), and some images may have more channels (such as images with an additional _alpha channel_ for transparency, or satellite images, which often contain channels for additional light frequencies (like infrared). 

The following code reshapes the array to get a long list of RGB colors, then it clusters these colors using _k_ -means with eight clusters. It creates a `segmented_img` array containing the nearest cluster center for each pixel (i.e., the mean color of each pixel’s cluster), and lastly it reshapes this array to the original image shape. The third line uses advanced NumPy indexing; for example, if the first 10 labels in `kmeans_.labels_` are equal to 1, then the first 10 colors in `segmented_img` are equal to `kmeans.cluster_centers_[1]` : 

```
X=image.reshape(-1, 3)
kmeans=KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img=kmeans.cluster_centers_[kmeans.labels_]
segmented_img=segmented_img.reshape(image.shape)
```

This outputs the image shown in the upper right of Figure 9-12. You can experiment with various numbers of clusters, as shown in the figure. When you use fewer than eight clusters, notice that the ladybug’s flashy red color fails to get a cluster of its own: it gets merged with colors from the environment. This is because _k_ -means prefers clusters of similar sizes. The ladybug is small—much smaller than the rest of the image—so even though its color is flashy, _k_ -means fails to dedicate a cluster to it. 

**274 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-12. Image segmentation using k-means with various numbers of color clusters_ 

That wasn’t too hard, was it? Now let’s look at another application of clustering. 

## **Using Clustering for Semi-Supervised Learning** 

Another use case for clustering is in semi-supervised learning, when we have plenty of unlabeled instances and very few labeled instances. In this section, we’ll use the digits dataset, which is a simple MNIST-like dataset containing 1,797 grayscale 8 × 8 images representing the digits 0 to 9. First, let’s load and split the dataset (it’s already shuffled): 

**`from sklearn.datasets import`** `load_digits` 

`X_digits, y_digits = load_digits(return_X_y=` **`True`** `) X_train, y_train = X_digits[:1400], y_digits[:1400] X_test, y_test = X_digits[1400:], y_digits[1400:]` 

We will pretend we only have labels for 50 instances. To get a baseline performance, let’s train a logistic regression model on these 50 labeled instances: 

**`from sklearn.linear_model import`** `LogisticRegression` 

`n_labeled = 50 log_reg = LogisticRegression(max_iter=10_000) log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])` 

**Clustering Algorithms: k-means and DBSCAN | 275** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

We can then measure the accuracy of this model on the test set (note that the test set must be labeled): 

**`>>>`** `log_reg.score(X_test, y_test) 0.7481108312342569` 

The model’s accuracy is just 74.8%. That’s not great: indeed, if you try training the model on the full training set, you will find that it will reach about 90.7% accuracy. Let’s see how we can do better. First, let’s cluster the training set into 50 clusters. Then, for each cluster, we’ll find the image closest to the centroid. We’ll call these images the _representative images_ : 

`k = 50 kmeans = KMeans(n_clusters=k, random_state=42) X_digits_dist = kmeans.fit_transform(X_train) representative_digit_idx = np.argmin(X_digits_dist, axis=0) X_representative_digits = X_train[representative_digit_idx]` 

Figure 9-13 shows the 50 representative images. 

_Figure 9-13. Fifty representative digit images (one per cluster)_ 

Let’s look at each image and manually label them: 

`y_representative_digits = np.array([1, 3, 6, 0, 7, 9, 2, ..., 5, 1, 9, 9, 3, 7])` 

Now we have a dataset with just 50 labeled instances, but instead of being random instances, each of them is a representative image of its cluster. Let’s see if the perfor‐ mance is any better: 

**`>>>`** `log_reg = LogisticRegression(max_iter=10_000)` **`>>>`** `log_reg.fit(X_representative_digits, y_representative_digits)` **`>>>`** `log_reg.score(X_test, y_test) 0.8488664987405542` 

Wow! We jumped from 74.8% accuracy to 84.9%, although we are still only training the model on 50 instances. Since it is often costly and painful to label instances, especially when it has to be done manually by experts, it is a good idea to label representative instances rather than just random instances. 

**276 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

But perhaps we can go one step further: what if we propagated the labels to all the other instances in the same cluster? This is called _label propagation_ : 

```
y_train_propagated=np.empty(len(X_train), dtype=np.int64)
foriinrange(k):
y_train_propagated[kmeans.labels_==i] =y_representative_digits[i]
```

Now let’s train the model again and look at its performance: 

```
>>> log_reg=LogisticRegression()
>>> log_reg.fit(X_train, y_train_propagated)
>>> log_reg.score(X_test, y_test)
0.8942065491183879
```

We got another significant accuracy boost! Let’s see if we can do even better by ignoring the 1% of instances that are farthest from their cluster center: this should eliminate some outliers. The following code first computes the distance from each instance to its closest cluster center, then for each cluster it sets the 1% largest distances to –1. Lastly, it creates a set without these instances marked with a –1 distance: 

```
percentile_closest=99
```

```
X_cluster_dist=X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
foriinrange(k):
in_cluster= (kmeans.labels_==i)
cluster_dist=X_cluster_dist[in_cluster]
cutoff_distance=np.percentile(cluster_dist, percentile_closest)
above_cutoff= (X_cluster_dist>cutoff_distance)
X_cluster_dist[in_cluster&above_cutoff] =-1
```

```
partially_propagated= (X_cluster_dist!=-1)
X_train_partially_propagated=X_train[partially_propagated]
y_train_partially_propagated=y_train_propagated[partially_propagated]
```

Now let’s train the model again on this partially propagated dataset and see what accuracy we get: 

```
>>> log_reg=LogisticRegression(max_iter=10_000)
>>> log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
>>> log_reg.score(X_test, y_test)
0.9093198992443325
```

Nice! With just 50 labeled instances (only 5 examples per class on average!) we got 90.9% accuracy, which is actually slightly higher than the performance we got on the fully labeled digits dataset (90.7%). This is partly thanks to the fact that we dropped some outliers, and partly because the propagated labels are actually pretty good—their accuracy is about 97.5%, as the following code shows: 

```
>>> (y_train_partially_propagated==y_train[partially_propagated]).mean()
0.9755555555555555
```

**Clustering Algorithms: k-means and DBSCAN | 277** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Scikit-Learn also offers two classes that can propagate labels automatically: `LabelSpreading` and `LabelPropagation` in the `sklearn.semi_supervised` package. Both classes construct a simi‐ larity matrix between all the instances, and iteratively propagate labels from labeled instances to similar unlabeled instances. There’s also a very different class called `SelfTrainingClassifier` in the same package: you give it a base classifier (such as a `RandomForest Classifier` ) and it trains it on the labeled instances, then uses it to predict labels for the unlabeled samples. It then updates the train‐ ing set with the labels it is most confident about, and repeats this process of training and labeling until it cannot add labels anymore. These techniques are not magic bullets, but they can occasionally give your model a little boost. 

## **Active Learning** 

To continue improving your model and your training set, the next step could be to do a few rounds of _active learning_ , which is when a human expert interacts with the learning algorithm, providing labels for specific instances when the algorithm requests them. There are many different strategies for active learning, but one of the most common ones is called _uncertainty sampling_ . Here is how it works: 

**1.** The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled instances. 

**2.** The instances for which the model is most uncertain (i.e., where its estimated probability is lowest) are given to the expert for labeling. 

**3.** You iterate this process until the performance improvement stops being worth the labeling effort. 

Other active learning strategies include labeling the instances that would result in the largest model change or the largest drop in the model’s validation error, or the instances that different models disagree on (e.g., an SVM and a random forest). 

Before we move on to Gaussian mixture models, let’s take a look at DBSCAN, another popular clustering algorithm that illustrates a very different approach based on local density estimation. This approach allows the algorithm to identify clusters of arbitrary shapes. 

## **278 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## **DBSCAN** 

The _density-based spatial clustering of applications with noise_ (DBSCAN) algorithm defines clusters as continuous regions of high density. Here is how it works: 

- For each instance, the algorithm counts how many instances are located within a small distance ε (epsilon) from it. This region is called the instance’s _ε-neighborhood_ . 

- If an instance has at least `min_samples` instances in its ε-neighborhood (includ‐ ing itself), then it is considered a _core instance_ . In other words, core instances are those that are located in dense regions. 

- All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood may include other core instances; therefore, a long sequence of neighboring core instances forms a single cluster. 

- Any instance that is not a core instance and does not have one in its neighbor‐ hood is considered an anomaly. 

This algorithm works well if all the clusters are well separated by low-density regions. The `DBSCAN` class in Scikit-Learn is as simple to use as you might expect. Let’s test it on the moons dataset, introduced in Chapter 5: 

```
fromsklearn.clusterimportDBSCAN
fromsklearn.datasetsimportmake_moons
X, y=make_moons(n_samples=1000, noise=0.05)
dbscan=DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

The labels of all the instances are now available in the `labels_` instance variable: 

```
>>> dbscan.labels_
array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5, [...], 3,  3,  4,  2,  6,  3])
```

Notice that some instances have a cluster index equal to –1, which means that they are considered as anomalies by the algorithm. The indices of the core instances are available in the `core_sample_indices_` instance variable, and the core instances themselves are available in the `components_` instance variable: 

```
>>> dbscan.core_sample_indices_
array([  0,   4,   5,   6,   7,   8,  10,  11, [...], 993, 995, 997, 998, 999])
>>> dbscan.components_
array([[-0.02137124,  0.40618608],
       [-0.84192557,  0.53058695],
       [...],
       [ 0.79419406,  0.60777171]])
```

**Clustering Algorithms: k-means and DBSCAN | 279** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

This clustering is represented in the lefthand plot of Figure 9-14. As you can see, it identified quite a lot of anomalies, plus seven different clusters. How disappointing! Fortunately, if we widen each instance’s neighborhood by increasing `eps` to 0.2, we get the clustering on the right, which looks perfect. Let’s continue with this model. 

_Figure 9-14. DBSCAN clustering using two different neighborhood radiuses_ 

Surprisingly, the `DBSCAN` class does not have a `predict()` method, although it has a `fit_predict()` method. In other words, it cannot predict which cluster a new instance belongs to. This decision was made because different classification algo‐ rithms can be better for different tasks, so the authors decided to let the user choose which one to use. Moreover, it’s not hard to implement. For example, let’s train a `KNeighborsClassifier` : 

**`from sklearn.neighbors import`** `KNeighborsClassifier` 

`knn = KNeighborsClassifier(n_neighbors=50) knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])` 

Now, given a few new instances, we can predict which clusters they most likely belong to and even estimate a probability for each cluster: 

**`>>>`** `X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])` **`>>>`** `knn.predict(X_new) array([1, 0, 1, 0])` **`>>>`** `knn.predict_proba(X_new) array([[0.18, 0.82], [1.  , 0.  ], [0.12, 0.88], [1.  , 0.  ]])` 

Note that we only trained the classifier on the core instances, but we could also have chosen to train it on all the instances, or all but the anomalies: this choice depends on the final task. 

**280 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

The decision boundary is represented in Figure 9-15 (the crosses represent the four instances in `X_new` ). Notice that since there is no anomaly in the training set, the classifier always chooses a cluster, even when that cluster is far away. It is fairly straightforward to introduce a maximum distance, in which case the two instances that are far away from both clusters are classified as anomalies. To do this, use the `kneighbors()` method of the `KNeighborsClassifier` . Given a set of instances, it returns the distances and the indices of the _k_ -nearest neighbors in the training set (two matrices, each with _k_ columns): 

**`>>>`** `y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)` **`>>>`** `y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]` **`>>>`** `y_pred[y_dist > 0.2] = -1` **`>>>`** `y_pred.ravel() array([-1,  0,  1, -1])` 

_Figure 9-15. Decision boundary between two clusters_ 

In short, DBSCAN is a very simple yet powerful algorithm capable of identifying any number of clusters of any shape. It is robust to outliers, and it has just two hyperparameters ( `eps` and `min_samples` ). If the density varies significantly across the clusters, however, or if there’s no sufficiently low-density region around some clusters, DBSCAN can struggle to capture all the clusters properly. Moreover, its computational complexity is roughly _O_ ( _m_[2] _n_ ), so it does not scale well to large datasets. 

You may also want to try _hierarchical DBSCAN_ (HDBSCAN), which is implemented in the `scikit-learn-contrib` project, as it is usually better than DBSCAN at finding clusters of varying densities. 

**Clustering Algorithms: k-means and DBSCAN | 281** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## **Other Clustering Algorithms** 

Scikit-Learn implements several more clustering algorithms that you should take a look at. I cannot cover them all in detail here, but here is a brief overview: 

## _Agglomerative clustering_ 

A hierarchy of clusters is built from the bottom up. Think of many tiny bubbles floating on water and gradually attaching to each other until there’s one big group of bubbles. Similarly, at each iteration, agglomerative clustering connects the nearest pair of clusters (starting with individual instances). If you drew a tree with a branch for every pair of clusters that merged, you would get a binary tree of clusters, where the leaves are the individual instances. This approach can capture clusters of various shapes; it also produces a flexible and informative cluster tree instead of forcing you to choose a particular cluster scale, and it can be used with any pairwise distance. It can scale nicely to large numbers of instances if you provide a connectivity matrix, which is a sparse _m_ × _m_ matrix that indicates which pairs of instances are neighbors (e.g., returned by `sklearn.neighbors.kneighbors_graph()` ). Without a connectivity matrix, the algorithm does not scale well to large datasets. 

## _BIRCH_ 

The balanced iterative reducing and clustering using hierarchies (BIRCH) algo‐ rithm was designed specifically for very large datasets, and it can be faster than batch _k_ -means, with similar results, as long as the number of features is not too large (<20). During training, it builds a tree structure containing just enough information to quickly assign each new instance to a cluster, without having to store all the instances in the tree: this approach allows it to use limited memory while handling huge datasets. 

## _Mean-shift_ 

This algorithm starts by placing a circle centered on each instance; then for each circle it computes the mean of all the instances located within it, and it shifts the circle so that it is centered on the mean. Next, it iterates this mean-shifting step until all the circles stop moving (i.e., until each of them is centered on the mean of the instances it contains). Mean-shift shifts the circles in the direction of higher density, until each of them has found a local density maximum. Finally, all the instances whose circles have settled in the same place (or close enough) are assigned to the same cluster. Mean-shift has some of the same features as DBSCAN, like how it can find any number of clusters of any shape, it has very few hyperparameters (just one—the radius of the circles, called the _bandwidth_ ), and it relies on local density estimation. But unlike DBSCAN, mean-shift tends to chop clusters into pieces when they have internal density variations. Unfortu‐ nately, its computational complexity is _O_ ( _m_[2] _n_ ), so it is not suited for large datasets. 

## **282 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## _Affinity propagation_ 

In this algorithm, instances repeatedly exchange messages between one another until every instance has elected another instance (or itself) to represent it. These elected instances are called _exemplars_ . Each exemplar and all the instances that elected it form one cluster. In real-life politics, you typically want to vote for a candidate whose opinions are similar to yours, but you also want them to win the election, so you might choose a candidate you don’t fully agree with, but who is more popular. You typically evaluate popularity through polls. Affinity propagation works in a similar way, and it tends to choose exemplars located near the center of clusters, similar to _k_ -means. But unlike with _k_ -means, you don’t have to pick a number of clusters ahead of time: it is determined during training. Moreover, affinity propagation can deal nicely with clusters of different sizes. Sadly, this algorithm has a computational complexity of _O_ ( _m_[2] ), so it is not suited for large datasets. 

## _Spectral clustering_ 

This algorithm takes a similarity matrix between the instances and creates a lowdimensional embedding from it (i.e., it reduces the matrix’s dimensionality), then it uses another clustering algorithm in this low-dimensional space (Scikit-Learn’s implementation uses _k_ -means). Spectral clustering can capture complex cluster structures, and it can also be used to cut graphs (e.g., to identify clusters of friends on a social network). It does not scale well to large numbers of instances, and it does not behave well when the clusters have very different sizes. 

Now let’s dive into Gaussian mixture models, which can be used for density estima‐ tion, clustering, and anomaly detection. 

## **Gaussian Mixtures** 

A _Gaussian mixture model_ (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. All the instances generated from a single Gaussian distri‐ bution form a cluster that typically looks like an ellipsoid. Each cluster can have a different ellipsoidal shape, size, density, and orientation, just like in Figure 9-11. When you observe an instance, you know it was generated from one of the Gaussian distributions, but you are not told which one, and you do not know what the parameters of these distributions are. 

There are several GMM variants. In the simplest variant, implemented in the `GaussianMixture` class, you must know in advance the number _k_ of Gaussian dis‐ tributions. The dataset **X** is assumed to have been generated through the following probabilistic process: 

**Gaussian Mixtures | 283** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

- For each instance, a cluster is picked randomly from among _k_ clusters. The probability of choosing the _j_[th] cluster is the cluster’s weight _ϕ_[(] _[j]_[)] .[6] The index of the cluster chosen for the _i_[th] instance is noted _z_[(] _[i]_[)] . 

- If the _i_[th] instance was assigned to the _j_[th] cluster (i.e., _z_[(] _[i]_[)] = _j_ ), then the location **x**[(] _[i]_[)] of this instance is sampled randomly from the Gaussian distribution with mean **μ**[(] _[j]_[)] and covariance matrix **Σ**[(] _[j]_[)] . This is noted **x**[(] _[i]_[)] ~ N ( **μ**[(] _[j]_[)] , **Σ**[(] _[j]_[)] ). 

So what can you do with such a model? Well, given the dataset **X** , you typically want to start by estimating the weights **ϕ** and all the distribution parameters **μ**[(1)] to **μ**[(] _[k]_[)] and **Σ**[(1)] to **Σ**[(] _[k]_[)] . Scikit-Learn’s `GaussianMixture` class makes this super easy: 

```
fromsklearn.mixtureimportGaussianMixture
```

```
gm=GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
```

Let’s look at the parameters that the algorithm estimated: 

```
>>> gm.weights_
array([0.39025715, 0.40007391, 0.20966893])
>>> gm.means_
array([[ 0.05131611,  0.07521837],
       [-1.40763156,  1.42708225],
       [ 3.39893794,  1.05928897]])
>>> gm.covariances_
array([[[ 0.68799922,  0.79606357],
        [ 0.79606357,  1.21236106]],
       [[ 0.63479409,  0.72970799],
        [ 0.72970799,  1.1610351 ]],
       [[ 1.14833585, -0.03256179],
        [-0.03256179,  0.95490931]]])
```

Great, it worked fine! Indeed, two of the three clusters were generated with 500 instances each, while the third cluster only contains 250 instances. So the true cluster weights are 0.4, 0.4, and 0.2, respectively, and that’s roughly what the algorithm found. Similarly, the true means and covariance matrices are quite close to those found by the algorithm. But how? This class relies on the _expectation-maximization_ (EM) algorithm, which has many similarities with the _k_ -means algorithm: it also ini‐ tializes the cluster parameters randomly, then it repeats two steps until convergence, first assigning instances to clusters (this is called the _expectation step_ ) and then updating the clusters (this is called the _maximization step_ ). Sounds familiar, right? In the context of clustering, you can think of EM as a generalization of _k_ -means that not only finds the cluster centers ( **μ**[(1)] to **μ**[(] _[k]_[)] ), but also their size, shape, and 

**==> picture [170 x 9] intentionally omitted <==**

## **284 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

orientation ( **Σ**[(1)] to **Σ**[(] _[k]_[)] ), as well as their relative weights ( _ϕ_[(1)] to _ϕ_[(] _[k]_[)] ). Unlike _k_ -means, though, EM uses soft cluster assignments, not hard assignments. For each instance, during the expectation step, the algorithm estimates the probability that it belongs to each cluster (based on the current cluster parameters). Then, during the maximi‐ zation step, each cluster is updated using _all_ the instances in the dataset, with each instance weighted by the estimated probability that it belongs to that cluster. These probabilities are called the _responsibilities_ of the clusters for the instances. During the maximization step, each cluster’s update will mostly be impacted by the instances it is most responsible for. 

Unfortunately, just like _k_ -means, EM can end up converging to poor solutions, so it needs to be run several times, keeping only the best solution. This is why we set `n_init` to 10. Be careful: by default `n_init` is set to 1. 

You can check whether or not the algorithm converged and how many iterations it took: 

**`>>>`** `gm.converged_ True` **`>>>`** `gm.n_iter_ 4` 

Now that you have an estimate of the location, size, shape, orientation, and relative weight of each cluster, the model can easily assign each instance to the most likely cluster (hard clustering) or estimate the probability that it belongs to a particular cluster (soft clustering). Just use the `predict()` method for hard clustering, or the `predict_proba()` method for soft clustering: 

**`>>>`** `gm.predict(X) array([0, 0, 1, ..., 2, 2, 2])` **`>>>`** `gm.predict_proba(X).round(3) array([[0.977, 0.   , 0.023], [0.983, 0.001, 0.016], [0.   , 1.   , 0.   ], ..., [0.   , 0.   , 1.   ], [0.   , 0.   , 1.   ], [0.   , 0.   , 1.   ]])` 

A Gaussian mixture model is a _generative model_ , meaning you can sample new instances from it (note that they are ordered by cluster index): 

**Gaussian Mixtures | 285** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

**`>>>`** `X_new, y_new = gm.sample(6)` **`>>>`** `X_new array([[-0.86944074, -0.32767626], [ 0.29836051,  0.28297011], [-2.8014927 , -0.09047309], [ 3.98203732,  1.49951491], [ 3.81677148,  0.53095244], [ 2.84104923, -0.73858639]])` **`>>>`** `y_new array([0, 0, 1, 2, 2, 2])` 

It is also possible to estimate the density of the model at any given location. This is achieved using the `score_samples()` method: for each instance it is given, this method estimates the log of the _probability density function_ (PDF) at that location. The greater the score, the higher the density: 

**`>>>`** `gm.score_samples(X).round(2) array([-2.61, -3.57, -3.33, ..., -3.51, -4.4 , -3.81])` 

If you compute the exponential of these scores, you get the value of the PDF at the location of the given instances. These are not probabilities, but probability _densities_ : they can take on any positive value, not just a value between 0 and 1. To estimate the probability that an instance will fall within a particular region, you would have to integrate the PDF over that region (if you do so over the entire space of possible instance locations, the result will be 1). 

Figure 9-16 shows the cluster means, the decision boundaries (dashed lines), and the density contours of this model. 

_Figure 9-16. Cluster means, decision boundaries, and density contours of a trained Gaussian mixture model_ 

## **286 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Nice! The algorithm clearly found an excellent solution. Of course, we made its task easy by generating the data using a set of 2D Gaussian distributions (unfortunately, real-life data is not always so Gaussian and low-dimensional). We also gave the algorithm the correct number of clusters. When there are many dimensions, or many clusters, or few instances, EM can struggle to converge to the optimal solution. You might need to reduce the difficulty of the task by limiting the number of parameters that the algorithm has to learn. One way to do this is to limit the range of shapes and orientations that the clusters can have. This can be achieved by imposing constraints on the covariance matrices. To do this, set the `covariance_type` hyperparameter to one of the following values: 

## `"spherical"` 

All clusters must be spherical, but they can have different diameters (i.e., differ‐ ent variances). 

## `"diag"` 

Clusters can take on any ellipsoidal shape of any size, but the ellipsoid’s axes must be parallel to the coordinate axes (i.e., the covariance matrices must be diagonal). 

## `"tied"` 

All clusters must have the same ellipsoidal shape, size, and orientation (i.e., all clusters share the same covariance matrix). 

By default, `covariance_type` is equal to `"full"` , which means that each cluster can take on any shape, size, and orientation (it has its own unconstrained cova‐ riance matrix). Figure 9-17 plots the solutions found by the EM algorithm when `covariance_type` is set to `"tied"` or `"spherical"` . 

_Figure 9-17. Gaussian mixtures for tied clusters (left) and spherical clusters (right)_ 

**Gaussian Mixtures | 287** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

The computational complexity of training a `GaussianMixture` model depends on the number of instances _m_ , the number of dimensions _n_ , the number of clusters _k_ , and the constraints on the covariance matrices. If `covariance_type` is `"spherical"` or `"diag"` , it is _O_ ( _kmn_ ), assuming the data has a clustering structure. If `covariance_type` is `"tied"` or `"full"` , it is _O_ ( _kmn_[2] + _kn_[3] ), so it will not scale to large numbers of features. 

Gaussian mixture models can also be used for anomaly detection. We’ll see how in the next section. 

## **Using Gaussian Mixtures for Anomaly Detection** 

Using a Gaussian mixture model for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly. You must define what density threshold you want to use. For example, in a manufacturing company that tries to detect defective products, the ratio of defective products is usually well known. Say it is equal to 2%. You then set the density threshold to be the value that results in having 2% of the instances located in areas below that threshold density. If you notice that you get too many false positives (i.e., perfectly good products that are flagged as defective), you can lower the threshold. Conversely, if you have too many false negatives (i.e., defective products that the system does not flag as defective), you can increase the threshold. This is the usual precision/recall trade-off (see Chapter 3). Here is how you would identify the outliers using the second percentile lowest density as the threshold (i.e., approximately 2% of the instances will be flagged as anomalies): 

`densities = gm.score_samples(X) density_threshold = np.percentile(densities, 2) anomalies = X[densities < density_threshold]` 

## Figure 9-18 represents these anomalies as stars. 

A closely related task is _novelty detection_ : it differs from anomaly detection in that the algorithm is assumed to be trained on a “clean” dataset, uncontaminated by outliers, whereas anomaly detection does not make this assumption. Indeed, outlier detection is often used to clean up a dataset. 

Gaussian mixture models try to fit all the data, including the outli‐ ers; if you have too many of them this will bias the model’s view of “normality”, and some outliers may wrongly be considered as normal. If this happens, you can try to fit the model once, use it to detect and remove the most extreme outliers, then fit the model again on the cleaned-up dataset. Another approach is to use robust covariance estimation methods (see the `EllipticEnvelope` class). 

## **288 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-18. Anomaly detection using a Gaussian mixture model_ 

Just like _k_ -means, the `GaussianMixture` algorithm requires you to specify the number of clusters. So how can you find that number? 

## **Selecting the Number of Clusters** 

With _k_ -means, you can use the inertia or the silhouette score to select the appropriate number of clusters. But with Gaussian mixtures, it is not possible to use these metrics because they are not reliable when the clusters are not spherical or have different sizes. Instead, you can try to find the model that minimizes a _theoretical information criterion_ , such as the _Bayesian information criterion_ (BIC) or the _Akaike information criterion_ (AIC), defined in Equation 9-1. 

_Equation 9-1. Bayesian information criterion (BIC) and Akaike information criterion (AIC)_ 

**==> picture [126 x 35] intentionally omitted <==**

In these equations: 

- _m_ is the number of instances, as always. 

- _p_ is the number of parameters learned by the model. 

- ℒ is the maximized value of the _likelihood function_ of the model. 

**Gaussian Mixtures | 289** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Both the BIC and the AIC penalize models that have more parameters to learn (e.g., more clusters) and reward models that fit the data well. They often end up selecting the same model. When they differ, the model selected by the BIC tends to be simpler (fewer parameters) than the one selected by the AIC, but tends to not fit the data quite as well (this is especially true for larger datasets). 

## **Likelihood Function** 

The terms “probability” and “likelihood” are often used interchangeably in everyday language, but they have very different meanings in statistics. Given a statistical model with some parameters **θ** , the word “probability” is used to describe how plausible a future outcome **x** is (knowing the parameter values **θ** ), while the word “likelihood” is used to describe how plausible a particular set of parameter values **θ** are, after the outcome **x** is known. 

Consider a 1D mixture model of two Gaussian distributions centered at –4 and +1. For simplicity, this toy model has a single parameter _θ_ that controls the standard devi‐ ations of both distributions. The top-left contour plot in Figure 9-19 shows the entire model _f_ ( _x_ ; _θ_ ) as a function of both _x_ and _θ_ . To estimate the probability distribution of a future outcome _x_ , you need to set the model parameter _θ_ . For example, if you set _θ_ to 1.3 (the horizontal line), you get the probability density function _f_ ( _x_ ; _θ_ =1.3) shown in the lower-left plot. Say you want to estimate the probability that _x_ will fall between –2 and +2. You must calculate the integral of the PDF on this range (i.e., the surface of the shaded region). But what if you don’t know _θ_ , and instead if you have observed a single instance _x_ =2.5 (the vertical line in the upper-left plot)? In this case, you get the likelihood function ℒ ( _θ_ | _x_ =2.5)=f( _x_ =2.5; _θ_ ), represented in the upper-right plot. 

_Figure 9-19. A model’s parametric function (top left), and some derived functions: a PDF (lower left), a likelihood function (top right), and a log likelihood function (lower right)_ 

## **290 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

In short, the PDF is a function of _x_ (with _θ_ fixed), while the likelihood function is a function of _θ_ (with _x_ fixed). It is important to understand that the likelihood function is _not_ a probability distribution: if you integrate a probability distribution over all possible values of _x_ , you always get 1, but if you integrate the likelihood function over all possible values of _θ_ the result can be any positive value. 

Given a dataset **X** , a common task is to try to estimate the most likely values for the model parameters. To do this, you must find the values that maximize the likelihood function, given **X** . In this example, if you have observed a single instance _x_ =2.5, the _maximum likelihood estimate_ (MLE) of _θ_ is θ ≈1.66. If a prior probability distribution _g_ over _θ_ exists, it is possible to take it into account by maximizing ℒ ( _θ_ | _x_ )g( _θ_ ) rather than just maximizing ℒ ( _θ_ | _x_ ). This is called _maximum a-posteriori_ (MAP) estimation. Since MAP constrains the parameter values, you can think of it as a regularized version of MLE. 

Notice that maximizing the likelihood function is equivalent to maximizing its loga‐ rithm (represented in the lower-right plot in Figure 9-19). Indeed, the logarithm is a strictly increasing function, so if _θ_ maximizes the log likelihood, it also maximizes the likelihood. It turns out that it is generally easier to maximize the log likelihood. For example, if you observed several independent instances _x_[(1)] to _x_[(] _[m]_[)] , you would need to find the value of _θ_ that maximizes the product of the individual likelihood functions. But it is equivalent, and much simpler, to maximize the sum (not the product) of the log likelihood functions, thanks to the magic of the logarithm which converts products into sums: log( _ab_ ) = log( _a_ ) + log( _b_ ). 

Once you have estimated θ , the value of _θ_ that maximizes the likelihood function, then you are ready to compute ℒ = ℒ θ , X , which is the value used to compute the AIC and BIC; you can think of it as a measure of how well the model fits the data. 

To compute the BIC and AIC, call the `bic()` and `aic()` methods: 

```
>>> gm.bic(X)
8189.747000497186
>>> gm.aic(X)
8102.521720382148
```

Figure 9-20 shows the BIC for different numbers of clusters _k_ . As you can see, both the BIC and the AIC are lowest when _k_ =3, so it is most likely the best choice. 

**Gaussian Mixtures | 291** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-20. AIC and BIC for different numbers of clusters k_ 

## **Bayesian Gaussian Mixture Models** 

Rather than manually searching for the optimal number of clusters, you can use the `BayesianGaussianMixture` class, which is capable of giving weights equal (or close) to zero to unnecessary clusters. Set the number of clusters `n_components` to a value that you have good reason to believe is greater than the optimal number of clusters (this assumes some minimal knowledge about the problem at hand), and the algorithm will eliminate the unnecessary clusters automatically. For example, let’s set the number of clusters to 10 and see what happens: 

**`>>> from sklearn.mixture import`** `BayesianGaussianMixture` **`>>>`** `bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)` **`>>>`** `bgm.fit(X)` **`>>>`** `bgm.weights_.round(2) array([0.4 , 0.21, 0.4 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])` 

Perfect: the algorithm automatically detected that only three clusters are needed, and the resulting clusters are almost identical to the ones in Figure 9-16. 

A final note about Gaussian mixture models: although they work great on clusters with ellipsoidal shapes, they don’t do so well with clusters of very different shapes. For example, let’s see what happens if we use a Bayesian Gaussian mixture model to cluster the moons dataset (see Figure 9-21). 

Oops! The algorithm desperately searched for ellipsoids, so it found eight different clusters instead of two. The density estimation is not too bad, so this model could perhaps be used for anomaly detection, but it failed to identify the two moons. To conclude this chapter, let’s take a quick look at a few algorithms capable of dealing with arbitrarily shaped clusters. 

**292 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

_Figure 9-21. Fitting a Gaussian mixture to nonellipsoidal clusters_ 

## **Other Algorithms for Anomaly and Novelty Detection** 

Scikit-Learn implements other algorithms dedicated to anomaly detection or novelty detection: 

## _Fast-MCD (minimum covariance determinant)_ 

Implemented by the `EllipticEnvelope` class, this algorithm is useful for outlier detection, in particular to clean up a dataset. It assumes that the normal instances (inliers) are generated from a single Gaussian distribution (not a mixture). It also assumes that the dataset is contaminated with outliers that were not generated from this Gaussian distribution. When the algorithm estimates the parameters of the Gaussian distribution (i.e., the shape of the elliptic envelope around the inliers), it is careful to ignore the instances that are most likely outliers. This technique gives a better estimation of the elliptic envelope and thus makes the algorithm better at identifying the outliers. 

## _Isolation forest_ 

This is an efficient algorithm for outlier detection, especially in high-dimensional datasets. The algorithm builds a random forest in which each decision tree is grown randomly: at each node, it picks a feature randomly, then it picks a random threshold value (between the min and max values) to split the dataset in two. The dataset gradually gets chopped into pieces this way, until all instances end up isolated from the other instances. Anomalies are usually far from other instances, so on average (across all the decision trees) they tend to get isolated in fewer steps than normal instances. 

## _Local outlier factor (LOF)_ 

This algorithm is also good for outlier detection. It compares the density of instances around a given instance to the density around its neighbors. An anom‐ aly is often more isolated than its _k_ -nearest neighbors. 

**Gaussian Mixtures | 293** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

## _One-class SVM_ 

This algorithm is better suited for novelty detection. Recall that a kernelized SVM classifier separates two classes by first (implicitly) mapping all the instances to a high-dimensional space, then separating the two classes using a linear SVM classifier within this high-dimensional space (see Chapter 5). Since we just have one class of instances, the one-class SVM algorithm instead tries to separate the instances in high-dimensional space from the origin. In the original space, this will correspond to finding a small region that encompasses all the instances. If a new instance does not fall within this region, it is an anomaly. There are a few hyperparameters to tweak: the usual ones for a kernelized SVM, plus a margin hyperparameter that corresponds to the probability of a new instance being mistakenly considered as novel when it is in fact normal. It works great, especially with high-dimensional datasets, but like all SVMs it does not scale to large datasets. 

_PCA and other dimensionality reduction techniques with an_ `inverse_transform()` _method_ 

If you compare the reconstruction error of a normal instance with the recon‐ struction error of an anomaly, the latter will usually be much larger. This is a simple and often quite efficient anomaly detection approach (see this chapter’s exercises for an example). 

## **Exercises** 

**1.** How would you define clustering? Can you name a few clustering algorithms? 

**2.** What are some of the main applications of clustering algorithms? 

**3.** Describe two techniques to select the right number of clusters when using _k_ -means. 

**4.** What is label propagation? Why would you implement it, and how? 

**5.** Can you name two clustering algorithms that can scale to large datasets? And two that look for regions of high density? 

**6.** Can you think of a use case where active learning would be useful? How would you implement it? 

**7.** What is the difference between anomaly detection and novelty detection? 

**8.** What is a Gaussian mixture? What tasks can you use it for? 

**9.** Can you name two techniques to find the right number of clusters when using a Gaussian mixture model? 

**10.** The classic Olivetti faces dataset contains 400 grayscale 64 × 64–pixel images of faces. Each image is flattened to a 1D vector of size 4,096. Forty different people were photographed (10 times each), and the usual task is to train a model that 

## **294 | Chapter 9: Unsupervised Learning Techniques** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

can predict which person is represented in each picture. Load the dataset using the `sklearn.datasets.fetch_olivetti_faces()` function, then split it into a training set, a validation set, and a test set (note that the dataset is already scaled between 0 and 1). Since the dataset is quite small, you will probably want to use stratified sampling to ensure that there are the same number of images per person in each set. Next, cluster the images using _k_ -means, and ensure that you have a good number of clusters (using one of the techniques discussed in this chapter). Visualize the clusters: do you see similar faces in each cluster? 

**11.** Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set. Next, use _k_ -means as a dimensionality reduction tool, and train a classifier on the reduced set. Search for the number of clusters that allows the classifier to get the best performance: what performance can you reach? What if you append the features from the reduced set to the original features (again, searching for the best number of clusters)? 

**12.** Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should probably reduce the dataset’s dimensionality (e.g., use PCA, preserving 99% of the variance). Use the model to generate some new faces (using the `sample()` method), and visualize them (if you used PCA, you will need to use its `inverse_transform()` method). Try to modify some images (e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e., compare the output of the `score_samples()` method for normal images and for anomalies). 

**13.** Some dimensionality reduction techniques can also be used for anomaly detec‐ tion. For example, take the Olivetti faces dataset and reduce it with PCA, preserv‐ ing 99% of the variance. Then compute the reconstruction error for each image. Next, take some of the modified images you built in the previous exercise and look at their reconstruction error: notice how much larger it is. If you plot a reconstructed image, you will see why: it tries to reconstruct a normal face. 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

**Exercises | 295** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:15:38. 

