# Unsupervised Machine Learning — Clustering

> **Interview prep notes (mid-level Data Scientist)**
> Covers: fundamentals → K-Means → K-Medoids/PAM → Hierarchical → DBSCAN → GMM → evaluation → comparison → interview Q&A.
> Each algorithm follows the same drill: **What it is → How it works (math + step-by-step) → Worked example → Variants → Hyperparameters → Advantages → Limitations → Code → What interviewers dig into.**

---

## Table of Contents

1. [Unsupervised Learning — The Big Picture](#1-unsupervised-learning--the-big-picture)
2. [Clustering Fundamentals](#2-clustering-fundamentals)
3. [Distance & Similarity Measures](#3-distance--similarity-measures)
4. [Evaluating Clustering (Goodness Metrics)](#4-evaluating-clustering-goodness-metrics)
5. [K-Means](#5-k-means)
6. [K-Medoids / PAM](#6-k-medoids--pam)
7. [Hierarchical Clustering](#7-hierarchical-clustering)
8. [DBSCAN (and HDBSCAN / OPTICS)](#8-dbscan-and-hdbscan--optics)
9. [Gaussian Mixture Models (soft clustering)](#9-gaussian-mixture-models-soft-clustering)
10. [Preprocessing Checklist Before Clustering](#10-preprocessing-checklist-before-clustering)
11. [Algorithm Comparison Cheat Sheet](#11-algorithm-comparison-cheat-sheet)
12. [Corrections / Clarifications to Common Notes](#12-corrections--clarifications-to-common-notes)
13. [Interview Questions & Answers](#13-interview-questions--answers)
14. [Rapid-Fire One-Liners](#14-rapid-fire-one-liners)

---

## 1. Unsupervised Learning — The Big Picture

**Definition:** Learning patterns/structure from data **without labels (no `y`)**. The model discovers structure instead of mapping `X → y`.

### Types of unsupervised learning

| Type | Goal | Examples |
|---|---|---|
| Dimensionality reduction | Compress features, keep structure | PCA, t-SNE, UMAP, autoencoders |
| **Clustering** | Group similar points | K-Means, Hierarchical, DBSCAN, GMM |
| Association rule mining | Find co-occurrence rules | Apriori, FP-Growth ("beer → diapers") |
| Anomaly detection | Find the odd ones out | Isolation Forest, One-Class SVM, DBSCAN noise |
| Generative modeling | Learn `P(X)` and sample from it | GMM, VAE, GANs, diffusion |

### Why clustering matters in practice

- **Customer segmentation** — group customers by behaviour to target offers/pricing.
- **Image segmentation** — group pixels to separate grass / human / animal.
- **Document / topic clustering** — group articles, tickets, search queries.
- **Anomaly & fraud detection** — small or noise clusters = suspicious.
- **Cutting labeling cost (the big one for interviews):**
  Manual labeling of millions of rows is expensive. Instead:
  1. Cluster the unlabeled data into `k` groups.
  2. Manually label 1–5 representative points per cluster.
  3. Propagate the label to the whole cluster → now you have a (noisy) supervised dataset.

  **Reduction factor = N / k.** e.g. 10M points → 10k clusters = **1000× less labeling effort**.
  ⚠️ Caveat interviewers love: this only works if clusters are *pure*. Always spot-check purity on a small hand-labeled sample before trusting propagated labels.

- **Feature engineering** — `cluster_id` as a categorical feature, or distance-to-each-centroid as `k` numeric features, fed into a supervised model.
- **Recommender systems** — cluster users/items to solve cold start and to reduce a sparse user–item matrix.

---

## 2. Clustering Fundamentals

### What is a cluster?

A set of points that are **more similar to each other** than to points in other groups. "Similar" is **problem-specific** — you define it through your distance metric and feature representation.

### Formal partition property (K-Means style / hard clustering)

Given dataset `D = {x1, x2, ..., xn}`, we produce sets `S1, S2, ..., Sk` such that:

- `S1 ∪ S2 ∪ ... ∪ Sk = D` → every point is assigned (exhaustive)
- `Si ∩ Sj = ∅` for all `i ≠ j` → no point in two clusters (mutually exclusive)

*(DBSCAN breaks the first rule — noise points belong to no cluster. GMM/fuzzy break the second — soft membership.)*

### Intra-cluster vs Inter-cluster — the core intuition

| Term | Meaning | We want |
|---|---|---|
| **Intra-cluster distance** | Distance *within* a cluster (compactness) | **SMALL** |
| **Inter-cluster distance** | Distance *between* clusters (separation) | **LARGE** |

> Almost every internal evaluation metric (SSE, Silhouette, Dunn, Davies-Bouldin, Calinski-Harabasz) is some ratio or combination of these two ideas. If you remember only one sentence about clustering evaluation, remember this one.

### Taxonomy of clustering algorithms

| Family | Idea | Algorithms |
|---|---|---|
| **Partitional / centroid-based** | Split into k groups around a center | K-Means, K-Medoids, K-Medians, Mini-batch K-Means |
| **Hierarchical / connectivity-based** | Build a tree of nested clusters | Agglomerative, Divisive (BIRCH for scale) |
| **Density-based** | Dense regions = clusters, sparse = noise | DBSCAN, HDBSCAN, OPTICS, Mean-Shift |
| **Model / distribution-based** | Data generated by a mixture of distributions | GMM (via EM) |
| **Graph / spectral** | Cut a similarity graph | Spectral clustering |
| **Fuzzy / soft** | Points get membership degrees | Fuzzy C-Means |

Also worth knowing: **hard vs soft** clustering, **flat vs hierarchical**, **complete vs partial** (DBSCAN leaves noise unassigned).

### Are all clusters circular/globular?

**No.** Real clusters can be elongated, ring-shaped, crescent-shaped, or nested. This single fact explains most of the K-Means vs DBSCAN comparison in interviews.

---

## 3. Distance & Similarity Measures

Choosing the distance is arguably a bigger decision than choosing the algorithm.

### Euclidean (L2) — default for K-Means

```
d(x, y) = sqrt( Σᵢ (xᵢ - yᵢ)² )
```

- Straight-line distance.
- Good when features are on comparable scales, roughly normally distributed, and equally important.
- ❌ Not robust to outliers; degrades badly in high dimensions (curse of dimensionality).
- **K-Means actually minimizes *squared* Euclidean distance** — that's what makes the mean the optimal centroid.

### Manhattan (L1 / city-block)

```
d(x, y) = Σᵢ |xᵢ - yᵢ|
```

- Sum of absolute coordinate differences.
- More robust to outliers than L2; good for high-dimensional / sparse / discrete data, and when movement along axes is restricted.
- Pairs with **K-Medians** (the median minimizes L1, just as the mean minimizes L2).

### Cosine similarity

```
cos(x, y) = (x · y) / (||x||₂ ||y||₂)        cosine distance = 1 - cos(x, y)
```

- Measures the **angle**, ignores magnitude → use when **direction matters more than magnitude**.
- Standard for **text / TF-IDF / embeddings**, where document length shouldn't dominate.
- 🔑 Interview nugget: on **L2-normalized vectors**, minimizing squared Euclidean distance is *equivalent* to maximizing cosine similarity — so "spherical k-means" = L2-normalize your vectors, then run ordinary K-Means.

### Others you should be able to name

| Metric | Use case |
|---|---|
| Minkowski (Lᵖ) | Generalization: p=1 Manhattan, p=2 Euclidean |
| Pearson correlation distance | Shape of profile matters, not level (gene expression, time series) |
| Jaccard | Binary / set data (market basket, tags) |
| Hamming | Categorical / binary strings |
| Mahalanobis | Accounts for feature covariance & scale |
| Gower | **Mixed numeric + categorical** data |
| DTW | Time series of different lengths / phase shifts |

### Strengths & weaknesses

| Metric | ✔ Strengths | ✘ Weaknesses |
|---|---|---|
| Euclidean | Most common, cheap, works with normally distributed data | Not robust to outliers; bad at capturing "similarity" for text; suffers in high-d |
| Cosine | Robust to magnitude/outliers, great for similarity & sparse text | Ignores magnitude, so ignores true distance; slightly costlier |
| Manhattan | Easy, works with discrete data, captures distance well | Not as good for similarity; still not robust to extreme outliers |

### Curse of dimensionality (must-mention)

As `d` grows, all pairwise distances converge toward the same value — the contrast between "near" and "far" collapses, so distance-based clustering becomes meaningless. **Fix:** feature selection, PCA/UMAP first, or switch to cosine on sparse text.

---

## 4. Evaluating Clustering (Goodness Metrics)

Two families:

- **Internal** (no ground truth) — use the data itself: SSE, Silhouette, Dunn, Davies-Bouldin, Calinski-Harabasz.
- **External** (ground truth available, e.g. benchmark data) — ARI, NMI, Homogeneity/Completeness/V-measure, Purity, Fowlkes-Mallows.

### 4.1 Sum of Squared Errors (SSE) / Inertia / WCSS

```
SSE = Σ_{i=1..k} Σ_{x ∈ Sᵢ} ||x - cᵢ||²
```

Total squared distance of every point to its own centroid = the K-Means objective itself. In sklearn: `kmeans.inertia_`.

- ✔ Simplest, cheapest, directly optimized by K-Means.
- ✘ **Monotonically decreases as k increases** (SSE = 0 when k = n), so it can *never* be used to pick k by minimization — only by the **elbow**.
- ✘ Says nothing about separation; assumes spherical clusters.

### 4.2 Silhouette Score

For a single point `i`:

```
a(i) = mean distance from i to all OTHER points in its OWN cluster       (cohesion)
b(i) = mean distance from i to all points in the NEAREST OTHER cluster   (separation)

s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Silhouette score = mean of `s(i)` over all points. Range **[-1, 1]**:

| Value | Meaning |
|---|---|
| ≈ **+1** | Point is well inside its cluster, clusters well separated |
| ≈ **0** | Point sits on the boundary; clusters overlap / separation not significant |
| ≈ **-1** | Point is probably in the **wrong** cluster |

⚠️ Precision point interviewers check: `b(i)` is the distance to the **single nearest neighbouring cluster**, *not* the average over all other clusters.

**Worked example (1-D).** Cluster A = {1, 2}, Cluster B = {10, 11}. For point `x = 1`:
- `a(1)` = |1−2| = 1
- `b(1)` = mean(|1−10|, |1−11|) = (9 + 11)/2 = 10
- `s(1)` = (10 − 1)/10 = **0.9** → strongly well-clustered.

- ✔ Sensitive to cluster shape, captures separation, works for any distance metric, gives a **per-point diagnostic** (silhouette plot shows which clusters are bad).
- ✘ O(n²) distance computations (sample it for big data), not robust to outliers, still biased toward convex/globular clusters, unreliable for DBSCAN-shaped clusters.

### 4.3 Dunn Index

```
Dunn = (minimum inter-cluster distance) / (maximum intra-cluster diameter)
     = min_{i≠j} d(Cᵢ, Cⱼ) / max_k diam(C_k)
```

- Numerator: over **all pairs of clusters**, the **smallest** separation (the two closest clusters).
- Denominator: over **all clusters**, the **largest** diameter (the widest cluster).
- **Higher is better** (compact clusters, well separated).

⚠️ Very common note error: the numerator is the **MINIMUM** inter-cluster distance, not the maximum. Think "worst-case separation ÷ worst-case compactness".

- ✔ Directly encodes the compactness-vs-separation trade-off; intuitive.
- ✘ **Extremely sensitive to outliers and noise** — a single stray point inflates the denominator or shrinks the numerator and destroys the score. O(n²) cost. (Slide decks that list Dunn as "robust to outliers" are wrong — see §12.)

### 4.4 Davies-Bouldin Index

Average, over clusters, of the worst-case similarity ratio `(scatterᵢ + scatterⱼ) / distance(cᵢ, cⱼ)`.
**Lower is better** (0 = perfect). Cheap — O(n) after centroids. `sklearn.metrics.davies_bouldin_score`.

### 4.5 Calinski-Harabasz (Variance Ratio Criterion)

`(between-cluster dispersion / within-cluster dispersion)`, scaled by degrees of freedom. **Higher is better**, very fast, but biased toward convex clusters and tends to increase with k.

### 4.6 External metrics (when labels exist)

| Metric | Range | Notes |
|---|---|---|
| **Adjusted Rand Index (ARI)** | [-1, 1], 0 = random | Chance-corrected agreement of pairings. Default choice. |
| **Normalized Mutual Information (NMI)** | [0, 1] | Information shared between the two partitions. |
| **Homogeneity / Completeness / V-measure** | [0, 1] | Each cluster one class / each class one cluster / their harmonic mean. |
| **Purity** | [0, 1] | Simple, but rises trivially with k — mention its weakness. |
| **Fowlkes-Mallows** | [0, 1] | Geometric mean of pairwise precision & recall. |

### 4.7 Metric strengths & weaknesses summary

| Metric | ✔ Strengths | ✘ Weaknesses |
|---|---|---|
| SSE / Inertia | Most common, cheap, is the K-Means objective | Always decreases with k; ignores separation; assumes spherical |
| Silhouette | Captures cohesion **and** separation; per-point diagnostic | O(n²); not robust to outliers; biased to convex shapes |
| Dunn | Explicitly compactness + separation | Very sensitive to noise/outliers; O(n²) |
| Davies-Bouldin | Fast, easy to interpret, lower = better | Assumes convex, centroid-based |
| Calinski-Harabasz | Very fast, good for comparing k | Grows with k; convex bias |
| ARI / NMI | Chance-corrected, rigorous | **Need ground-truth labels** |

### 4.8 Beyond metrics — what a senior answer adds

1. **Stability analysis** — bootstrap/subsample the data, re-cluster, measure ARI between runs. Unstable clusters aren't real.
2. **Business validation** — do the segments differ on metrics the business cares about (ARPU, churn, basket size)? A statistically pretty cluster that no one can action is worthless.
3. **Interpretability** — profile each cluster (mean/median per feature vs global mean), give it a name ("price-sensitive weekend shoppers").
4. **Downstream lift** — does adding `cluster_id` improve a supervised model or a campaign's conversion?

---

## 5. K-Means

### 5.1 Intro — the one-liner

> K-Means is an unsupervised, **centroid-based, iterative** partitioning algorithm that splits an unlabeled dataset into a **pre-specified number `k`** of non-overlapping clusters, by minimizing the within-cluster sum of squared distances to each cluster's mean.

`k` is a **hyperparameter** — the user must supply it.

### 5.2 Mathematical formulation

Given `D = {x1, ..., xn}`, find centroids `c1..ck` and sets `S1..Sk` that solve:

```
argmin           Σ_{i=1..k}  Σ_{x ∈ Sᵢ}  || x - cᵢ ||²
 c₁..c_k, S₁..S_k

subject to:   S₁ ∪ ... ∪ S_k = D      and      Sᵢ ∩ Sⱼ = ∅
```

- This is the **intra-cluster distance** (SSE / WCSS) being minimized.
- Solving it **exactly is NP-hard** (exponential, ~O(2ⁿ) in the naive search) → so we use an **approximation algorithm: Lloyd's algorithm**, which converges to a **local optimum**.

Given centroids, the optimal assignment is trivial: **assign each point to its nearest centroid**. Given an assignment, the optimal center is the **mean** of the cluster (that's why it's called K-*Means*). Alternating these two is exactly Lloyd's algorithm — it is a form of **coordinate descent / hard EM**.

### 5.3 How it works — Lloyd's algorithm, step by step

```
Step 1  Choose k (the number of clusters).
Step 2  Initialize k centroids c1..ck  (randomly, or with k-means++).
Step 3  ASSIGNMENT: for each point xi, find the nearest centroid cj
        (argmin_j ||xi - cj||²) and add xi to cluster Sj.
Step 4  UPDATE: recompute each centroid as the MEAN of the points in Sj:
                cj = (1/|Sj|) * Σ_{x ∈ Sj} x
Step 5  Repeat steps 3 and 4 until convergence:
          - no change in assignments, OR
          - centroid movement < tol (sklearn default 1e-4), OR
          - max_iter reached (sklearn default 300).
END     Clusters are ready.
```

**Convergence guarantee:** SSE is non-increasing at every step and there are finitely many partitions → Lloyd's algorithm **always converges**, but only to a **local** minimum, not the global one.

### 5.4 Worked example — and why initialization matters

**Data (1-D):** `X = {0, 1, 10, 11, 20, 21}`, `k = 3`. Obvious truth: `{0,1} {10,11} {20,21}`, for this best separation to compute SSE, first compute their centroids. (0+1)/2=0.5 ,(10+11)/2=11.5, (20+21)/2 =20.5 . Now compute SSE for each cluster.
Cluster 1: (0 - 0.5)² + (1 - 0.5)² = 0.25 + 0.25 = 0.5
Cluster 2: (10 - 10.5)² + (11 - 10.5)² = 0.25 + 0.25 = 0.5
Cluster 3: (20 - 20.5)² + (21 - 20.5)² = 0.25 + 0.25 =0.5
SSE = 0.5 × 3 = **1.5**.

**Bad init:** `c1 = 0, c2 = 1, c3 = 10`

| Iter | Assignment | New centroids |
|---|---|---|
| 1 | `{0} {1} {10,11,20,21}` | 0, 1, 15.5 |
| 2 | `{0} {1} {10,11,20,21}` (10 is 9 from c2, 5.5 from c3) | 0, 1, 15.5 → **converged** |

Final SSE :For each cluster compute squared distance of each point to its centroid and add them.SSE = 0 + 0 + (30.25 + 20.25 + 20.25 + 30.25) = **101** — massively worse than the optimum of 1.5, and the algorithm has *converged*. This is **initialization sensitivity** in one table, and it's the classic whiteboard question.

**Good init:** `c1 = 0, c2 = 10, c3 = 20` → converges immediately to the correct partition with SSE = 1.5.

### 5.5 Initialization: random vs K-Means++

**Problem:** the final clustering depends on the initial centroids (see above).

**Fix 1 — multiple restarts (`n_init`).** Run K-Means several times with different random seeds and keep the run with the lowest SSE (equivalently: small intra-cluster, large inter-cluster distance). Simple and effective; the standard practical safety net.

**Fix 2 — K-Means++ (smart initialization).** Only the initialization changes; steps 3–5 are identical.

```
a) Pick the first centroid c1 uniformly at random from D.
b) For every point xi, compute D(xi) = distance to its NEAREST already-chosen centroid.
c) Pick the next centroid from the remaining points with probability
   proportional to D(xi)²   (squared distance).
d) Repeat (b)-(c) until k centroids are chosen. Then run standard Lloyd.
```

# K-Means++ Initialization Example

**Dataset:** $X = \{0, 1, 10, 11, 20, 21\}$
**Target Clusters:** $k = 3$

---

## Step 1: Choose First Center ($C_1$)
Pick one point uniformly at random from the dataset $X$.
* **Selection:** $C_1 = 0$

---

## Step 2: Choose Second Center ($C_2$)
Calculate the shortest distance $D(x)$ from each point to the nearest existing center ($C_1 = 0$), square it ($D(x)^2$), and convert to a selection probability:

$$P(x) = \frac{D(x)^2}{\sum D(x)^2}$$

| Point ($x$) | Nearest Center | Distance $D(x)$ | $D(x)^2$ | Probability |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0 | 0 | 0 | 0.0% |
| **1** | 0 | 1 | 1 | ~0.1% |
| **10** | 0 | 10 | 100 | ~9.4% |
| **11** | 0 | 11 | 121 | ~11.4% |
| **20** | 0 | 20 | 400 | ~37.6% |
| **21** | 0 | 21 | 441 | ~41.5% |
| **Total** | | | **1063** | **100%** |

* **Selection:** $C_2 = 21$ (Points furthest from $0$ have the highest probability)

---

## Step 3: Choose Third Center ($C_3$)
Recalculate distances to the closest center among $C_1 = 0$ and $C_2 = 21$:

| Point ($x$) | Nearest Center | Distance $D(x)$ | $D(x)^2$ | Probability |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0 | 0 | 0 | 0.0% |
| **1** | 0 | 1 | 1 | 0.5% |
| **10** | 0 | 10 | 100 | 49.5% |
| **11** | 21 | 10 | 100 | 49.5% |
| **20** | 21 | 1 | 1 | 0.5% |
| **21** | 21 | 0 | 0 | 0.0% |
| **Total** | | | **202** | **100%** |

* **Selection:** $C_3 = 10$ (Points $10$ and $11$ hold a combined 99% probability)

---

## Final Result
* **Initial Centers:** $\{0, 10, 21\}$
* **Impact:** Centers are optimally spread across all 3 natural data density groups, guaranteeing convergence to the global minimum in 1 standard K-Means iteration.

**Why probability ∝ D(x)² and not "just take the farthest point"?**
Because the farthest point is very often an **outlier**. A deterministic farthest-point rule would make outliers centroids every single time. Probabilistic sampling makes far points *likely* but not *certain*, so outliers rarely dominate — while still spreading the seeds across different regions.

> Note: K-Means++ **reduces** but does not eliminate outlier sensitivity — still combine it with `n_init > 1`. K-Means++ also comes with a theoretical guarantee: expected SSE ≤ 8(ln k + 2) × optimal SSE.

**Intuition walkthrough (3 clusters):** pick `c1` in group 1 at random. Points far from `c1` (groups 2 and 3) get large `D(x)²` → `c2` most likely lands in group 2 or 3, say group 2. Now recompute `D(x)` = distance to the *nearest* of {c1, c2}: group-1 and group-2 points have small `D`, group-3 points still have large `D` → `c3` very likely lands in group 3. Seeds end up spread over all three regions.

### 5.6 How to choose k

| Method | How | Watch out |
|---|---|---|
| **Domain knowledge** | You *know* there are 2 sentiment groups, 4 pricing tiers, 5 store formats | The best answer when available — always mention it first |
| **Elbow method** | Run K-Means for k = 1..K, plot SSE vs k, pick the **inflection point** where the rate of decrease flattens | SSE always decreases — the elbow is a *knee*, **not the minimum**. Often ambiguous |
| **Silhouette analysis** | Plot mean silhouette vs k, pick the **maximum** | More reliable than the elbow; O(n²) |
| **Gap statistic** | Compare SSE against SSE under a uniform null reference distribution | Principled, computationally heavy |
| **Davies-Bouldin / Calinski-Harabasz** | Minimize DB / maximize CH | Fast alternatives to plot alongside |
| **BIC / AIC** | Use GMM instead of K-Means and pick k by information criterion | Model-based, principled |
| **Downstream metric** | Choose the k that maximizes business/model lift | Strongest practical justification |

**Elbow method steps:** ① run K-Means for a range of k ② compute SSE (`inertia_`) for each ③ plot SSE vs k ④ find the elbow/inflection point ⑤ that k is your choice.

### 5.7 Variants of K-Means

| Variant | What changes | When to use |
|---|---|---|
| **Mini-Batch K-Means** | Updates centroids using random **mini-batches** instead of the full dataset each iteration | Very large datasets; much faster, slightly worse SSE |
| **K-Means++** | Smart seeding only | Default in sklearn (`init='k-means++'`) |
| **K-Medoids / PAM** | Centroid must be an **actual data point**; minimizes distances (not squared) | Interpretability, outlier robustness, arbitrary distance/kernel matrices |
| **K-Medians** | Uses median + L1 distance | Outlier-heavy data |
| **K-Modes / K-Prototypes** | Modes for categorical / mixed data (Huang's dissimilarity) | Categorical or mixed-type data |
| **Bisecting K-Means** | Repeatedly split the worst cluster with 2-means | Better than plain K-Means on some data; produces a hierarchy |
| **Fuzzy C-Means** | Soft memberships in [0,1] | Overlapping clusters |
| **Kernel K-Means** | K-Means in an implicit feature space | Non-linearly separable clusters |
| **Spherical K-Means** | L2-normalize vectors → cosine geometry | Text / embeddings |
| **Elkan / Hamerly** | Triangle-inequality speedups, same result | Speed on low-dim data (`algorithm='elkan'`) |

**Mini-batch K-Means steps:** ① initialize centroids & mini-batch size ② draw a mini-batch ③ assign each point in the batch to its nearest centroid ④ update centroids as a **running weighted mean** (weighted by how many points that centroid has seen so far) ⑤ repeat until convergence.

### 5.8 Advantages

- ✔ **Simple, fast, scalable** — near-linear: `O(n · k · d · i)`.
- ✔ Easy to explain to stakeholders; centroids are cluster "profiles".
- ✔ Works very well when clusters are **globular, similar in size and density, and well separated**.
- ✔ Guaranteed to converge; can be warm-started; can `predict()` on new points (out-of-sample assignment) — hierarchical clustering and DBSCAN cannot do this natively.
- ✔ Mini-batch version handles millions of rows.

### 5.9 Limitations (know all six)

1. **You must specify k in advance.**
2. **Initialization sensitivity** → local optima; different runs → different clusters (non-deterministic unless you fix the seed).
3. **Assumes spherical/globular clusters of similar size and density** — it fails on:
   - **differing sizes** (large cluster gets split, small one absorbed),
   - **differing densities** (sparse cluster carved up),
   - **non-globular / non-convex shapes** (crescents, rings, concentric circles).
   > *Non-convex intuition:* a shape is convex if you can join any two of its points with a straight line that never leaves the shape. In a crescent/bean shape you can't — and K-Means, which draws straight (Voronoi) boundaries, cannot represent it.
4. **Not robust to outliers** — the mean is dragged by extreme values; an outlier can even claim its own cluster.
5. **Every point must be assigned** — there is no "noise" label.
6. **Curse of dimensionality** — Euclidean distances lose contrast in high-d; also requires **numeric, scaled** features (categoricals need encoding, and one-hot + Euclidean is often a poor fit).

**Partial workaround for shapes:** over-cluster (large k) and then merge sub-clusters — but merging is itself hard and manual. Better: use DBSCAN, spectral, or kernel K-Means.

### 5.10 Complexity

| | Cost |
|---|---|
| **Time** | `O(n · k · d · i)` — n points, k clusters, d dims, i iterations. With `k ≈ 10` and `i < 300` this is effectively **O(n·d)** → linear |
| **Space** | `O(n·d + k·d) = O(n·d)` → linear |

### 5.11 Code (scikit-learn)

| Method | What it does | What it returns | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`.fit(X)`** | Calculates cluster centers based on data `X`. | **`None`** (updates model state in-place). | When you want to train the model now, but use it to cluster different data later. |
| **`.fit_predict(X)`** | Calculates cluster centers and assigns labels to `X`. | **1D Array** of integer cluster assignments. | When you just need the final cluster categories for your training dataset. |
| **`.fit_transform(X)`** | Calculates cluster centers and calculates distances. | **2D Array** of distances to each centroid. | When using KMeans as a feature extraction step for another ML model. |


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

X = StandardScaler().fit_transform(X)   # ALWAYS scale before K-Means

km = KMeans(
    n_clusters=2,        # k
    init='k-means++',    # 'random' | 'k-means++' | ndarray of seeds
    n_init='auto',       # restarts. 'auto' -> 1 for k-means++, 10 for random
                         # (default changed from 10 to 'auto' in sklearn 1.4)
    max_iter=300,
    tol=1e-4,            # stop when centroid shift < tol
    algorithm='lloyd',   # 'lloyd' | 'elkan'
    random_state=42      # REQUIRED for reproducibility
).fit(X)

km.labels_           # cluster label of each point,  shape (n_samples,)
km.cluster_centers_  # centroid coordinates,         shape (n_clusters, n_features)
km.inertia_          # SSE — sum of squared distances to closest centroid
km.n_iter_           # iterations actually run
km.predict([[0, 0]]) # assign a NEW point to a cluster
```

**Key attributes:**
- `cluster_centers_` — `(n_clusters, n_features)`. If the run stops before full convergence (`tol`/`max_iter`), these may be slightly inconsistent with `labels_`.
- `labels_` — `(n_samples,)`, label per point.
- `inertia_` — SSE, used for the elbow plot.
- `n_iter_` — number of iterations run.

**Elbow + silhouette in one loop:**

```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, round(km.inertia_, 2), round(silhouette_score(X, km.labels_), 3))
```

**Generating toy data:**

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=[3, 3, 4],  # int -> split equally; list -> per-cluster sizes
                  centers=None, n_features=2, cluster_std=1.0, random_state=0)
```

---

## 6. K-Medoids / PAM

### 6.1 Intro — why it exists

Two problems with K-Means centroids:

1. **Interpretability** — a K-Means centroid is an *averaged vector* that may not correspond to any real object. "The average review vector is [0.13, −0.42, …]" means nothing to a business user. If instead you can say *"review #4102 is the representative of this cluster — go read it"*, that's actionable.
2. **Outlier sensitivity** — the mean is dragged by extremes; a **medoid** (an actual point) is not.

> **K-Medoids:** same idea as K-Means, but each cluster center (**medoid**) must be an **actual data point `xⱼ ∈ D`**, and the objective uses distances rather than squared distances.

### 6.2 Objective

```
min  Σ_{i=1..k}  Σ_{x ∈ Sᵢ}  || x - mᵢ ||        where mᵢ ∈ D  (medoid of cluster i)
```

Using the plain distance (not squared) is another reason it's more outlier-robust.

### 6.3 PAM (Partitioning Around Medoids) — step by step

```
Step 1  INITIALIZE: pick k medoids (random, or a k-means++-style probabilistic pick).
Step 2  ASSIGN: assign every point to its closest medoid  (same as K-Means).
Step 3  SWAP (this is the difference):
          for each medoid m and each non-medoid point o:
              tentatively swap m ↔ o and recompute the total loss
              if the loss DECREASES  -> keep the swap
              else                   -> revert the swap
Step 4  If any swap succeeded, re-assign all points to the new medoids and repeat.
        Stop when no swap improves the loss.
```

**K-Means update vs K-Medoids update:**

| | K-Means | K-Medoids (PAM) |
|---|---|---|
| Update rule | `cⱼ = mean of points in Sⱼ` | Try swapping medoid ↔ non-medoid; keep swap only if loss drops |
| Center | Virtual point (mean vector) | **Real data point** |
| Objective | Sum of **squared** distances | Sum of distances |

**Worked micro-example.** `D = {x1 … x10}`, `k = 2`, current medoids `M1 = x1`, `M2 = x5`, loss `= L1`.
Try swapping `M1 = x1` with the non-medoid `x2` → medoids become `{x2, x5}`, loss `= L2`.
If `L2 < L1`, keep it and re-assign every point to the closer of `{x2, x5}`. Otherwise revert and try the next candidate.

### 6.4 The killer advantage — kernelizability

K-Medoids only ever needs **distances between pairs of existing points**. It never computes a mean, so it never needs the raw coordinates.

➡️ **You can run K-Medoids directly on a precomputed distance / similarity / kernel matrix.** That means you can cluster:
- graphs, strings (edit distance), DNA sequences,
- mixed-type data via **Gower distance**,
- anything where a distance is definable but an "average" is not.

K-Means cannot do this — computing a mean requires a vector space.

### 6.5 Advantages / Limitations

**Advantages**
- ✔ Centers are real, interpretable, presentable data points.
- ✔ More robust to outliers and noise than K-Means.
- ✔ Works with **any** distance metric / precomputed distance matrix (kernelizable).

**Limitations**
- ✘ **Much slower** — classic PAM is roughly `O(k(n−k)² · i)`; the swap step is the bottleneck. (FasterPAM / CLARA / CLARANS mitigate this by sampling.)
- ✘ Needs an `O(n²)` distance matrix → memory-heavy for large n.
- ✘ Still needs `k`; still converges to a local optimum; still struggles with non-globular shapes.

### 6.6 Code

Not in core scikit-learn — use `scikit-learn-extra` or the `kmedoids` package:

```python
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

data = np.array([[1, 1], [2, 2], [10, 10]])

# Option A: fit on raw features
km = KMedoids(n_clusters=2, metric='euclidean', init='k-medoids++',
              random_state=42).fit(data)
km.medoid_indices_   # INDICES of the real points chosen as medoids
km.labels_

# Option B: fit on a precomputed distance/kernel matrix  <-- the kernelizable path
D = pairwise_distances(data, metric='euclidean')
km2 = KMedoids(n_clusters=2, metric='precomputed').fit(D)
```

---

## 7. Hierarchical Clustering

### 7.1 Intro

> Hierarchical clustering builds a **hierarchy (tree) of nested clusters** by recursively **merging** (agglomerative) or **splitting** (divisive) groups based on similarity. Running it once gives you the clustering for **every possible k** at the same time.

- Output is a **dendrogram** — a tree that records *which* clusters were merged and *at what distance*.
- You choose k afterwards by **cutting the dendrogram horizontally** at a chosen height; the number of vertical lines the cut crosses = k.
- No random initialization → **deterministic** (given the same data, metric and linkage).

### 7.2 Two types

| | **Agglomerative (bottom-up)** — popular | **Divisive (top-down)** — rare |
|---|---|---|
| Start | Every point is its **own** cluster → **k = n** | All points in **one** cluster → **k = 1** |
| Move | Repeatedly **merge** the two closest clusters | Repeatedly **split** a cluster into two or more |
| End | One cluster containing everything | Every point is its own cluster |
| Cost | ~`O(n³)` naive, `O(n² log n)` with heaps | Even worse (choosing the best split is exponential); usually done heuristically with 2-means (Bisecting K-Means) |

⚠️ Slide decks often say "agglomerative starts with k = n−1" and "divisive starts with k = 2" — that's the state *after the first step*, not the start. Agglomerative **starts at k = n**; divisive **starts at k = 1**.

### 7.3 Agglomerative algorithm — step by step

```
Step 1  INITIALIZATION   Every data point is a singleton cluster.
Step 2  COMPUTE the pairwise proximity (distance/similarity) matrix
        between all clusters.                    <-- the main computation
Step 3  REPEAT:
Step 4     FIND & MERGE the two closest clusters (highest similarity /
           smallest distance) into one.
Step 5     UPDATE the proximity matrix: recompute distances between the
           NEW cluster and every remaining cluster, using the LINKAGE rule.
Step 6  UNTIL only a single cluster remains.
```

The **key operation is step 5** — how you define the distance between two *clusters* (not two points) is what distinguishes the algorithms. Implementation tip: cluster membership is naturally handled with **set union** operations, and the update only needs the previous matrix (no full recomputation) — the Lance-Williams update formula.

### 7.4 Linkage methods (inter-cluster similarity) — high-yield
see below ex: Single linkage (MIN): When A,B is combined into one cluster next we take AUB as one row , one column and when we put distance from this union to other points we take either min list of each of A, B to other points (C,D,E) for single link or max in complete and update the matrix and cross out the individual A,B rows and columns.

| Linkage | Definition | Behaviour |
|---|---|---|
| **Single (MIN)** | `d(Ci,Cj) = min` distance over all pairs `pi ∈ Ci, pj ∈ Cj` | Can capture **non-elliptical / elongated** shapes. **Susceptible to noise** — suffers from *chaining*: it "gobbles up" nearest points and grows a long straggly cluster |
| **Complete (MAX)** | `d(Ci,Cj) = max` distance over all pairs | Less susceptible to noise/outliers. **Biased toward globular clusters** and **tends to break up large clusters** |
| **Average (Group average)** | Mean of all pairwise distances: `Σ sim(pi,pj) / (|Ci| · |Cj|)` | Compromise between single and complete; less noise-sensitive; mild globular bias |
| **Ward's** | Merge the pair that produces the **smallest increase in total within-cluster SSE** | Least noise-sensitive; globular bias; **hierarchical analogue of K-Means**; can be used to initialize K-Means. **Requires Euclidean distance** |
| **Centroid** | Distance between the two cluster centroids | Not popular; can produce **inversions** (a merge at a lower height than a previous one), which makes the dendrogram non-monotonic |

⚠️ Naming trap that appears in almost every set of notes: **MIN = single linkage**, **MAX = complete linkage**. Average linkage = group average. Ward = "squared-error" linkage.
⚠️ Ward is **not** just "group average with squared distances" — it minimizes the *increase in SSE* when merging, which is a different criterion (it also weights by cluster sizes).

### 7.5 Worked example — single vs complete linkage

**Data (1-D):** `A=1, B=2, C=5, D=9, E=10`

Distance matrix:

|   | A | B | C | D | E |
|---|---|---|---|---|---|
| **A** | 0 | 1 | 4 | 8 | 9 |
| **B** | 1 | 0 | 3 | 7 | 8 |
| **C** | 4 | 3 | 0 | 4 | 5 |
| **D** | 8 | 7 | 4 | 0 | 1 |
| **E** | 9 | 8 | 5 | 1 | 0 |

**Single linkage (MIN):**
When A,B is combined into one cluster next we take AUB as one row , one column and when we put distance from this union to other points we take either min list of each of A, B to other points (C,D,E) for single link or max in complete and update the matrix and cross out the individual A,B rows and columns.
| Step | Merge | Height | Updated distances |
|---|---|---|---|
| 1 | {A,B} | 1 | AB–C = min(4,3) = 3; AB–D = 7; AB–E = 8 |
| 2 | {D,E} | 1 | DE–AB = min(7,8) = 7; DE–C = min(4,5) = 4 |
| 3 | {A,B,C} | 3 | ABC–DE = min(7,4) = **4** |
| 4 | {A,B,C,D,E} | 4 | — |

**Complete linkage (MAX):**

| Step | Merge | Height | Updated distances |
|---|---|---|---|
| 1 | {A,B} | 1 | AB–C = max(4,3) = 4; AB–D = 8; AB–E = 9 |
| 2 | {D,E} | 1 | DE–AB = max(8,9) = 9; DE–C = max(4,5) = 5 |
| 3 | {A,B,C} | 4 | ABC–DE = max(9,5) = **9** |
| 4 | {A,B,C,D,E} | 9 | — |

**Takeaway:** the *merge order* is the same here, but the **heights differ** (final merge at 4 vs 9). Cutting the dendrogram at height 3.5 gives k = 2 → `{A,B,C}` and `{D,E}` under both. Complete linkage's exaggerated heights make the last merge look far more "expensive", which is why it resists chaining.

### 7.6 Reading a dendrogram

- **x-axis:** data points; **y-axis:** the distance at which two clusters merged.
- **Tall vertical lines** = merges that cost a lot = well-separated groups.
- **Choosing k:** cut at the height of the **longest vertical gap not crossed by any horizontal merge line**; count the crossings.
- In `scipy`, `fcluster(Z, t, criterion='distance' or 'maxclust')` performs the cut.

### 7.7 Advantages

- ✔ **No need to fix k up front** — one run gives all k.
- ✔ **Dendrogram** = excellent visual explanation of structure (great for stakeholders).
- ✔ **Deterministic** — no random init.
- ✔ Flexible — works with any distance metric (except Ward) and can consume a **precomputed distance matrix**, so it handles non-vector data.
- ✔ Single linkage can find non-elliptical shapes.
- ✔ Naturally produces nested/taxonomic structure (phylogenetic trees, org taxonomies).

### 7.8 Limitations

1. **Computationally intensive** — time `O(n³)` (or `O(n² log n)`), space `O(n²)` for the proximity matrix → impractical beyond ~10k–50k points. Use **BIRCH**, or sample, or cluster K-Means centroids hierarchically.
2. **Greedy and irreversible** — a merge/split, once made, is never undone, so an early mistake propagates.
3. **No global objective function** to optimize (unlike K-Means' SSE) — makes it harder to justify mathematically.
4. **Linkage-dependent results:** single → noise/chaining; complete → breaks large clusters; neither handles clusters of very different sizes well.
5. **Sensitive to noise and outliers** (especially single linkage).
6. **Dendrogram interpretability collapses** for large n (thousands of leaves = unreadable).
7. No natural `predict()` for new points.

### 7.9 Applications

- Phylogenetic trees / evolutionary biology.
- Clustering US senators by Twitter behaviour (political grouping).
- Document taxonomies, product category trees.
- Gene expression analysis, customer hierarchies.

### 7.10 Code

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

model = AgglomerativeClustering(
    n_clusters=3,          # OR set n_clusters=None and use distance_threshold
    metric='euclidean',    # NOTE: parameter was called `affinity` before sklearn 1.2,
                           #       renamed to `metric` (1.2) and removed in 1.4
    linkage='ward'         # 'ward' | 'complete' | 'average' | 'single'
)
labels = model.fit_predict(X)

# Cut by distance instead of by k:
model2 = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0,
                                 linkage='average').fit(X)

# Dendrogram (scipy)
Z = sch.linkage(X, method='ward')     # 'single' | 'complete' | 'average' | 'ward'
sch.dendrogram(Z)
plt.show()
labels = sch.fcluster(Z, t=3, criterion='maxclust')
```

⚠️ `linkage='ward'` **only** supports `metric='euclidean'`. For cosine/precomputed, use `average` or `complete`.

---

## 8. DBSCAN (and HDBSCAN / OPTICS)

### 8.1 Intro

> **DBSCAN = Density-Based Spatial Clustering of Applications with Noise.** It groups points that lie in **dense regions** and labels points in **sparse regions as noise/outliers**. It finds **arbitrarily shaped** clusters and **does not require you to specify k**.

Core idea: **dense regions = clusters, sparse regions = noise.**

### 8.2 The two hyperparameters

| Parameter | Meaning |
|---|---|
| **`eps` (ε)** | Radius of the neighbourhood. Two points are neighbours if `dist(p, q) ≤ eps` |
| **`minPts` / `min_samples`** | Minimum number of points (**including the point itself** in sklearn) required inside the ε-radius for a region to count as dense |

**Density at point p** = number of points inside the hypersphere of radius ε centred at p. E.g. if 3 other points fall within radius 1.0 of p, the density at p is 4 (p + 3).
**Dense region** = a hypersphere of radius ε containing at least `minPts` points.

### 8.3 Point types

| Type | Definition |
|---|---|
| **Core point** | Has **≥ minPts** points within its ε-radius. Always lies in a dense region |
| **Border point** | Has **< minPts** in its ε-radius, **but** lies within ε of some **core point** Q (i.e. `dist(P, Q) ≤ eps`) |
| **Noise / outlier** | Neither core nor border — not reachable from any core point |

*Example:* with `eps = 1`, `minPts = 4` — point `p1` has 5 points in its circle → **core**. Point `p2` has only 3 → not core; but `dist(p2, p1) ≤ eps` and `p1` is core → `p2` is a **border** point. A point sitting alone in empty space → **noise**.

### 8.4 Density edge & density-connected points

- **Density edge:** if `P` and `Q` are **both core points** and `dist(P, Q) ≤ eps`, draw an edge between them.
- **Density connected:** `P` and `Q` are density-connected if there is a **path of density edges** `P → p1 → p2 → p3 → Q` linking them (all intermediate points being core points).

> Intuition: if you can **hop from one core point to another**, always within ε, and reach Q from P, then P and Q belong to the same cluster.
> **A cluster = a maximal set of density-connected core points + all border points attached to them.**

*(Formal vocabulary if the interviewer pushes: `q` is **directly density-reachable** from core `p` if `dist(p,q) ≤ eps`; **density-reachable** via a chain; **density-connected** if both are density-reachable from some common core point. Density-reachability is not symmetric — border points break symmetry — but density-connectivity is.)*

### 8.5 Algorithm — step by step

```
Step 1  For every point in D, use RangeQuery(xi, D, eps) to count neighbours and
        LABEL it as CORE / BORDER / NOISE.
        (RangeQuery is implemented with a KD-Tree / Ball-Tree for speed.)
Step 2  REMOVE the noise points — they belong to no cluster (sparse regions).
Step 3  Pick any CORE point P not yet assigned to a cluster:
          i)  create a new cluster containing P
          ii) add every point DENSITY-CONNECTED to P (found via repeated range
              queries) into that cluster
        Repeat with the next unassigned core point until all core points are used.
Step 4  Assign each BORDER point to the cluster of a nearby core point.
```

Equivalent "growth" phrasing often used on slides: start from an arbitrary point, retrieve its ε-neighbours; if `|neighbours| ≥ minPts`, start a new cluster and recursively expand through each neighbour that is itself a core point; otherwise mark the point as noise (it can be re-labelled as a border point later); repeat until every point has been visited.

### 8.6 Worked example (1-D)

`X = {1, 2, 3, 4, 10, 20, 21, 22}`, `eps = 1.5`, `minPts = 3` (counting the point itself)

| Point | ε-neighbours | Count | Type |
|---|---|---|---|
| 1 | {1, 2} | 2 | not core → **border** (within ε of core 2) |
| 2 | {1, 2, 3} | 3 | **CORE** |
| 3 | {2, 3, 4} | 3 | **CORE** |
| 4 | {3, 4} | 2 | not core → **border** (within ε of core 3) |
| 10 | {10} | 1 | **NOISE** |
| 20 | {20, 21} | 2 | not core → **border** |
| 21 | {20, 21, 22} | 3 | **CORE** |
| 22 | {21, 22} | 2 | not core → **border** |

Core points 2 and 3 are within ε of each other → density edge → same cluster.
**Result:** Cluster 0 = {1,2,3,4}; Cluster 1 = {20,21,22}; Noise = {10} → label **−1**.
Two clusters were discovered **without ever specifying k**, and the outlier was flagged automatically.

### 8.7 Choosing eps and minPts

**minPts:**
1. Rule of thumb: `minPts ≥ d + 1`, commonly `minPts ≈ 2 × d` (d = number of dimensions).
2. **Noisier data → larger minPts** (helps suppress noise).
3. Domain knowledge — the smallest group size that is *meaningfully* a cluster.
4. Start small and increase until the number of clusters stabilizes.

**eps (k-distance / elbow method):**
1. For every point `i`, compute `dᵢ` = distance to its **k-th nearest neighbour**, where `k = minPts`.
2. **Sort all `dᵢ` in increasing order** and plot them against the point index (the **k-distance graph**).
3. The **sharp elbow/knee** — where distances suddenly shoot up — is a good estimate for `eps`.
4. Points to the right of the knee (high `dᵢ`) are the noisy/isolated ones.

Plus: visual exploration (2-D projection), trial-and-error, and comparing **silhouette scores** across parameter grids.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np, matplotlib.pyplot as plt

k = 4  # = minPts
d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
plt.plot(np.sort(d[:, -1]))   # k-distance graph -> read eps off the knee
plt.ylabel(f'{k}-NN distance'); plt.xlabel('points sorted'); plt.show()
```

### 8.8 Advantages

- ✔ **No need to specify the number of clusters.**
- ✔ Finds **arbitrarily shaped** (non-convex, elongated, nested) clusters.
- ✔ **Robust to outliers** — it explicitly models noise instead of forcing every point into a cluster.
- ✔ Only 2 hyperparameters, both with physical meaning.
- ✔ Order of magnitude friendly to spatial databases — the whole algorithm reduces to **range queries**, which Oracle / PostGIS / SQL Server execute natively with spatial indexes.

### 8.9 Limitations

1. **Fails on clusters of varying density** — a single global `eps` cannot be right for both a dense and a sparse cluster. (This is DBSCAN's #1 weakness → answer: **HDBSCAN**.)
2. **Very sensitive to `eps` and `minPts`** — small changes flip the labelling completely (see the code demo below).
3. **Struggles in high dimensions** (curse of dimensionality makes ε meaningless) → poor for raw text/TF-IDF.
4. **Not fully deterministic** — **core points and the cluster structure are deterministic**, but a **border point reachable from two different clusters is assigned to whichever cluster is processed first**, so it depends on data ordering.
5. No `predict()` for new points out of the box (fit is transductive; `HDBSCAN` offers `approximate_predict`).
6. Can be memory-heavy if the implementation materializes neighbourhoods.

### 8.10 Complexity

| | Cost |
|---|---|
| **Time** | `O(n log n)` **average**, with a spatial index (KD-Tree/Ball-Tree) — each range query is `O(log n)` and we do `n` of them. **`O(n²)` worst case** (high dimensions, bad eps, no usable index) |
| **Space** | `O(n)` (plus the index). `O(n²)` if the full distance matrix is precomputed |

### 8.11 Code and a parameter-sensitivity demo

```python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

db = DBSCAN(
    eps=3,               # default 0.5 — the most important parameter
    min_samples=2,       # default 5
    metric='euclidean',  # or 'precomputed' to pass a distance matrix
    algorithm='auto',    # 'auto' | 'ball_tree' | 'kd_tree' | 'brute'
    leaf_size=30
).fit(X)

print(db.labels_)        # -> [ 0  0  0  1  1 -1 ]
db.core_sample_indices_  # indices of the core points
```

**`labels_` : `-1` means NOISE.** Above, `[25, 80]` is the noise point.

Same data, different `eps` — this is the sensitivity story in three lines:

| Parameters | Labels | Reading |
|---|---|---|
| `eps=3, min_samples=2` | `[0, 0, 0, 1, 1, -1]` | 2 clusters + 1 outlier ✅ |
| `eps=0.5, min_samples=2` | `[-1,-1,-1,-1,-1,-1]` | ε too small → **everything is noise** |
| `eps=25, min_samples=2` | `[0, 0, 0, 0, 0, -1]` | ε too large → **everything is one cluster** |
| `eps=3, min_samples=3` | `[0, 0, 0, -1, -1, -1]` | `{8,7},{8,8}` is only 2 points but needs 3 → **demoted to noise** |
| `eps=3, min_samples=4` | all `-1` | nothing reaches the density threshold |

> Great talking point: **"the labels are a function of the parameters, not just of the data"** — always report eps/minPts alongside DBSCAN results, and validate with the k-distance plot.

### 8.12 DBSCAN's successors (say these to stand out)

| Algorithm | What it fixes |
|---|---|
| **HDBSCAN** | Runs DBSCAN over **all** eps values and extracts the most stable clusters → **handles varying density**, and you only tune `min_cluster_size`. Available as `sklearn.cluster.HDBSCAN` (added in scikit-learn 1.3) or the `hdbscan` package |
| **OPTICS** | Produces a **reachability plot** ordering rather than a flat clustering; extract clusters at multiple densities. `sklearn.cluster.OPTICS` |
| **Mean-Shift** | Density mode-seeking, no k needed, but `O(n²)` and needs a bandwidth |

### 8.13 Applications

- **Geospatial clustering** — GPS pings, crime/incident hotspots, delivery zones (this is DBSCAN's home turf).
- **Anomaly / fraud detection** — noise points are the anomalies (global/point outliers and collective outliers).
- Image segmentation, network traffic analysis, sensor data.

---

## 9. Gaussian Mixture Models (soft clustering)

Frequently the follow-up question: *"K-Means gives hard assignments. What if a point could belong to two clusters?"*

> **GMM** models the data as a mixture of `k` **Gaussian distributions**: `P(x) = Σ πₖ · N(x | μₖ, Σₖ)`. It is fitted with the **Expectation-Maximization (EM)** algorithm and returns a **probability of belonging to each cluster** (soft assignment).

**EM loop:**
- **E-step:** given current parameters, compute each point's *responsibility* (posterior probability) for each component.
- **M-step:** given responsibilities, update `πₖ, μₖ, Σₖ` by weighted maximum likelihood.
- Repeat until the log-likelihood converges.

**Relationship to K-Means (great answer):** K-Means is the limiting case of a GMM with (a) hard assignments instead of soft responsibilities and (b) spherical covariances of equal, vanishing variance. GMM with `covariance_type='full'` can model **elliptical, rotated, differently-sized** clusters — which is exactly K-Means' blind spot.

| ✔ Advantages | ✘ Limitations |
|---|---|
| Soft/probabilistic memberships | Still need k (though **BIC/AIC** give a principled way to choose it) |
| Handles elliptical & differently-sized clusters | Assumes Gaussian components |
| Generative — can sample new data, score density | Sensitive to init (usually seeded by K-Means); can degenerate (singular covariance) |
| BIC/AIC for model selection | More parameters → needs more data, slower |

```python
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(X)
gm.predict(X)         # hard labels
gm.predict_proba(X)   # soft membership probabilities
gm.bic(X), gm.aic(X)  # model selection over n_components
```

---

## 10. Preprocessing Checklist Before Clustering

Clustering is unsupervised — there is no loss signal to rescue you from bad inputs. Preprocessing *is* the modelling.

| Step | Why | How |
|---|---|---|
| **1. Data cleaning** | Missing values break distance computations; outliers hijack centroids | Impute / drop; cap or winsorize; or choose an outlier-robust algorithm (DBSCAN, K-Medoids) |
| **2. Scaling — the non-negotiable one** | Distance is dominated by the largest-range feature. `income` in ₹ (0–10⁷) will completely drown `age` (18–80) | `StandardScaler` (default), `MinMaxScaler`, or `RobustScaler` when outliers are present |
| **3. Encoding categoricals** | Euclidean distance needs numbers | One-hot (careful: inflates dimensionality and distorts distance), ordinal where genuinely ordered, or use **K-Modes / K-Prototypes / Gower distance** instead |
| **4. Handling skew** | Long-tailed features dominate variance | `log1p`, square-root, Box-Cox / Yeo-Johnson |
| **5. Dimensionality reduction / feature selection** | Curse of dimensionality; correlated features get double-counted | PCA (also decorrelates), UMAP, or drop redundant features |
| **6. Deduplicate** | Duplicate rows act as artificial density | Drop or weight |

> ⚠️ **Fit the scaler on the data you cluster and reuse it** (`pipeline`), and remember: **scaling changes the answer**. If a stakeholder asks "why did the segments change?", scaling choice is usually the culprit.

> ⚠️ **PCA before clustering is a trade-off**: it helps distances behave, but it destroys interpretability of the cluster profiles. A common compromise: cluster in PCA space, then profile clusters back in the original feature space.

---

## 11. Algorithm Comparison Cheat Sheet

| | **K-Means** | **K-Medoids (PAM)** | **Hierarchical (Agglo.)** | **DBSCAN** | **GMM** |
|---|---|---|---|---|---|
| Need k? | ✅ Yes | ✅ Yes | ❌ No (cut later) | ❌ No | ✅ Yes (BIC helps) |
| Cluster shape | Spherical / convex | Spherical / convex | Depends on linkage | **Arbitrary** | Elliptical |
| Handles outliers | ❌ Poor | 🟡 Better | ❌ Poor (esp. single) | ✅ **Explicit noise label** | 🟡 Moderate |
| Varying density | ❌ | ❌ | 🟡 | ❌ (→ HDBSCAN) | 🟡 |
| Varying cluster size | ❌ | ❌ | 🟡 | ✅ | ✅ |
| Deterministic | ❌ (seed it) | ❌ | ✅ | 🟡 (border points only) | ❌ |
| Time complexity | `O(n·k·d·i)` ~linear | `O(k(n−k)²·i)` | `O(n³)` / `O(n² log n)` | `O(n log n)` avg | `O(n·k·d²·i)` |
| Space | `O(n·d)` | `O(n²)` | `O(n²)` | `O(n)` | `O(n·d)` |
| Scales to millions | ✅ (mini-batch) | ❌ | ❌ | 🟡 | 🟡 |
| Soft assignment | ❌ | ❌ | ❌ | ❌ | ✅ |
| Works on a distance matrix | ❌ | ✅ **Kernelizable** | ✅ | ✅ (`metric='precomputed'`) | ❌ |
| Predict new points | ✅ | ✅ | ❌ | ❌ | ✅ |
| Interpretable center | 🟡 Mean vector | ✅ **Real data point** | — (dendrogram) | — | 🟡 Mean + covariance |

### Which one do I pick? (decision flow)

```
Do I know k, and are clusters roughly round, similar size, big data?
        └─ YES ──> K-MEANS  (mini-batch if n is huge)
        └─ NO
             │
             ├─ Do clusters have weird shapes / is there real noise & outliers?
             │        └─ YES ──> DBSCAN  (HDBSCAN if densities vary)
             │
             ├─ Do I need a hierarchy / a dendrogram / small n (< ~10k)?
             │        └─ YES ──> AGGLOMERATIVE (Ward for globular, single for chains)
             │
             ├─ Do I only have a distance/similarity matrix, or need a real
             │  data point as the representative?
             │        └─ YES ──> K-MEDOIDS / PAM
             │
             └─ Do points plausibly belong to several groups / do I want
                probabilities and elliptical clusters?
                      └─ YES ──> GMM (choose k by BIC)
```

**Comparative summary table (short form):**

| Algorithm | ✔ Strengths | ✘ Weaknesses |
|---|---|---|
| K-means | Simple, efficient, works well for well-separated spherical data | Not good with noise or irregular shapes; must choose k |
| Hierarchical | Flexible, handles many shapes, gives all k at once, deterministic | More computationally expensive; hard to interpret at scale |
| DBSCAN | Finds arbitrary shapes, robust to outliers, no k needed | **Sensitive to eps/minPts**, struggles with varying density and high dimensions |

---

## 12. Corrections / Clarifications to Common Notes

These are the exact places where the source notes/slides slip, and where a sharp interviewer will catch you.

| # | Common statement | ✅ Correction |
|---|---|---|
| 1 | *"Dunn Index numerator: for every pair of points in Ci and Cj take the one with **maximum** distance"* | The numerator is the **MINIMUM** inter-cluster distance (closest pair of clusters). The **maximum** goes in the **denominator** (largest intra-cluster diameter). `Dunn = min-separation / max-diameter` |
| 2 | *"Dunn index is robust to outliers"* (slide table) | **False.** Dunn is one of the **least** robust internal metrics — it's built from two extremes (a min and a max), so a single outlier can wreck it |
| 3 | *"MIN = complete linkage"* (appears after "MIN = single linkage") | **MIN = single linkage; MAX = complete linkage.** The second statement is a typo for MAX |
| 4 | *"Ward's method = group average but with squared distances"* | Ward merges the pair that causes the **smallest increase in total within-cluster SSE**. Related to squared error, but it's a different criterion (and size-weighted), not "average linkage squared" |
| 5 | *"Agglomerative starts with k = n − 1 clusters; divisive starts with k = 2"* | Agglomerative **starts at k = n** (each point its own cluster); divisive **starts at k = 1** (all points together). n−1 and 2 are the states *after* the first step |
| 6 | *"K-Means++ picks the next centroid with probability proportional to the distance"* | Proportional to the **squared** distance `D(x)²` to the nearest already-chosen centroid |
| 7 | *"Elbow: K = 5 has the minimum loss"* | SSE **always decreases** as k increases (it hits 0 at k = n). The elbow is where the **rate of decrease** flattens — an inflection/knee, **not a minimum** |
| 8 | *"Silhouette b = average inter-cluster distance i.e. average distance between all clusters"* | `b(i)` = mean distance from point i to the points of the **single nearest other cluster**, not the average over all clusters |
| 9 | *"DBSCAN: strengths = robust to noise and outliers; weaknesses = sensitive to noise and outliers"* (slide contradicts itself) | DBSCAN **is** robust to noise/outliers. The real weakness is **sensitivity to the hyperparameters `eps` and `minPts`** (and to varying density) |
| 10 | *"DBSCAN is not deterministic"* | Mostly deterministic: **core points and cluster structure are fixed**. Only **border points reachable from two clusters** depend on processing order |
| 11 | *"DBSCAN time complexity is O(n log n)"* | That's the **average case with a spatial index (KD-Tree/Ball-Tree)**. Worst case is **O(n²)** (high dimensions, large eps, or brute-force) |
| 12 | *"Sparsity of A = # non-empty cells / # total cells"* (handwritten) | That is **density**. **Sparsity = # empty cells / # total cells = 1 − density.** (The typed note is right, the handwritten one is flipped) |
| 13 | *"Labeling 10M points via 10k clusters = 10k-fold reduction"* | Reduction factor = **N / k**. 10M → 10k clusters = **1000×**. State it as a ratio, not a fixed number |
| 14 | *"K-Medoid distance matrix advantage… it is kernelizable"* | Correct — but be precise: K-Medoids works on **any precomputed distance/similarity matrix** because it never needs to compute a mean. "Kernelizable" in the sense that a kernel/Gram matrix induces a valid distance |
| 15 | *"n_init = 10 is the default in sklearn"* | **Changed in scikit-learn 1.4**: the default is now `n_init='auto'`, which means **1** run for `init='k-means++'` and **10** for `init='random'`. Set `n_init=10` explicitly if you want the old safety net |
| 16 | *"AgglomerativeClustering(affinity=…)"* | `affinity` was **renamed to `metric`** in scikit-learn 1.2 and **removed** in 1.4 |
| 17 | *"K-Means minimizes intra-cluster distance"* | More precisely the **sum of squared** distances to the centroid. It is the squaring that makes the arithmetic mean the optimal center |
| 18 | *"MAX (complete linkage) is biased toward globular structure (large clusters)"* | Complete linkage is biased toward **globular clusters** and **tends to break large clusters apart** — the two ideas got merged in the note. It's **single (MIN)** linkage that grows large straggly clusters via chaining |
| 19 | *"Hierarchical: min has problem with outliers, max breaks large clusters, it cannot accommodate different sized clusters"* | Correct — worth adding the missing #1 limitation: **greedy merges are irreversible**, and complexity `O(n³)` time / `O(n²)` space is the practical blocker |
| 20 | *"if 3 points are available around a radius of 1.0 then density is 4"* | Right, **as long as you state that the point itself is counted**. sklearn's `min_samples` **includes** the point itself — many textbooks don't. Always clarify the convention |

---

## 13. Interview Questions & Answers

### A. Fundamentals

**Q1. What is unsupervised learning and how is it different from supervised learning?**
Unsupervised learning finds structure in data with **no target variable**; supervised learning learns a mapping `X → y` from labeled examples. Consequences: no ground truth, so **no accuracy/F1** — you evaluate with internal metrics (silhouette, SSE), stability, and business validation. It's used for exploration, segmentation, compression, anomaly detection, and to reduce labeling cost.

**Q2. What is clustering, and how do you know your clustering is "good"?**
Partitioning data so points in the same group are more similar to each other than to points in other groups. "Good" = **small intra-cluster distance + large inter-cluster distance**, measured by silhouette / Dunn / Davies-Bouldin (internal), ARI/NMI (if labels exist), plus **stability under resampling** and **business interpretability & actionability**.

**Q3. When would you use clustering in a real business problem?**
Customer segmentation for targeted offers, image/pixel segmentation, document/ticket grouping, fraud & anomaly detection, cutting annotation cost by labeling cluster representatives, generating `cluster_id` features for a supervised model, cold-start in recommenders.

**Q4. What is the difference between k-Means and k-NN?**

| | k-Means | k-NN |
|---|---|---|
| Type | **Unsupervised** (clustering) | **Supervised** (classification/regression) |
| Meaning of k | Number of **clusters** | Number of **neighbours** |
| Needs labels? | No | Yes |
| Output | Centroids + cluster assignments | Predicted label/value |
| Training | Iterative fitting | **Lazy** — no training, all work at prediction time |
| Determinism | **No** (random init — fix `random_state`) | **Yes**, given fixed data and k (ties may be broken randomly) |

**Q5. Will you get the same result every time you run k-Means? What about k-NN?**
k-Means: **no** — it randomly initializes centroids, so results vary run to run; set a fixed seed (`random_state`) and/or use `n_init > 1` for consistency. k-NN: **yes**, deterministic given the same data and k, unless there are ties in the neighbour voting.

**Q6. Why must you scale features before clustering?**
Distance-based algorithms let the largest-range feature dominate. Income (0–10⁷) vs age (18–80): the clustering becomes "clustering by income" regardless of age. Standardize (or min-max/robust scale) first. Tree-based supervised models don't care about scale — **distance-based unsupervised models absolutely do.**

**Q7. What is the curse of dimensionality and how does it affect clustering?**
As dimensions grow, pairwise distances concentrate — the ratio of nearest to farthest distance approaches 1, so "nearest neighbour" loses meaning. Every distance-based algorithm degrades. Mitigate with feature selection, PCA/UMAP, cosine distance on sparse text, or density algorithms designed for high-d (or avoid raw high-d entirely).

### B. K-Means

**Q8. Explain k-Means as if to a non-technical stakeholder.**
"Pick k meeting points. Send every customer to their nearest meeting point. Move each meeting point to the middle of the customers who came to it. Repeat until nobody has to move. The final groups are your segments."

**Q9. What exactly does k-Means optimize, and why is the mean the right center?**
It minimizes SSE = `Σᵢ Σ_{x∈Sᵢ} ||x − cᵢ||²`. For a fixed assignment, the point minimizing the sum of **squared** distances to a set is exactly its **arithmetic mean** — hence "k-Means". (If you minimized *absolute* distance, the optimum would be the median → k-Medians.)

**Q10. Is k-Means guaranteed to find the optimal clustering?**
**No.** The exact problem is **NP-hard**. Lloyd's algorithm is a heuristic that **always converges** (SSE is non-increasing and there are finitely many partitions) but only to a **local** optimum. Mitigate with `n_init` restarts and k-means++ seeding.

**Q11. Walk me through k-means++ and why it uses squared distances probabilistically.**
Pick the first centroid uniformly at random; for each remaining point compute `D(x)` = distance to the nearest chosen centroid; sample the next centroid with probability ∝ `D(x)²`; repeat. Squaring strongly favours far-away points, spreading the seeds. It is **probabilistic rather than deterministic-farthest** because the farthest point is usually an **outlier** — a deterministic rule would make outliers centroids every time. It gives an `O(log k)` expected approximation guarantee and typically converges in fewer iterations.

**Q12. How do you choose k?**
Domain knowledge first. Then: **elbow** on SSE (find the knee, not the minimum), **silhouette** (maximize), **gap statistic**, **Davies-Bouldin / Calinski-Harabasz**, **BIC/AIC** via GMM, and finally **downstream business/model performance**. In practice I'd plot elbow + silhouette together and cross-check that the resulting segments are distinguishable and actionable.

**Q13. Why can't you just pick the k with the lowest SSE?**
SSE decreases monotonically with k and reaches 0 when k = n (every point its own cluster). Minimizing it is degenerate — hence the elbow heuristic or a metric that penalizes complexity (silhouette, gap, BIC).

**Q14. What are k-Means' failure modes? Give me a picture for each.**
① Different **sizes** — the big cluster gets split and part of the small one absorbed. ② Different **densities** — the sparse cluster gets carved up. ③ **Non-globular shapes** — two crescents get sliced straight through. ④ **Outliers** — a far point drags a centroid or claims its own cluster. ⑤ **Wrong k**. ⑥ **Bad init** → poor local optimum.

**Q15. Your k-Means result changes every run. What do you do?**
Set `random_state`; increase `n_init` (explicitly, since sklearn ≥ 1.4 defaults to `'auto'` = 1 run with k-means++); use k-means++; check whether the instability is real (run stability analysis with ARI across bootstrap samples) — persistent instability usually means the data has **no strong cluster structure at that k**, or k is wrong, or scaling is off.

**Q16. Can k-Means handle categorical data?**
Not naturally — the mean of a one-hot vector isn't a valid category, and one-hot + Euclidean distorts distances. Use **K-Modes** (categorical), **K-Prototypes** (mixed), or **Gower distance + K-Medoids / hierarchical**.

**Q17. What is Mini-Batch k-Means and what do you trade away?**
It updates centroids using a random **mini-batch** each iteration rather than the whole dataset, with a running weighted mean. Dramatically faster and lower memory on millions of rows; the cost is a **slightly higher SSE** and noisier convergence.

**Q18. A cluster comes out empty. What happened, what do you do?**
An initialized centroid attracted no points (bad init, or k too large for the data's structure). Standard remedies: reinitialize that centroid at the point furthest from its centroid (or at a random point), or reduce k. sklearn handles this internally.

**Q19. How do you interpret and name the clusters you produced?**
Profile each cluster: mean/median of every feature vs the global mean, size, and a few representative members (nearest points to the centroid, or the medoid). Rank features by how much the cluster deviates from the population. Then give each segment a business label and validate that the segments differ on outcome metrics.

### C. K-Medoids

**Q20. What is k-Medoids and when would you prefer it over k-Means?**
Same iterative structure, but the center is an **actual data point (medoid)** and the objective uses distances (not squared). Prefer it when: (a) you need an **interpretable, real representative** ("show me the actual review that typifies this cluster"), (b) the data has **outliers**, (c) you only have a **distance/similarity matrix** (strings, graphs, mixed-type data via Gower) and can't compute a mean. Trade-off: much slower, `O(n²)` memory.

**Q21. How does the PAM update step differ from k-Means'?**
k-Means recomputes each centroid as the mean. PAM instead **tries swapping** each medoid with each non-medoid, keeps the swap **only if the total loss decreases**, otherwise reverts — then reassigns points. That exhaustive swap search is why PAM is expensive.

### D. Hierarchical

**Q22. Explain agglomerative clustering and the dendrogram.**
Start with each point as its own cluster; repeatedly merge the two closest clusters (per the linkage rule), updating the proximity matrix, until one cluster remains. The **dendrogram** records the merge order and the distance at each merge; cutting it horizontally at height h yields a flat clustering, and the number of branches crossed is k.

**Q23. Compare single, complete, average and Ward linkage.**
Single (MIN) = closest pair → handles elongated/non-elliptical shapes but **chains** and is noise-sensitive. Complete (MAX) = farthest pair → compact globular clusters, less noise-sensitive, but **breaks up large clusters**. Average = mean pairwise distance → a compromise. Ward = minimize the **increase in SSE** → globular, least noise-sensitive, the hierarchical analogue of k-Means (Euclidean only). Default choice in practice: **Ward**; switch to average/complete for non-Euclidean metrics; single only when you specifically expect chain-like structure.

**Q24. Why doesn't hierarchical clustering scale?**
`O(n²)` memory for the proximity matrix and `O(n³)` (or `O(n² log n)`) time. At n = 100k that's ~10¹⁰ pairwise entries. Workarounds: sample, use **BIRCH**, or run k-Means to get a few thousand micro-clusters and cluster *those* hierarchically.

**Q25. Advantages of hierarchical over k-Means?**
No need to pre-specify k (one run covers all k), deterministic, gives an interpretable dendrogram, works with arbitrary distance metrics and precomputed matrices, can capture nested/taxonomic structure and (with single linkage) non-elliptical shapes.

**Q26. What is divisive clustering and why is it rarely used?**
Top-down: start with everything in one cluster and recursively split. The number of ways to split a cluster is exponential, so exact divisive clustering is intractable; in practice it's approximated (e.g. Bisecting k-Means). Agglomerative is simpler and more common.

### E. DBSCAN

**Q27. Explain DBSCAN in one minute.**
Two parameters: `eps` (neighbourhood radius) and `minPts`. Any point with ≥ minPts neighbours within eps is a **core point**; a non-core point within eps of a core point is a **border point**; everything else is **noise**. Core points within eps of each other are chained together into clusters, then border points attach to their nearest cluster. Result: **arbitrarily shaped clusters, an explicit noise label, and no need to specify k**.

**Q28. Define core / border / noise and density-connected.**
Core: ≥ minPts points within eps (including itself, in sklearn). Border: fewer than minPts, but within eps of a core point. Noise: neither. Two core points within eps have a **density edge**; two points are **density-connected** if a chain of density edges links them. A cluster is a maximal set of density-connected core points plus their border points.

**Q29. How do you choose eps and minPts?**
`minPts`: start at `d + 1`, commonly `2 × d`; increase for noisy data; use domain knowledge about the smallest meaningful group. `eps`: build the **k-distance graph** — for `k = minPts`, compute each point's k-th nearest-neighbour distance, sort ascending, plot, and take eps at the **knee**. Cross-validate with silhouette across a small grid and sanity-check the number of clusters and the noise fraction.

**Q30. What is DBSCAN's biggest weakness and how do you fix it?**
**Varying density** — one global eps can't serve both a dense and a sparse cluster; either the sparse cluster becomes noise (eps too small) or the dense ones merge (eps too large). Fix: **HDBSCAN**, which sweeps over all eps values and keeps the most persistent clusters, needing only `min_cluster_size`. **OPTICS** is the other option.

**Q31. Is DBSCAN deterministic?**
Nearly. Core-point labelling and cluster structure are deterministic. Only **border points reachable from two clusters** are order-dependent — they go to whichever cluster is expanded first. Also note DBSCAN has no `predict()` for unseen points.

**Q32. K-Means vs DBSCAN — when do you choose which?**
K-Means when clusters are round-ish, of similar size and density, k is known or estimable, n is huge, and every point must be assigned. DBSCAN when shapes are arbitrary, noise/outliers are genuine and must be excluded, k is unknown, and n and d are moderate. If shapes vary *and* density varies → HDBSCAN. If you need probabilities → GMM.

**Q33. You ran DBSCAN and 80 % of your points came back as −1. What now?**
eps is too small or minPts too large for this data. Re-plot the k-distance graph and pick eps at the knee; lower minPts; check whether you **scaled** the features (unscaled data makes eps meaningless); check dimensionality — in high-d, reduce first. If a large noise fraction persists across sensible parameters, the data may genuinely lack dense structure.

### F. Evaluation & practice

**Q34. How do you evaluate clustering without labels?**
Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn), **stability** (re-cluster bootstrap samples and measure ARI between runs), **visual** inspection in PCA/UMAP space, **profiling** (are clusters distinguishable on features that matter?), and **downstream utility** (does `cluster_id` improve a model or a campaign?).

**Q35. Silhouette vs Dunn vs SSE — when do you use each?**
SSE only for the elbow within k-Means. Silhouette as the general-purpose default (per-point diagnostics, works with any metric) — sample it for large n. Dunn when you specifically care about worst-case separation, and only on clean data, since it's outlier-fragile. Davies-Bouldin / Calinski-Harabasz as fast companions when scanning many k.

**Q36. Your silhouette score is 0.15. Is that bad?**
It's weak — clusters overlap substantially. But context matters: high-dimensional and real-world behavioural data routinely score 0.1–0.3 even when the segments are commercially useful. I'd check whether it improves after scaling/PCA, compare across k and across algorithms, look at the **per-cluster** silhouette plot to see if one cluster is dragging the mean down, and then judge on stability and business lift rather than the number alone.

**Q37. How do you cluster mixed numeric + categorical data?**
**Gower distance** + K-Medoids or hierarchical (both accept precomputed matrices); or **K-Prototypes**; or embed categoricals (target/entity embeddings) and use K-Means. Avoid naive one-hot + Euclidean on high-cardinality categoricals.

**Q38. How would you cluster 50 million rows?**
Scale + reduce dimensions first (PCA/random projection). Use **Mini-Batch K-Means** or **BIRCH** for a first pass into micro-clusters, then cluster the micro-cluster centers with a more expensive algorithm (hierarchical/GMM) to get final segments. For density-based needs at scale, use a distributed HDBSCAN or run DBSCAN per geographic/partition shard. Evaluate silhouette on a random sample rather than the full data.

**Q39. How do you use clustering to reduce labeling cost, and what's the risk?**
Cluster the unlabeled pool, label a handful of representatives per cluster, propagate labels within the cluster → reduction factor `N / k`. **Risk:** impure clusters propagate wrong labels at scale. Mitigate by validating purity on a hand-labeled sample, using many small clusters rather than a few big ones, labeling points near the centroid (high confidence) and treating boundary/noise points separately, and comparing against **active learning** (label the most *uncertain* points), which is often the stronger method.

**Q40. How would you detect anomalies with clustering?**
DBSCAN noise points (label −1); distance from the nearest centroid in k-Means above a percentile threshold; low likelihood under a GMM; very small clusters as *collective* anomalies. Compare against Isolation Forest / One-Class SVM as baselines.

**Q41. Your customer segments look great statistically but marketing says they're useless. What do you do?**
Ask what actions they can actually take, and re-do the feature set to match: cluster on **behavioural, actionable** features (recency, frequency, monetary, category mix, channel) rather than everything available; drop features nobody can influence; reduce k to a number a campaign team can operationalize (3–6 usually); and validate segments against outcome metrics. The right features and the right granularity beat the right algorithm here.

**Q42. Data leakage / bias concerns in clustering?**
Clustering on proxies for protected attributes (postcode, name-derived features) can produce segments that encode discrimination and then flow into pricing/targeting decisions. Audit cluster composition against protected attributes, drop or test proxy features, and document what the segments actually mean before they drive decisions.

---

## 14. Rapid-Fire One-Liners

| Question | Answer |
|---|---|
| K-Means objective | Minimize SSE = sum of squared distances to the assigned centroid |
| Is K-Means NP-hard? | The exact problem yes; Lloyd's is a local-optimum heuristic |
| K-Means complexity | Time `O(n·k·d·i)`, space `O(n·d)` — both roughly linear |
| Why the *mean*? | The mean minimizes the sum of **squared** distances |
| Why k-means++ uses `D(x)²` | Spreads seeds; probabilistic sampling avoids locking onto outliers |
| Elbow method picks... | The **inflection/knee**, not the minimum SSE |
| Silhouette range | −1 to +1; ≈1 good, ≈0 overlapping, ≈−1 misassigned |
| Silhouette formula | `(b − a) / max(a, b)` |
| Dunn Index | min inter-cluster distance ÷ max intra-cluster diameter; higher is better |
| Davies-Bouldin | Lower is better |
| MIN linkage = | Single linkage — chaining, noise-sensitive, handles elongated shapes |
| MAX linkage = | Complete linkage — globular bias, breaks large clusters |
| Ward's linkage = | Minimizes increase in within-cluster SSE; Euclidean only |
| Agglomerative complexity | `O(n³)` time (or `O(n² log n)`), `O(n²)` space |
| DBSCAN parameters | `eps` (radius) and `minPts` (density threshold) |
| DBSCAN noise label | `-1` |
| DBSCAN complexity | `O(n log n)` average with a spatial index; `O(n²)` worst case |
| DBSCAN's fatal flaw | Varying density → use HDBSCAN |
| How to pick eps | k-distance graph, knee point, with `k = minPts` |
| Which algorithms need k? | K-Means, K-Medoids, GMM (not DBSCAN, not hierarchical) |
| Which give soft assignments? | GMM, Fuzzy C-Means |
| Which handle a precomputed distance matrix? | K-Medoids, hierarchical, DBSCAN (`metric='precomputed'`) |
| Which support `predict()` on new data? | K-Means, K-Medoids, GMM |
| Which are deterministic? | Hierarchical (fully); DBSCAN (except border points) |
| K-Means vs GMM | Hard vs soft; spherical vs elliptical; GMM = K-Means with covariances and responsibilities |
| Must you scale? | Yes for every distance-based clustering algorithm |
| sklearn `inertia_` is | SSE / within-cluster sum of squares |
| sklearn `n_init` default | `'auto'` since 1.4 → 1 run with k-means++, 10 with random init |

---

## Appendix A — Recommender Systems (adjacent unsupervised topic)

Often the follow-on topic after clustering, because it uses the same similarity machinery.

**The user–item matrix `A`:** rows = users, columns = items, cells = rating / watched flag. Almost all cells are empty.

```
sparsity = (# empty cells) / (# total cells)          density = 1 - sparsity
```

*(A frequent slip: `# non-empty / # total` is **density**, not sparsity.)*

**Recommendation as matrix completion:** some `Aᵢⱼ` are given, most are missing → fill in the empty cells with plausible values inferred from the non-empty ones, then recommend the highest predicted items.

**Two families:**

| | **Content-based filtering** | **Collaborative filtering (CF)** |
|---|---|---|
| Signal | Item/user **attributes** (genre, tags, text) | **Interaction patterns** only |
| Core assumption | "You'll like items similar to what you liked" | **"Users who agreed in the past tend to agree in the future"** |
| Cold start | Handles **new items** well | Struggles with new users **and** new items |
| Diversity | Tends to over-specialize (filter bubble) | Can surface serendipitous items |

**CF sub-types:** *memory-based* (user–user or item–item similarity, typically cosine or Pearson) and *model-based* (**matrix factorization** — SVD / ALS / NMF — learning latent user and item vectors so that `Aᵢⱼ ≈ uᵢ · vⱼ`).
**Clustering's role:** cluster users or items to shrink the similarity search space, to smooth sparse rows, and to mitigate cold start by assigning a new user to a segment and recommending that segment's favourites. Production systems are usually **hybrid** (content + CF + popularity fallback).

---

## Appendix B — Sample Data Generators & Sanity Checks

```python
from sklearn.datasets import make_blobs, make_moons, make_circles

X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)  # K-Means wins
X, y = make_moons(n_samples=300, noise=0.05)      # K-Means FAILS, DBSCAN wins
X, y = make_circles(n_samples=300, factor=0.5, noise=0.05)  # K-Means FAILS, DBSCAN/spectral win
```

> Interview move: if asked "when does k-Means fail?", say **"make_moons and make_circles"** — everyone recognizes the picture instantly.

**Minimum sanity checklist before you trust any clustering:**

- [ ] Features scaled (and the choice justified)
- [ ] `random_state` set for reproducibility
- [ ] k chosen with at least two independent methods
- [ ] Result visualized in 2-D (PCA/UMAP)
- [ ] Cluster sizes checked (one giant cluster + tiny ones = suspicious)
- [ ] Stability checked across bootstrap resamples (ARI)
- [ ] Clusters profiled and named in business language
- [ ] Compared against a second algorithm with a different inductive bias
