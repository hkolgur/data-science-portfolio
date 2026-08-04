# Boosting: AdaBoost, GBDT, XGBoost, LightGBM, CatBoost — Interview Notes

## 1. Boosting — Core Idea

Build an **additive model in stages**, where each new weak learner corrects the
errors of the ensemble so far:

```
F0(x) = simple initial guess
Fm(x) = Fm-1(x) + ν · hm(x)        (m = 1 … M)
Final: FM(x) = F0(x) + ν·Σ hm(x)
```

- Base learners are **weak: high bias, low variance** (shallow trees; leaves
  typically 8–32 for GBDT, stumps for AdaBoost)
- Boosting **reduces bias** stage by stage (bagging reduces variance)
- Training is inherently **sequential** — stage m needs Fm-1's errors

### The three ingredients every boosting algorithm needs
1. **A loss function** — differentiable (GBDT); defines what "error" means
2. **A weak learner** — usually a shallow tree
3. **An additive model** — trees are added one at a time and
   **previously added trees are never modified**

That third point is the one candidates skip, and it's what makes boosting greedy
and *stagewise* (not "stepwise"): each stage solves its own small problem and the
earlier stages are frozen.

### Weak learner vs strong learner (the concept behind boosting)

| | Definition | Examples |
|---|---|---|
| **Weak learner** | Only needs to beat a coin toss: `P[error] < 0.5` (more precisely, weighted error ≤ 0.5 − γ for some γ > 0). Also cheap at inference. | Decision **stump** (a tree with one split), shallow tree |
| **Strong learner** | In the **PAC** sense (Probably Approximately Correct): with enough data, error can be driven arbitrarily close to zero with high probability. | AdaBoost / GBDT over stumps |

**The historical question (Kearns & Valiant, late 80s):** *can a weak learner
always be boosted into a strong learner?* AdaBoost (Freund & Schapire, mid-90s)
answered **yes**, which is why boosting mattered — it was a theoretical result
first, a Kaggle tool second. Given weak learners that each beat chance by any
margin γ > 0, the boosted combination is a **universal function approximator**.

> Interview framing: bagging is a **variance-reduction** trick you'd invent from
> statistics. Boosting is an **existence proof** — "many barely-useful models can
> be combined into an arbitrarily good one" — and it happens to reduce bias.

---

## 2. GBDT by Hand — Regression Walkthrough
*(the StatQuest-style example: predicting weight)*

# Gradient Boosted Decision Trees (GBDT) Walkthrough Notes

### 1. Updated Master GBDT Walkthrough Table
* **Actual Targets**: Person 1 = 170 lbs, Person 2 = 190 lbs
* **Initial Baseline ($F_0$)**: $\frac{170 + 190}{2} = 180.0$ lbs
* **Learning Rate ($\nu$)**: 0.1
* **Loss Function**: Squared Error Loss = $\frac{1}{2}(\text{Actual} - \text{Predicted})^2$

| Stage | Action / Step | Person | Model Inputs (Features) | Current Prediction ($F$) | Target to Predict (Residual) | How Target was Formed | Tree Output | Learning Rate ($\nu$) | Prediction Update Formula | New Prediction | Stage Loss $\frac{1}{2}(y - F)^2$ | Total Stage Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Initialize Baseline** | **P1**<br>**P2** | None *(Global Mean)* | --<br>-- | **170**<br>**190** | Raw Target Weights | --<br>-- | -- | $F_0 = \text{mean}(y)$ | **180.0**<br>**180.0** | 50.00<br>50.00 | **100.00** |
| **1** | **Train & Run Tree 1** | **P1**<br>**P2** | Age: 25, Sex: M<br>Age: 45, Sex: F | 180.0<br>180.0 | **-10.0**<br>**+10.0** | $\text{Actual} - F_0$ | **-10.0**<br>**+10.0** | **0.1** | $F_1 = F_0 + (\nu \times \text{Tree}_1)$ | **179.0**<br>**181.0** | 40.50<br>40.50 | **81.00** |
| **2** | **Train & Run Tree 2** | **P1**<br>**P2** | Age: 25, Sex: M<br>Age: 45, Sex: F | 179.0<br>181.0 | **-9.0**<br>**+9.0** | $\text{Actual} - F_1$ | **-9.0**<br>**+9.0** | **0.1** | $F_2 = F_1 + (\nu \times \text{Tree}_2)$ | **178.1**<br>**181.9** | 32.81<br>32.81 | **65.61** |
| **3** | **Train & Run Tree 3** | **P1**<br>**P2** | Age: 25, Sex: M<br>Age: 45, Sex: F | 178.1<br>181.9 | **-8.1**<br>**+8.1** | $\text{Actual} - F_2$ | **-8.1**<br>**+8.1** | **0.1** | $F_3 = F_2 + (\nu \times \text{Tree}_3)$ | **177.29**<br>**182.71** | 26.57<br>26.57 | **53.14** |

---

### 2. The Ensemble Equation Freezes
Once stopping criteria are met (max trees reached, or early stopping rules trigger), training freezes into a permanent mathematical chain. 

For the 3-stage model above, the frozen production formula is:
$$F_3(x) = F_0 + (\nu \times \text{Tree}_1(x)) + (\nu \times \text{Tree}_2(x)) + (\nu \times \text{Tree}_3(x))$$

---

### 3. Inference Example (Predicting on New Data)

#### New Data Point:
* **Person 3**: Age = 25, Sex = Male

#### Execution Path Through the Frozen Chain:
1. **Base Leaf ($F_0$)**: Everyone starts at the global mean $\rightarrow \mathbf{180.0}$
2. **Tree 1 Evaluation**: Tree 1 sees `Age: 25` $\rightarrow$ Routes to left leaf $\rightarrow$ Outputs $\mathbf{-10.0}$
3. **Tree 2 Evaluation**: Tree 2 sees `Age: 25` $\rightarrow$ Routes to left leaf $\rightarrow$ Outputs $\mathbf{-9.0}$
4. **Tree 3 Evaluation**: Tree 3 sees `Age: 25` $\rightarrow$ Routes to left leaf $\rightarrow$ Outputs $\mathbf{-8.1}$

#### Final Summation:
$$F_3(\text{Person 3}) = 180.0 + (0.1 \times -10.0) + (0.1 \times -9.0) + (0.1 \times -8.1)$$
$$F_3(\text{Person 3}) = 180.0 - 1.0 - 0.9 - 0.81$$
$$\text{Final Prediction} = \mathbf{177.29 \text{ lbs}}$$

---

## 3. Pseudo-Residuals — Gradient Descent in Function Space

For **squared loss** L = (y − F(x))², the residual y − F(x) *is* (proportional to)
the **negative gradient** of the loss w.r.t. the prediction:

```
−∂L/∂F(x) = 2(y − F(x))  ∝  residual
```

Generalization (Friedman): at each stage fit the tree to the **pseudo-residuals**

```
rim = − [ ∂L(yi, F(xi)) / ∂F(xi) ]  evaluated at F = Fm-1
```

**Why this matters (the crux):** GBDT can minimize **any differentiable loss** —
squared/absolute/Huber loss for regression, logistic loss for classification,
ranking losses, custom losses. (Contrast: RF just optimizes impurity; it can't
target an arbitrary loss like hinge/logistic.)

### Friedman's Gradient Boosting Algorithm 🔍 *(Deep-dive — interview-level answer
is just "fit each tree to the negative gradients of the loss, scale by learning
rate." The per-leaf line search below is senior/research-round depth.)*
```
1. F0(x) = argmin_γ Σ L(yi, γ)            (for squared loss → γ = mean of y)
2. For m = 1 … M:
   a. rim = −∂L(yi, F(xi))/∂F(xi) |F=Fm-1      (pseudo-residuals)
   b. Fit a regression tree to {(xi, rim)} → terminal regions Rjm
   c. For each leaf j: γjm = argmin_γ Σ(xi∈Rjm) L(yi, Fm-1(xi) + γ)   (line search)
   d. Fm(x) = Fm-1(x) + ν · Σj γjm · 1(x ∈ Rjm)
3. Output FM(x)
```

---

## 4. GBDT for Classification (the version the walkthrough above skips)

Everything so far was regression. Interviewers often follow up with *"and how
does this work for classification?"* — the answer surprises people:

**The trees are still regression trees.** They always are. What changes is the
space you're working in.

| Step | Regression | Binary classification |
|---|---|---|
| Work in the space of | the target y | **log-odds** (the logit) |
| F0 = | mean of y | `log(p/(1−p))` of the base rate |
| Loss | squared error | **log loss** (binary cross-entropy) |
| Pseudo-residual | `y − F(x)` | `y − p` where `p = sigmoid(F(x))` |
| Trees fit | residuals | those same `y − p` values |
| Final output | `F(x)` directly | `sigmoid(F(x))` → a probability |

**The clean summary line:** *"boosting always adds up regression trees; for
classification it adds them up in log-odds space and squashes the sum through a
sigmoid at the end."*

**Why log-odds?** Predictions must stay in [0,1]. Adding trees directly to a
probability would blow past that range. Log-odds is unbounded, so you can add
freely, and the sigmoid maps back into a valid probability.

**The pleasing part:** the pseudo-residual for log loss works out to exactly
`y − p` — observed label minus predicted probability. Same shape as the
regression residual, which is why the algorithm looks identical from the outside.

Multi-class: fit **one tree per class per round** (K trees per iteration), then
softmax. That's why multi-class boosting is ~K× more expensive.

---

## 5. Regularization

- **Shrinkage / learning rate ν (0 < ν ≤ 1, typically 0.1):** scales each tree's
  contribution. Smaller ν → less overfitting/variance, but needs larger M.
  **M and ν trade off** — tune jointly with cross-validation.
- **Subsample < 1 (Stochastic Gradient Boosting):** each tree fits a random
  fraction of rows → adds bagging-style variance reduction.
- **Tree constraints:** max_depth (default 3 in sklearn), min_samples_leaf.
- **More trees CAN overfit** (unlike RF!) — M is a real capacity knob; use early
  stopping on a validation set.

### Tuning in practice — the order to do it in ⭐
Asked constantly as *"how would you tune an XGBoost model?"* A structured answer
beats "grid search everything":

| Order | Knob | Typical range | What it controls |
|---|---|---|---|
| 1 | `learning_rate` (ν) | 0.01–0.1 | Set it **low and leave it** (0.05 is a fine default) |
| 2 | `n_estimators` (M) | 100–2000 | Don't tune by hand — use **early stopping** on a validation set |
| 3 | `max_depth` / `num_leaves` | 3–8 / 31–127 | Model capacity; **the highest-leverage knob** |
| 4 | `subsample`, `colsample_bytree` | 0.6–1.0 | Row/column sampling → variance reduction |
| 5 | `min_child_weight`, `lambda`, `gamma` | problem-specific | Fine-grain regularization, only if still overfitting |

**The ν ↔ M trade-off in one sentence:** halving the learning rate roughly
doubles the trees you need. Lower ν almost always generalizes better, so the real
strategy is *fix a small ν, then let early stopping pick M*.

### Why shallow trees? (depth = interaction order)
A tree of depth **d** can express interactions among at most **d features** —
depth 1 (a stump) is an additive model with no interactions at all, depth 2
captures pairwise interactions, and so on.

So `max_depth` isn't just an abstract capacity dial; it's a statement about
**how many features you think interact at once**. Depth 3–8 covers the
interaction order that real tabular data usually has, and going deeper mostly
buys variance. This is a strong thing to say when asked why boosting uses shallow
trees while Random Forest grows them fully.

### Complexity
- Train: O(n log n · M) — **stages are sequential**, not parallelizable across
  trees (modern libraries do parallelize *within* a tree's split search)
- Test: O(depth · M) — shallow trees → very low latency
- Space: O(trees + leaf values γ)
- sklearn defaults: `learning_rate=0.1`, `max_depth=3`, `n_estimators=100`

---

## 6. AdaBoost (Adaptive Boosting)

The original boosting algorithm — reweights **points** instead of fitting residuals.

### The weighted training set (the mechanism)
Normally every example counts equally: `error = Σ (1/N)·1[h(xᵢ) ≠ yᵢ]`.
Boosting replaces the uniform 1/N with a **per-example weight w⁽ⁱ⁾**:

```
weighted error  ε = Σ w⁽ⁱ⁾ · 1[h(xᵢ) ≠ yᵢ]        (weights sum to 1)
```

Higher weight = higher cost of getting that point wrong = the next learner
"tries harder" on it. This is how boosting *shifts focus* to hard examples —
and that shifting focus is also what decorrelates consecutive learners.

### The algorithm
1. Initialize equal weights wi = 1/n (weights always sum to 1)
2. Train a weak learner (a **stump**: root + 2 leaves); compute weighted error ε
3. Compute the model's **"amount of say"**: α = ½·ln((1−ε)/ε)
4. **Increase weights of misclassified points** (×e^α), decrease weights of
   correct ones (×e^−α); renormalize to sum 1
5. Next stump focuses on the hard (high-weight) points; repeat
6. Final prediction = sign(Σ αm·hm(x)) — a **weighted** vote

### Understanding α (know the shape of this curve)
| ε (weighted error) | α | Meaning |
|---|---|---|
| → 0 (near-perfect stump) | → +∞ | Huge say in the final vote |
| = 0.5 (coin flip) | **0** | No say at all — useless model, ignored |
| > 0.5 (worse than chance) | **negative** | Its vote gets *flipped* — a consistently wrong classifier is still informative |

α falls steeply as ε rises: it's ~2.2 at ε=0.01, ~1.1 at ε=0.1, ~0.2 at ε=0.4,
and 0 at ε=0.5. So good stumps dominate the vote and mediocre ones fade out.

> 📎 **Two conventions in the wild.** ESL's AdaBoost.M1 writes `α = ln((1−ε)/ε)`
> without the ½; the exponential-loss derivation gives `α = ½·ln((1−ε)/ε)`.
> They differ by a constant factor, and since the prediction is
> `sign(Σ αm hm)`, scaling *all* α by 2 doesn't change a single prediction.
> Use ½ (it's the derivation-correct one) and note the other exists if pushed.

### Why renormalize the weights?
Not cosmetic. A point the weak learners keep getting right has its weight
multiplied by e^(−α) every round → it **underflows toward zero**. A point they
keep getting wrong blows up. After enough iterations that's numerical
instability. Renormalizing to sum to 1 each round keeps the weights a proper
distribution and the arithmetic well-scaled.

Classic application: **face detection (Viola–Jones)** in computer vision,
combined with a cascade.

### AdaBoost vs Gradient Boosting

**The one-line unifying idea:** both ask *"where is the current ensemble falling
short?"* and build the next learner to fix it. They differ only in how
"shortcoming" is measured — AdaBoost reads it off **high-weight data points**,
gradient boosting reads it off **gradients**.

| | AdaBoost | Gradient Boosting |
|---|---|---|
| "Shortcomings" identified by | **High-weight points** (re-weighting) | **Gradients** (pseudo-residuals) |
| Corrects errors via | **Re-weighting** misclassified points | Fitting **pseudo-residuals** (negative gradients) |
| Base trees | Stumps (1 split) | Shallow trees (≈8–32 leaves) |
| Loss | Exponential loss (implicitly — AdaBoost is GB with exponential loss) | Any differentiable loss |
| Model weights | Unequal "say" α per stump | Uniform ν·γ (learned leaf values) |

### AdaBoost vs Random Forest
1. AdaBoost = boosting; RF = bagging
2. AdaBoost sequential (stage m depends on m−1); RF trees independent/parallel
3. AdaBoost uses stumps; RF grows full trees
4. AdaBoost models have unequal say (α); RF trees vote equally

---

## 7. XGBoost

**XGBoost ≈ GBDT + row sampling + column sampling + heavy regularization +
systems engineering** (it borrows RF's sampling ideas):

- **Regularized objective:** loss + Ω(tree) where Ω = γ·(#leaves) + ½λ·Σ(leaf weights)²
- 🔍 *Deep-dive:* uses **second-order Taylor expansion** (gradients *and*
  hessians) for split gain — senior-ML-engineer follow-up; the expected answer is
  the regularization + missing-value + sampling line above
- **Sparsity-aware:** learns a **default direction** per split for missing values
- Parallelized/cache-optimized split finding, histogram approximation
- Released 2014; huge community; `objective='reg:squarederror'`,
  `'binary:logistic'`, etc.; supports custom objectives

> Relation to linear models: gradient boosting is gradient descent in function
> space — same descent idea as training linear regression, but each "step" is a
> whole tree instead of a parameter update.

---

## 8. XGBoost vs LightGBM vs CatBoost

| | **XGBoost** (2014) | **LightGBM** (Microsoft, 2016) | **CatBoost** (Yandex, 2017) |
|---|---|---|---|
| Tree growth | **Level/depth-wise** — grows whole levels; different conditions per node | **Leaf-wise** — repeatedly split the single leaf with max gain → asymmetric trees | **Symmetric (oblivious)** — same split condition for all nodes at a level |
| Categorical features | Encode manually (one-hot/target); newer versions have experimental native support | Native — bin/bucket-based grouping of categories | Native — **Ordered Target Encoding** (leakage-safe); `one_hot_max_size` controls small categories |
| Sampling | Row/column subsample (without replacement) | **GOSS** — Gradient-based One-Side Sampling | Minimal-variance / uniform sampling; 🔍 Ordered Boosting (internals rarely asked) |
| Speed tricks | Histogram, parallel split search | Histogram **binning**, **EFB** (Exclusive Feature Bundling) | Symmetric trees → very fast inference; strong GPU support |
| When to pick | Max control/tuning, community support, robust all-rounder | **Speed** on large data; good results with little tuning | **Many categorical features**; large data; GPU |

### LightGBM speed tricks (know these!)
1. **Histogram binning:** bucket continuous values (e.g. salary → 50–80k, 80–90k)
   and search splits over bins, not raw values
2. **EFB (Exclusive Feature Bundling):** mutually exclusive sparse features
   (e.g. one-hot columns where only one is nonzero) get bundled into a single
   feature → fewer columns
3. **GOSS:** keep the **large-gradient points** (poorly predicted → most to
   learn from) and randomly sample only a fraction of the small-gradient rest
   (already well fit). 🔍 *Deep-dive:* the sampled small-gradient points are
   re-weighted by (1−a)/b to keep gradients unbiased — implementation trivia,
   rarely asked
4. **Leaf-wise growth:** spend splits only where loss reduction is largest
   (deeper, asymmetric trees; cap `num_leaves`/depth to avoid overfitting)

---

## 9. Practical: Advantages, Limitations, When to Use

### Advantages
- **Best-in-class accuracy on tabular data** — the default answer for structured
  data problems
- **Any differentiable loss** → huge flexibility (regression, classification,
  ranking, custom business losses)
- **No feature scaling needed**, handles mixed numeric/categorical (with the
  right library — see caveats)
- **Fast inference** despite many trees: the trees are shallow, so
  `O(depth × M)` is small

### Limitations
- **Overfits if unchecked** — it will keep driving training error down, chasing
  outliers. Requires early stopping / CV.
- **Sensitive to noisy labels and outliers** — it *by design* focuses on the
  points it gets wrong, and a mislabeled point is permanently "wrong"
- **Computationally expensive** — often hundreds to >1000 trees, trained
  sequentially
- **Many interacting hyperparameters** (M, ν, depth, subsample, λ, γ) → real
  tuning budget needed
- **Not interpretable** — a black box like RF, worse

### When to use vs avoid

| ✅ Use boosting | ❌ Avoid / think twice |
|---|---|
| You need **maximum accuracy** on tabular data | You need **interpretability** → linear model, single tree, or add PDP/SHAP |
| **Complex non-linear** relationships and interactions | The relationship is genuinely **linear** → linear regression / LASSO |
| You have the **compute and tuning budget** | **Limited compute** → RF, or use LightGBM/histogram methods |
| Enough data to support a high-capacity model | **Small datasets** → it will overfit; lean on regularization or use RF |
| | **Noisy / mislabeled data** → prefer bagging |
| | **Heavy class imbalance** → adjust `scale_pos_weight` / class weights first |

> ⚠️ **Correct this before an interview:** one slide lists "robustness to noise"
> as a *reason to use* boosting. That is backwards, and the same deck contradicts
> it two slides later ("can overemphasize outliers"). **Bagging is the
> noise-robust one** (averaging cancels noise); **boosting is noise-sensitive**
> because re-weighting/residual-fitting makes it chase mislabeled points
> relentlessly. This is a classic exam question — get the direction right.
>
> ⚠️ Also treat "handles missing data / categorical features natively" with care.
> That's **XGBoost** (default direction per split for missing) and **LightGBM /
> CatBoost** (native categoricals). Vanilla `sklearn.GradientBoosting` does
> neither — say *which library* you mean.

---

## 10. Frequently Asked Interview Questions

1. Boosting vs bagging: which term of the error does each reduce, and what base
   learners does each use? Why sequential vs parallel?
2. Walk through GBDT for regression by hand: initial prediction, residuals,
   learning-rate update, final prediction.
3. Show that for squared loss the residual is the negative gradient. What is a
   pseudo-residual and why does it let GB minimize any differentiable loss?
4. 🔍 (Senior rounds) Write/explain Friedman's gradient boosting algorithm (init,
   pseudo-residuals, fit tree, leaf line-search, shrinkage update). Standard
   rounds only expect the one-line version from Q3.
5. What does the learning rate (shrinkage ν) do? Describe the M–ν trade-off and
   how you'd tune both.
6. Can gradient boosting overfit as you add trees? How do you detect and prevent
   it? (Contrast with RF.)
7. How does AdaBoost work: weight updates, amount of say α = ½ln((1−ε)/ε),
   final weighted vote. How does it relate to GB? (Exponential loss.)
8. AdaBoost vs GBDT vs Random Forest — a three-way comparison.
9. What does XGBoost add on top of vanilla GBDT? (Regularization Ω, second-order
   gradients, sparsity-aware missing handling, sampling, systems optimizations.)
10. How does XGBoost handle missing values at split time?
11. Level-wise vs leaf-wise vs symmetric tree growth — which library does which,
    and the overfitting/speed implications of each.
12. Explain GOSS and EFB. Why do they make LightGBM fast without hurting accuracy?
13. What is CatBoost's ordered target encoding, and what leakage problem does it
    solve compared to naive target/response encoding?
14. You have a dataset with 200 high-cardinality categorical columns — which
    booster do you reach for and why?
15. Why is boosting hard to parallelize across trees, and what do modern
    libraries parallelize instead?
16. Train/test complexity of GBDT; why is it good for low-latency serving despite
    having M trees?
17. Define a **weak learner** and a **strong learner**. What question did AdaBoost
    answer, and why did it matter?
18. What are the **three ingredients** any boosting algorithm needs?
19. What is a **weighted training set**, and how does the weighted error rate
    differ from ordinary error?
20. Sketch α as a function of weighted error. What is α when ε = 0.5? What if
    ε > 0.5?
21. Why must AdaBoost **renormalize** the weights every round?
22. In a GBDT tree, several residuals land in one leaf. What value does the leaf
    output, and why?
23. **Is boosting robust to noisy labels?** (No — bagging is. Know why.)
24. Which is more sensitive to outliers, bagging or boosting, and what's the
    mechanism?
25. When would you pick Random Forest *over* a gradient booster?
26. **How does gradient boosting work for classification?** What is F0, what
    space do the trees live in, and how do you get a probability out?
27. Are the trees in a gradient boosting *classifier* classification trees?
    (No — regression trees, fit to `y − p` in log-odds space.)
28. **How would you tune an XGBoost model?** In what order, and which parameter
    would you not tune by hand?
29. Why does boosting use shallow trees while RF grows them fully? (Bias vs
    variance, plus depth = maximum interaction order.)
30. What happens to the number of trees needed if you halve the learning rate?
