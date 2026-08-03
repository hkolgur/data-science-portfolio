# Random Forest — Interview Notes

## 1. Definition

**Random Forest = Bagging with Decision Trees + Column (feature) Sampling**

```
RF = Base learners (deep DTs) + Row sampling (bootstrap) + Column sampling + Aggregation
```

Two sources of randomness decorrelate the trees:
1. **Row sampling (bagging):** each tree trains on a bootstrap sample
   (with replacement) — ≈63.2% unique points in-bag, ≈36.8% out-of-bag
2. **Column/feature sampling:** each **split** considers only a random subset of
   features — typically **√d** for classification, **d/3** for regression
   (feature subsets are drawn **without** replacement; "with replacement"
   applies only to the row bootstrap)

Aggregation: **majority vote** for classification; **mean** (sometimes median)
for regression.

"Why do you use Random Forests over a regular Decision Tree?
I use Random Forests over a single Decision Tree primarily to combat variance and overfitting. A single decision tree has high capacity and can easily memorize training noise, leading to high variance. A Random Forest solves this by combining an ensemble of trees to average out that noise, giving us a far more stable and generalizable model."

## Single Decision Tree vs. Random Forest

* **The Problem:** Single trees have high variance, low bias, and overfit easily by creating deep, complex decision boundaries.
* **The Solution (RF):** Keeps the low bias of trees but slashes variance by averaging an ensemble of uncorrelated trees.
* **The Variance Killers:** 
  1. Row Bootstrapping (Bagging)
  2. Column Subsampling (Forces tree diversity so they don't all split on the same dominant feature)
* **Production Perks:** Free validation via OOB error, highly robust to outliers, and built-in feature importance for business transparency.


---

## 2. Why does RF work? (Decorrelation — key interview story)

Example: predicting employee attrition with features {salary, age, commute}.
If salary is the strongest feature, **every** bagged tree would split on salary
first → trees are highly correlated → averaging correlated models barely reduces
variance.

With **column sampling**, some trees never see salary and are forced to learn
patterns from age/commute. These "different viewpoints" make tree errors less
correlated → averaging now genuinely cancels errors → the forest captures the
data's patterns "from various ends."

> Averaging k models reduces variance by ~1/k only if errors are independent;
> column sampling is what pushes tree errors toward independence.

### The formula version (say this and you've answered the question completely)
```
Var(forest) = ρσ² + (1 − ρ)·σ²/k
```
Adding trees only shrinks the **second** term. `ρσ²` is a **floor** that more
trees can never get below. Bagging alone leaves ρ high (every tree splits on
salary first); **column sampling is the only lever RF has on ρ**, which is why
it's not optional — it's what separates RF from plain bagged trees.

Trade-off to name: lowering `max_features` pushes ρ down (good) but also makes
each individual tree weaker (raises σ² and bias slightly). `√d` is the empirical
sweet spot, not a theorem.

---

## 3. Bias–Variance in RF

- Base trees are grown **deep / fully (even overfitting)** → each is
  **low bias, high variance**. That's fine — aggregation will fix the variance.
- **Final bias ≈ base-tree bias** (aggregation doesn't add bias)
- **Variance decreases as k (number of trees) increases** — and as row/column
  sampling rates decrease (more randomness → less correlation → lower variance,
  at some cost in individual-tree strength)
- **More trees never overfit** — test error plateaus as k grows; k is limited by
  compute, not overfitting. Tune k with CV or the OOB score.

---

## 4. Out-of-Bag (OOB) Evaluation (oob_score=True)

For each tree Mi, the ≈36.8% points not in its bootstrap sample (OOB points)
serve as a **free validation set**:

- Predict each training point using only the trees for which it was OOB
- Aggregate → **OOB score** ≈ cross-validation accuracy, with **no extra data
  split and no extra training**
- In sklearn: `oob_score=True`
- If checking independently no need of k-fklod. But if we compare algorithms performance with multiple we may need to keep same folds in each algorithm. In such case turn off the OOB so it will save computation time and K-fold can be used.
  
**Why it's a valid estimate:** a point is predicted only by trees that never saw
it, which is exactly the condition a validation set satisfies. So OOB gives a
nearly unbiased estimate of generalization error essentially for free — the
single best practical argument for RF on small-to-medium data, where you'd
otherwise sacrifice rows to a holdout.

> ⚠️ Caveats worth having ready: OOB uses only ~37% of the trees per point, so it
> is **slightly pessimistic** for small k; it is unreliable when k is small; and
> it does **not** replace a proper test set when your data has time ordering or
> grouped/duplicated rows (the bootstrap ignores both).

---

## 5. Time & Space Complexity

| Phase | Cost | Notes |
|---|---|---|
| Train | O(n log n · d′ · k) | k trees; **trivially parallelizable** (`n_jobs=-1`) — trees are independent |
| Test | O(depth · k) | fast, but k× slower than one tree |
| Space (run) | O(nodes per tree · k) | store k trees |

RF's parallel training is a major practical advantage over boosting (whose
stages are sequential).

---

## 6. sklearn Hyperparameters

- `n_estimators` — number of trees k (more is better until plateau)
- `max_features` — features per split (`sqrt` default for classification)
- `max_depth` — usually left `None` (grow fully; bagging handles variance)
- `min_samples_split`, `min_samples_leaf` — per-tree regularization if needed
- `oob_score=True` — free validation estimate
- `class_weight='balanced'` — for imbalanced data
- `n_jobs=-1` — use all cores

---

## 7. Extremely Randomized Trees (ExtraTrees) 🔍 *(occasional one-liner, not core)*

Variation of RF with **even more randomness**:
- Instead of searching the **best threshold** for each candidate feature,
  pick thresholds **at random** and keep the best random split
- (sklearn's `ExtraTreesClassifier` also uses the whole dataset per tree by
  default — no bootstrap)

Effect: **faster training** (no threshold search), **lower variance**, slightly
**higher bias**. Useful when RF still overfits or training time matters.

---

## 8. Practical Considerations

### Feature importance — the three flavors (know the differences) ⭐
How do you know which features your model actually cares about?
Avoid using the default feature_importances_ (MDI) from Scikit-Learn for anything that impacts business decisions, because it is heavily biased toward high-cardinality and continuous variables, and it ignores overfitting.

Instead, run Permutation Importance on a held-out validation set. Shuffling the data points on unseen data gives an honest, unbiased global ranking of what the model actually relies on to generalize.

## Permutation Importance on Held-Out Data:

### 1. Step-by-Step Algorithm
1. Calculate baseline metric (e.g., AUC = 0.85 or  R²) on an unseen validation set.
2. For each feature:
   * You randomly shuffle the values of a single feature column in that held-out dataset, keeping all other columns unchanged..
   * Pass the altered dataset through the trained model.
   * Calculate the new validation metric (e.g., AUC drops to 0.60 or R²).
   * Feature Importance = Baseline Score - New Score (0.85 - 0.60 = 0.25).The difference between the baseline performance and the new performance is       that feature's importance score.
3. Reset column and repeat for the next feature/column.

### 2. Why it is Superior to Default Gini / MDI
* **No Overfitting Reward:** Evaluated on unseen data, meaning it measures real predictive power, not training data memorization.
* **No Cardinality Bias:** Does not artificially favor continuous or high-cardinality numbers just because they have more split points.


To explain a specific model decision to a stakeholder or an auditor—for instance, exactly why a customer was denied a credit card—I use SHAP


## The Three Flavors of Feature Importance

| Method | How it works | Critical Defect | Best Use Case |
| :--- | :--- | :--- | :--- |
| **MDI / Impurity** (`sklearn` default) | Tracks average impurity decrease during tree training splits. | **Highly biased** toward continuous/high-cardinality features. Rewards overfitting. | Never trust for real business decisions; use only for fast, dirty sanity checks. |
| **Permutation Importance** | Shuffles one column's rows on validation data; measures drop in accuracy. | Can split/hide credit unpredictably between highly correlated features. | **Default choice** for finding a true, unbiased global ranking of feature power. |
| **SHAP** | Uses game theory to calculate per-prediction feature attribution. | Exceptionally slow and computationally expensive to calculate. | Mandatory when you need **per-row transparency** for stakeholders or risk auditors. |

**The correlated-features trap** (very common follow-up): with two highly
correlated features, RF splits on them roughly at random across trees, so *each*
gets ~half the importance. Neither looks important, even though the underlying
signal is strong. Never conclude "this feature doesn't matter" from a low
importance score without checking correlations first.

### Missing values
RF has **no native handling** — `sklearn`'s implementation requires you to impute
before fitting. (Contrast: XGBoost learns a default direction per split, so it
handles NaNs natively. This is a real, frequently-asked difference between RF and
boosted libraries.)

Practical options: median/mode imputation, or add a binary `was_missing`
indicator column so the tree can split on missingness itself — often the better
move when the data is **not** missing at random.

### Other practical points
- **Cases:** everything from the Decision Tree notes carries over (no scaling
  needed, native multi-class, no similarity-matrix input, axis-parallel
  boundaries, response-encode high-cardinality categoricals) — *except* the
  bias–variance handling, which bagging changes.
- **Imbalanced data:** still biased to majority class → `class_weight`,
  resampling, or `BalancedRandomForestClassifier` (imblearn).
- **Cannot extrapolate** (regression) — predictions bounded by training targets.

### Disadvantages
1. Training slows with **high dimensionality** (many features per split search)
2. **Less interpretable** than one tree — hundreds of trees ≈ black box
   (use feature importance, **PDP / ICE plots**, or SHAP for explanations)
3. Larger memory footprint and higher prediction latency than a single tree
4. Usually beaten on accuracy by well-tuned gradient boosting on tabular data

---

## 9. RF vs Single DT vs GBDT

| | Single DT | Random Forest | GBDT |
|---|---|---|---|
| Ensemble type | — | Bagging + column sampling | Boosting |
| Base tree depth | tuned (5–10) | deep/full | shallow (3–8) |
| Reduces | — | Variance | Bias |
| Training | fast | parallel | sequential (stages) |
| Overfitting w/ more models | n/a | No (plateaus) | Yes (tune M, lr) |
| Tuning effort | low | **low (robust defaults)** | high |
| Typical accuracy (tabular) | baseline | strong | strongest |

---

## 10. Frequently Asked Interview Questions

1. What two kinds of sampling does RF do, and why is column sampling essential
   on top of bagging? (Decorrelation story.)
2. Why are RF base trees grown fully instead of pruned?
3. Show that ≈63.2% of points are in-bag. What is the OOB score and why is it a
   valid CV substitute?
4. Does adding more trees overfit an RF? Why not? What *is* the hyperparameter
   story for RF vs GBDT?
5. Bias and variance of the forest vs the base trees — what changes, what doesn't?
6. Why is RF training trivially parallelizable but boosting isn't?
7. max_features: what are the defaults for classification vs regression, and
   what happens as you lower it?
8. 🔍 (Occasional) What are Extremely Randomized Trees and when would you prefer them?
9. How is feature importance computed in RF, and what are its pitfalls?
9b. Impurity vs permutation vs SHAP importance — when would you use each?
9c. Two of your features are highly correlated and both show low importance.
    What's going on?
9d. How does RF handle missing values? (It doesn't — contrast with XGBoost.)
10. Train/test complexity of RF; how does it behave in very high dimensions?
11. Why might a tuned XGBoost beat RF on tabular data — and when would you still
    pick RF? (Low tuning budget, parallel training, robustness.)
12. Can RF extrapolate in regression? Why not?
