l# Ensembles (Bagging, Boosting, Stacking, Cascading) — Interview Notes

## 1. What is an Ensemble?

An **ensemble** combines multiple models ("base learners") so the group predicts
better than any single model — the ML version of "wisdom of crowds."

**Key requirement: diversity.** The more *different* (decorrelated) the base
models' errors are, the more the combination helps. Uncorrelated mistakes cancel
out when aggregated; identical models add nothing.

### Why it works — the number that makes it click ⭐
Take 3 independent models, each **70% accurate**, and majority-vote them. The
ensemble is right whenever ≥2 are right:

```
P(all 3 right)      = 0.7³              = 0.343
P(exactly 2 right)  = 3 × 0.7² × 0.3    = 0.441
                                   Total = 0.784   →  78.4%
```

**Three 70% models became a 78% model.** Push it to 5 models → 83.7%; to 101
models → >99.9%. This is the whole promise of ensembling in one calculation, and
it's a great thing to have ready when asked "why do ensembles work?"

**Now the two conditions the math quietly assumes** — and naming these is what
separates a strong answer:
1. **Each model must beat chance** (>50%). Voting models that are 40% accurate
   makes things *worse*, and just as fast.
2. **Errors must be independent.** If all 3 models fail on the same examples,
   majority vote returns exactly 70% — you gained nothing. Real models are
   partially correlated, so you land somewhere between 70% and 78%.

Condition 2 is why every ensemble method is fundamentally a **diversity-injection
scheme**: bagging varies the rows, Random Forest also varies the columns,
boosting varies the focus, stacking varies the algorithm.

Four main families:
1. **Bagging** (Bootstrap Aggregation) — e.g. Random Forest
2. **Boosting** — e.g. AdaBoost, GBDT, XGBoost
3. **Stacking** (and Blending)
4. **Cascading**

---

## 2. Bias–Variance Decomposition (the "why" of ensembles)

```
Expected Test Error = Bias² + Variance + Irreducible error (noise)
```

> ⚠️ The third term is **irreducible error** (noise inherent in the data),
> not "generalization error."

### 2.1 Why do the decomposition at all? (It's a debugging tool)

You train a model, compute test MSE, and it's high. *Now what* — collect more
data? add features? go bigger? add regularization? These pull in opposite
directions, so guessing wastes weeks.

Bias–variance decomposition converts **"my error is high"** into **"my error is
high *because of this specific term*,"** which maps to a specific fix. That is
the entire point of it, and it's the answer to "why do we do bias-variance
decomposition?" — a debugging framework, not a piece of trivia.

### 2.2 Setup

Assume training and test points are drawn from the same distribution:

```
y = f(xᵢ) + εᵢ      with   E[εᵢ] = 0,   Var(εᵢ) = σ²
```

- `f` = the true (unknown) function we're trying to recover
- `f̂` = the estimate we fit from a training set D

**Key point people miss:** `f̂` is itself a *random variable*. It depends on
which training set D we happened to draw. Every expectation `E[·]` below is over
**random draws of the training set, at a fixed x** — not over the data points in
one dataset.

### 2.3 The derivation

**What you actually need to produce on demand** is the result plus the *shape* of
the argument, not the algebra:

> "Split the error into noise plus estimation error. Then split estimation error
> by adding and subtracting the average model E[f̂]: what's left is how far the
> average model is from the truth (bias²) plus how much any one model deviates
> from that average (variance). The cross terms vanish."

That answer is enough for a DS interview. The algebra below is for when someone
says "show me."

<details>
<summary>🔍 <b>Full algebra</b> (senior/quant rounds — skip on a first pass)</summary>

**Step 1 — peel off the noise.**

```
Test MSE = E[(y − f̂(x))²]
         = E[(ε + f(x) − f̂(x))²]
         = E[ε²] + E[(f(x) − f̂(x))²]
```

The cross term drops because ε is independent of f̂ and E[ε] = 0.
`E[ε²] = σ²` → the irreducible error.

**Step 2 — add and subtract E[f̂(x)] inside the second term.**

```
E[(f − f̂)²] = E[ ((f − E[f̂]) − (f̂ − E[f̂]))² ]
            = (E[f̂] − f)²          ← bias²
            + E[(f̂ − E[f̂])²]       ← variance
            − 2·E[(f − E[f̂])(f̂ − E[f̂])]
```

**Why the cross term is zero:** `(f − E[f̂])` is a *constant* with respect to the
randomness in D, so it factors out of the expectation, and `E[f̂ − E[f̂]] = 0` by
definition of the mean.

</details>

```
⇒  Expected Test MSE = Bias[f̂]² + Var(f̂) + σ²
```

**One scope caveat worth knowing:** this is derived for **regression with squared
loss**. It doesn't decompose this cleanly for classification with 0–1 loss — but
everyone still uses "bias" and "variance" as shorthand for underfitting and
overfitting there. Just don't claim the *equation* holds exactly for
classification.

### 2.4 The three terms

| Term | Formula | Means | Symptom |
|---|---|---|---|
| **Bias²** | `(E[f̂(x)] − f(x))²` | How wrong the *average* model is | Underfitting |
| **Variance** | `E[(f̂(x) − E[f̂(x)])²]` | How much predictions wobble as the training data changes | Overfitting |
| **Irreducible / Bayes error** | `σ²` | Inherent unpredictability of the target | Nothing fixes it |

> The slides call the third term **Bayes error**; other sources call it
> irreducible error or noise floor. Same thing. Know both names.

### 2.5 The one-line mnemonic ⭐

| | **Bias** | **Variance** |
|---|---|---|
| **Shorthand** | **actual vs. predicted** | **predicted vs. average of all predictions** |
| **Formula** | `E[f̂(x)] − f(x)` | `E[(f̂(x) − E[f̂(x)])²]` |
| **Measured against** | the ground truth | the model's *own* mean prediction |
| **Where you see it** | **training error is already high** | **large train → test gap**; predictions swing on retraining |
| **Fix** | more complex model, more/better features, less regularization, **boosting** | more data, regularization, simpler model, **bagging** |

**Say the shorthand, then immediately qualify it — that's what separates a good
answer from a memorized one:**

- **"Bias = training error"** is a *symptom*, not the definition. High bias ⇒ the
  model can't fit even the data it already saw, so training error is high. But
  the converse doesn't hold: high training error can also just be a noisy label
  set (high σ²). Bias is formally the gap between the *average model over many
  training sets* and the truth.
- **"Variance = test error"** is too loose to survive a follow-up. Test error =
  bias² + variance + noise, so test error alone doesn't isolate variance. The
  clean statement: **variance is the spread of f̂(x) around its own mean, at the
  same x, across different training sets.** You observe it as the *train–test
  gap*, or by refitting on bootstrap samples and watching one point's prediction
  jump around.
- **"Average of all predicted values"** = the average across **models trained on
  different samples of the data**, evaluated at *one* x. It is *not* the average
  of your predictions across the test set — that quantity is just the mean
  prediction and carries no information about variance.

**Dartboard picture (slides 20–21):** bullseye = truth `f(x)`; each dart = the
prediction of a model trained on a different sample.
- **Bias** = how far the *centre of the cluster* sits from the bullseye
- **Variance** = how *spread out* the cluster is
- Four quadrants: low/low (goal), low bias–high variance (overfit, scattered
  around the target), high bias–low variance (underfit, tight cluster in the
  wrong place), high/high (worst).

### 2.6 Diagnosing from train vs. test error (the practical version)

| Training error | Test error | Diagnosis | What to do |
|---|---|---|---|
| High | High (≈ train) | **Underfitting / high bias** | Bigger model, more features, less regularization, train longer, **boosting** |
| Low | High (big gap) | **Overfitting / high variance** | More data, regularization, feature selection, simpler model, early stopping, **bagging / Random Forest** |
| Low | Low | Well fit | Ship it |
| High | Low | Suspicious | Check the split, leakage, or train-time-only dropout/augmentation |

> **Always compare against the irreducible floor.** If Bayes error is ~10%, a 12%
> test error is nearly optimal — not a bias problem. Chasing it wastes effort.
> Human-level performance is the usual proxy for this floor.

### 2.6b "Will more data help?" — learning curves ⭐

Extremely common interview question, and the train/test table above can't answer
it. Plot **training error and validation error vs. training set size**:

| Pattern | Diagnosis | Will more data help? |
|---|---|---|
| Curves **converge to a high error**, small gap | **High bias** | **No.** Both curves have flattened — more rows of the same thing won't help. Add capacity/features instead. |
| Curves still have a **large gap**, validation still falling | **High variance** | **Yes.** Keep collecting; the gap closes as n grows. |
| Curves converged, error near the noise floor | Well fit | No — you're done |


The intuition: **more data reduces variance, not bias.** A linear model fed a
million points is still a linear model. So "should we spend three months
labelling more data?" is a bias-vs-variance question, and learning curves are how
you answer it with evidence rather than a hunch.

![Training Error-Test Error VS Size of Test Data](images/train-test-error.png)
Above we see as the Test Data size increases beyond 8000 points, the training error doesn't improve. 

### 2.7 The complexity curve (slide 28)

As model complexity increases:
- **Bias²** falls monotonically
- **Variance** rises
- **Total error** is **U-shaped** → minimum at the sweet spot

Left of the minimum = **underfitting zone**, right = **overfitting zone**.

> ⚠️ **Two slide statements to correct before you memorize them:**
> 1. Slide 27's "inherent trade-off between **noise** and variance" is a typo/error
>    — the trade-off is between **bias and variance**. Noise is a constant floor
>    and doesn't trade off against anything.
> 2. Slide 32's "reducing bias increases variance and vice versa" is a **rule of
>    thumb, not a law.** Counterexamples worth naming:
>    - **Ensembles** — bagging cuts variance at essentially no bias cost. If the
>      trade-off were strict, ensembles couldn't work.
>    - **More training data** — reduces variance at fixed bias.
>    - 🔍 **Double descent** — in heavily overparameterized nets, test error falls
>      *again* past the interpolation threshold, so the curve isn't a clean U.
>      (Deep-learning roles only.)

### 2.8 Worked example: KNN — the classic "high bias or high variance?" (slides 29–30)

| | **k = 1** | **k large (→ n)** |
|---|---|---|
| Model is | Overly **complex** | Overly **simple** |
| Bias | **Low** — can represent any structure | **High** — approaches the global mean/majority, can't capture structure |
| Variance | **High** — one relabeled point moves the decision boundary | **Low** — averaging many neighbours gives a concentrated, stable estimate |
| Training error | 0 (each point is its own neighbour) | High |

**The counterintuitive bit to state explicitly:** in KNN, **k is an *inverted*
complexity knob** — small k = complex model, large k = simple model. Choose it by
cross-validation.

**Same pattern, other models** (good to rattle off — it shows you understand the
concept, not one example):

| Knob | High bias end | High variance end |
|---|---|---|
| Tree depth | depth-1 stump | fully grown tree |
| Polynomial degree | linear | degree 10 |
| Regularization λ (ridge/lasso) | large λ | λ → 0 |
| NN size / epochs | tiny net, early stop | huge net, trained to convergence |
| KNN k | large k | k = 1 |

### 2.9 Simple vs. complex — the practical framing (slides 17–18)

| | Explainable | Accurate |
|---|---|---|
| Simple model | ✅ | ⚠️ |
| Complex model | ⚠️ | ✅ |

Regulated domains (credit scoring, healthcare, hiring) may **force** the simple
model regardless of accuracy, because decisions must be explainable. Ensembles
are precisely the attempt to buy complex-model accuracy while keeping variance
under control — and the bill comes as lost interpretability plus serving cost.

### 2.10 Which term does each ensemble attack

| | Bagging | Boosting |
|---|---|---|
| Attacks | **Variance** | **Bias** |
| Base learners should be | **Low bias, high variance** (deep/fully-grown trees) | **High bias, low variance** (shallow trees / stumps) |
| Training | **Parallel** (models independent) | **Sequential** (each model fixes the previous one's errors) |
| Aggregation | Majority vote / mean | Weighted additive sum |
| Effect on the other term | Bias essentially unchanged | Variance *can* rise, but not by much |
| Noise/outlier sensitivity | Robust (averaging) | **Sensitive** — re-weights hard points, so mislabels get chased | 
| Examples | Random Forest, ExtraTrees | AdaBoost, GBDT, XGBoost, LightGBM, CatBoost |

**The framing that ties it together:** the naive reading of the trade-off says you
must pick a point on the U-curve. Ensembles refuse the premise — they start from
one extreme and attack the offending term directly. Bagging takes low-bias,
high-variance learners and averages the variance away. Boosting takes low-variance,
high-bias learners and additively drives the bias down.
---

## 3. Bagging (Bootstrap Aggregation)

### Why bootstrapping is legitimate (the one-line justification)
Ideally we'd draw k *fresh* datasets from the true distribution P and average the
models — that's what kills variance. We can't; we only have D. So we use the
**empirical distribution P_D as a proxy for P**, and sampling with replacement
from D *is* sampling from P_D. As |D| grows, P_D → P, so the proxy gets better.

That's the whole idea, and it's also the honest caveat: bootstrap samples are
**not independent** (they all come from the same D), which is why the clean 1/k
variance reduction below doesn't fully materialize.

### Procedure
1. From training data D (n points), draw k **bootstrap samples** D1, …, Dk —
   each of size n, **sampled with replacement**
2. Train one model Mi per sample (in parallel)
3. **Aggregate:**
   - Regression → mean (sometimes median)
   - Classification → **hard voting** (majority of predicted labels) or
     **soft voting** (average the predicted probabilities, then threshold)

> **Soft voting usually wins** and is a good thing to volunteer in an interview:
> averaging probabilities keeps each model's *confidence*, so a model that is
> 51% sure doesn't get the same say as one that's 99% sure. Hard voting throws
> that information away. (`sklearn` `VotingClassifier(voting='soft')`;
> `RandomForestClassifier.predict_proba` averages probabilities by default.)

### Why it reduces variance
Any single point influences only some of the bootstrap samples, so
adding/removing a few points changes only a few base models — the aggregate
barely moves. Averaging k models with (partially) independent errors shrinks
variance roughly like averaging noisy measurements, **without increasing bias**
(the final bias ≈ base-model bias).

Result with deep-tree bases: `low bias + high variance` bases → aggregated model
is `low bias + low variance`.

### Bagging's effect on each of the three terms (two-line proofs — worth memorizing)

Let ȳ = (1/k)·Σ yᵢ be the bagged prediction, where yᵢ is base model i's output.

| Term | Effect | Why |
|---|---|---|
| **Bayes error σ²** | **Unchanged** | We have no control over noise in the data |
| **Bias** | **Unchanged** | `E[ȳ] = E[(1/k)Σyᵢ] = E[yᵢ]` — the average has the *same expectation* as one model, so its bias is identical ⇒ **bagging cannot fix underfitting** |
| **Variance** | **Reduced** | If the yᵢ were independent: `Var(ȳ) = (1/k²)·Σ Var(yᵢ) = (1/k)·Var(yᵢ)` |

The bias line is the one people forget. It's also the reason base learners must
be **low-bias to begin with** (deep trees): bagging will not rescue a stump.

### The formula behind "diversity matters" (ties back to §2)
The 1/k above assumed **independent** models. Bootstrap samples come from the
same D, so the models are correlated. With pairwise correlation ρ and per-model
variance σ²:

```
Var(average) = ρσ² + (1 − ρ)·σ²/k
```

- ρ = 0 (fully decorrelated) → variance shrinks to **σ²/k**: more models keep helping
- ρ = 1 (identical models) → variance stays **σ²**: the ensemble does *nothing*

**Q: does correlation increase or decrease variance?** It *increases* it —
correlation is what stops bagging from working. Anything that adds variability
between models lowers ρ and therefore helps: different bootstrap samples,
feature subsampling, different algorithms (tree + NN + logistic regression), or
even the same algorithm with different configurations (varying NN depth/width).

This is the precise answer to "why must base models be diverse?" — the second
term vanishes as k grows, so **ρσ² is the floor**. It's also exactly why Random
Forest adds feature subsampling on top of bootstrapping: bootstrapping alone
leaves the trees correlated (they all pick the same dominant split feature), so
RF attacks ρ directly.

### Bootstrap math (interview favorite)
Probability a given point is NOT picked in one draw = (1 − 1/n).
Not picked in n draws = (1 − 1/n)ⁿ → **1/e ≈ 36.8%** as n grows.

- Each bootstrap sample contains ≈ **63.2%** unique points ("in-bag")
- ≈ **36.8%** are left out → **Out-of-Bag (OOB) points**, usable as a free
  validation set for that model (see Random Forest notes)

### When to use bagging — and when not to

| ✅ Use it when | ❌ Avoid / think twice when |
|---|---|
| Base model is **high variance** (deep trees, k=1 KNN) | You need an **interpretable** model — 100 trees is a black box |
| Data is **noisy** — averaging smooths noise out | The true relationship is **linear** — bagging a linear model barely helps (see below) |
| Dataset is **large and complex** | **Limited compute/latency** — k× the training and inference cost |
| You want a strong baseline with **little tuning** | **Small datasets** — bootstrap samples overlap heavily, ρ stays high |
| | **Imbalanced classes** — bootstrap samples can miss the minority class entirely |

**Mitigations:** interpretability → feature importance, PDP, SHAP after the fact;
compute → parallelize (`n_jobs=-1`); imbalance → stratified bootstrap sampling,
`class_weight='balanced'`, or `BalancedRandomForestClassifier`.

> ⚠️ **Why bagging a linear model does almost nothing** — good follow-up to have
> ready. Linear regression is already a *low-variance, high-bias* model, and
> bagging only attacks variance. There's a second reason: the average of linear
> fits is itself a linear fit close to the fit on the full data, so the ensemble
> ≈ the single model. Bagging needs an **unstable** base learner to have anything
> to average away.
>
> ⚠️ One slide suggests fixing small datasets by "increasing the size of bootstrap
> samples to enhance diversity." Treat this with suspicion: *larger* samples
> overlap **more**, which *raises* ρ and reduces diversity. Standard practice is
> sample size = n; if you want more diversity on small data, lower `max_features`
> or use ExtraTrees.

---

## 4. Stacking

### Idea
Train **heterogeneous** base learners (e.g. logistic regression + SVM + NN + tree)
in parallel, then train a **meta-model** (often logistic regression) whose inputs
are the base models' predictions:

```
Level 0:  h1(x), h2(x), …, hm(x)        (diverse base models)
Level 1:  meta(h1(x), h2(x), …, hm(x))  → final prediction
```

Using **predicted probabilities** (not hard labels) as meta-features usually
works better — the meta-model sees confidence, not just votes.

### Avoiding leakage — Stacking vs Blending (get this right!)
The meta-model must be trained on predictions for data the base models did NOT
train on, or it just learns to trust overfit outputs.

- **Stacking (k-fold / out-of-fold):** split training data into k folds; for each
  fold, train base models on the other k−1 folds and predict the held-out fold.
  Concatenating these **out-of-fold (OOF) predictions** covers the whole training
  set → meta-model trains on all of it. More data-efficient, more robust.
- **Blending (holdout):** simpler variant — train base models on e.g. 80% of the
  data, predict on the remaining 20% holdout, and train the meta-model only on
  that holdout's predictions. Faster but the meta-model sees less data.

> ⚠️ Common notes error: these two definitions often get swapped.
> **k-fold OOF = stacking; single holdout = blending.**

### Practical notes
- Available in **sklearn** as `StackingClassifier` / `StackingRegressor`
  (since v0.22); `mlxtend` also provides `StackingClassifier`
  (older resources saying "not in sklearn" are outdated)
- Dominant in **Kaggle competitions** (squeezes the last % of accuracy)
- In production the cost is high: many models to train, serve, and maintain —
  latency and maintenance often outweigh the small accuracy gain

### Stacking vs Bagging/Boosting
- Bagging & boosting use **homogeneous** base learners (all trees);
  stacking uses **heterogeneous** learners
- Bagging/boosting aggregate by vote/sum; stacking **learns** the combination

---

## 5. Cascading 🔍 *(rarely a named interview topic — the concept survives in
system-design questions like "design a fraud-detection pipeline")*

Used when the **cost of a mistake is very high** (credit-card fraud, medical
screening). Chain models with confidence gates: M1 handles the easy ~99%
(e.g. predict "not fraud" only if P > 0.99); uncertain cases pass to M2 — trained
only on points M1 was unsure about — then M3, …, and finally a **human** decides.
Cheap models handle most traffic; expensive stages see only hard cases.

Vs stacking: stacking sends **every** query through all models and learns to
combine them; cascading gates queries **sequentially by confidence**, so most
never reach later models. (🔍 Trivia: Viola–Jones face detection is a cascade of
AdaBoost classifiers — CV-role material only.)

---

## 6. Choosing an Ensemble — quick guide

| Situation | Use |
|---|---|
| Single tree overfits (high variance) | **Bagging / Random Forest** |
| Model underfits, need accuracy (bias problem) | **Boosting (GBDT/XGBoost)** |
| Several strong-but-different models available; competition setting | **Stacking** |
| Mistakes are very costly; need high-confidence decisions | **Cascading** |
| Need parallel/fast training | Bagging (boosting stages are sequential) |
| Noisy / mislabeled data | **Bagging** (boosting chases the noise) |
| Small dataset | Neither — a single regularized model or simple bagging; boosting overfits fast |
| Truly linear relationship | Neither — linear/ridge/LASSO |
| Interpretability is a hard requirement | Single tree or linear model; if you must ensemble, add PDP/SHAP |
| Low tuning budget | **Random Forest** (good defaults); GBDT needs real tuning |

---

## 7. Frequently Asked Interview Questions

1. Write the bias–variance decomposition. Which term does bagging reduce and
   which does boosting reduce? What base learners does each therefore prefer?
1b. **Derive it.** Where does the cross term go, and why is it zero?
1c. **Why do we do bias–variance decomposition at all?** (Answer: to debug a high
   test error — it tells you *which* fix to apply.)
1d. Your model has 2% training error and 15% test error. Diagnose it. Now: 14%
   training and 15% test — diagnose that. (Variance, then bias.)
1e. KNN with k = 1 vs. large k — high bias or high variance, and why?
1f. What is irreducible error, and how do you know when you've hit it?
1g. Is "reducing bias always increases variance" true? (Rule of thumb; bagging
   and more data are counterexamples.)
1h. Does the decomposition hold for classification with 0–1 loss? (Not additively
   — it's derived for squared loss.)
2. Why does bagging reduce variance without (much) affecting bias?
2b. Write Var(average) in terms of ρ and k. What happens at ρ = 1? Does
   correlation between models raise or lower the ensemble's variance?
2c. Prove in one line that bagging leaves bias unchanged. What does that imply
   about which base learners you should bag?
2d. Why is bootstrapping a valid substitute for drawing fresh datasets?
2e. Hard voting vs soft voting — which is better and why?
2f. Why does bagging a linear regression barely help?
2g. When would you *avoid* bagging altogether?
3. Prove ≈63.2% of points appear in a bootstrap sample. What are OOB points and
   what are they used for?
4. Why must base models in an ensemble be diverse/decorrelated? What happens if
   they are identical?
5. Bagging vs boosting — parallel vs sequential, and why.
6. Explain stacking. Why must the meta-model be trained on out-of-fold
   predictions (what leakage occurs otherwise)?
7. Stacking vs blending — precise difference.
8. Why use predicted probabilities instead of hard labels as meta-features?
9. Why is stacking common in Kaggle but rare in production?
10. Describe cascading. When is it preferred, and how does it control cost/latency?
11. 🔍 (Rare/CV roles) How does Viola–Jones face detection use a cascade?
12. (System-design framing) Design an ensemble for credit-card fraud detection
    with a human-in-the-loop — this is how cascading actually gets asked.
