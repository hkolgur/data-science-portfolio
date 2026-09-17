# Statistics for Data Science Interviews

### Sampling Distributions · Hypothesis Tests (Z, t, χ², F) · Non-Parametric Tests · A/B Testing · Anomaly Detection

> **How to use these notes**
> - **Part 1** is the big picture: which test to use and why. Read it first and review it right before an interview.
> - Every later section starts with a short **concept summary**, followed by collapsible **▶ Deep dive** blocks with the worked manual example and Python code. Click to expand.
> - All numeric results were verified with `scipy` (1.17) / `scikit-learn` (1.8).

---

## Table of Contents

**Part 1 — The Big Picture**
1. [The Inference Landscape](#1-the-inference-landscape)
2. [Which Test Do I Use?](#2-which-test-do-i-use)
3. [Parametric ↔ Non-Parametric Equivalents](#3-parametric--non-parametric-equivalents)
4. [Master Cheat Sheet](#4-master-cheat-sheet)

**Part 2 — Foundations**

5. [Core Vocabulary](#5-core-vocabulary)
6. [Point Estimates vs. Sampling Distributions](#6-point-estimates-vs-sampling-distributions)
7. [Central Limit Theorem & Standard Error](#7-central-limit-theorem--standard-error)
8. [Hypothesis Testing Framework](#8-hypothesis-testing-framework)
9. [Confidence Intervals](#9-confidence-intervals)
10. [How the Distributions Are Related](#10-how-the-distributions-are-related)

**Part 3 — Parametric Tests**

11. [Z-Test](#11-z-test)
12. [t-Test / t-Distribution](#12-t-test--t-distribution)
13. [Chi-Squared (χ²) Tests](#13-chi-squared-tests)
14. [F-Distribution & ANOVA](#14-f-distribution--anova)

**Part 4 — Non-Parametric Tests**

15. [When and Why to Go Non-Parametric](#15-when-and-why-to-go-non-parametric)
16. [Mann-Whitney U Test](#16-mann-whitney-u-test)
17. [Wilcoxon Signed-Rank Test](#17-wilcoxon-signed-rank-test)
18. [Kruskal-Wallis Test](#18-kruskal-wallis-test)
19. [Other Non-Parametric Tools](#19-other-non-parametric-tools)

**Part 5 — Applied Topics Interviewers Love**

20. [Checking Assumptions](#20-checking-assumptions)
21. [Power, Sample Size & Effect Size](#21-power-sample-size--effect-size)
22. [Multiple Testing Corrections](#22-multiple-testing-corrections)
23. [A/B Testing End-to-End](#23-ab-testing-end-to-end)
24. [Bootstrap & Permutation Tests](#24-bootstrap--permutation-tests)
25. [Correlation Tests](#25-correlation-tests)

**Part 6 — Anomaly Detection**

26. [Anomaly Detection Overview](#26-anomaly-detection-overview)
27. [Z-Score & Modified Z-Score](#27-z-score--modified-z-score)
28. [IQR Method](#28-iqr-method)
29. [Local Outlier Factor (LOF)](#29-local-outlier-factor-lof)
30. [One-Class SVM](#30-one-class-svm)
31. [Isolation Forest & Comparison](#31-isolation-forest--comparison)

**Part 7 — Interview Prep**

32. [Practice Questions](#32-practice-questions)
33. [Common Mistakes](#33-common-mistakes)

---

# Part 1 — The Big Picture

## 1. The Inference Landscape

**Statistical inference** = using a **sample** to draw conclusions about a **population**.

```
Population (unknown parameters: μ, σ, p)
        │  random sampling
        ▼
Sample ──► statistic (x̄, s, p̂)  = POINT ESTIMATE
        │
        │  "How much would this change with another sample?"
        ▼
SAMPLING DISTRIBUTION of the statistic (Z, t, χ², F ...)
        │
        ├──► Confidence interval   (range of plausible values)
        └──► Hypothesis test       (is the effect real or noise?)
```

Every test in these notes follows the same recipe:

$$\text{test statistic} = \frac{\text{signal (observed effect)}}{\text{noise (expected variability)}}$$

Then compare that statistic to its sampling distribution under H₀ to get a p-value.

| Family | Assumes a distribution shape? | Works on | Examples |
| :--- | :--- | :--- | :--- |
| **Parametric** | Yes (usually normality) | Means, variances | Z, t, ANOVA (F), Pearson r |
| **Non-parametric** | No (distribution-free) | Ranks, medians, counts | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Spearman |
| **Resampling** | No — simulates the sampling distribution | Any statistic | Bootstrap, permutation test |
| **Anomaly detection** | Varies | Individual observations | Z-score, IQR, LOF, One-Class SVM, Isolation Forest |

---

## 2. Which Test Do I Use?

```
START: What are you comparing?
│
├── CATEGORICAL counts
│     ├── 1 variable vs. expected proportions ─────────► χ² goodness of fit
│     ├── 2 variables — associated? ───────────────────► χ² test of independence
│     │                                                   (expected count < 5 → Fisher's exact)
│     ├── 2 proportions (A/B conversion rate) ─────────► two-proportion Z-test
│     └── paired yes/no (before/after) ────────────────► McNemar's test
│
├── NUMERIC data
│     │
│     ├── 1 group vs. a known value
│     │     ├── σ known (or n very large) ─────────────► one-sample Z-test
│     │     ├── σ unknown, ~normal ────────────────────► one-sample t-test
│     │     └── not normal, small n ───────────────────► Wilcoxon signed-rank (vs. median)
│     │
│     ├── 2 INDEPENDENT groups
│     │     ├── ~normal ───────────────────────────────► Welch two-sample t-test
│     │     └── not normal / ordinal / outliers ───────► Mann-Whitney U
│     │
│     ├── 2 PAIRED groups (before/after, same subject)
│     │     ├── differences ~normal ───────────────────► paired t-test
│     │     └── not normal / ordinal ──────────────────► Wilcoxon signed-rank
│     │
│     ├── 3+ INDEPENDENT groups
│     │     ├── normal + equal variances ──────────────► one-way ANOVA (F) → Tukey HSD
│     │     ├── normal, unequal variances ─────────────► Welch's ANOVA
│     │     └── not normal / ordinal ──────────────────► Kruskal-Wallis → Dunn's test
│     │
│     ├── 3+ REPEATED measures on same subjects
│     │     ├── normal ────────────────────────────────► repeated-measures ANOVA
│     │     └── not normal ────────────────────────────► Friedman test
│     │
│     ├── Variance / spread
│     │     ├── one variance vs. a value ──────────────► χ² variance test
│     │     └── 2+ variances equal? ───────────────────► F-test (normal) / Levene (robust)
│     │
│     └── Relationship between 2 numeric variables
│           ├── linear, ~normal ───────────────────────► Pearson r
│           └── monotonic / ranks / outliers ──────────► Spearman ρ (or Kendall τ)
│
├── Unusual INDIVIDUAL observations? ──────────────────► Anomaly detection (Part 6)
│
└── Weird statistic (median, ratio, percentile)? ───────► Bootstrap / permutation test
```

---

## 3. Parametric ↔ Non-Parametric Equivalents

| Situation | Parametric test | Non-parametric equivalent | Non-parametric H₀ (roughly) |
| :--- | :--- | :--- | :--- |
| 1 sample vs. a value | One-sample t / Z | **Wilcoxon signed-rank** (or sign test) | Median = value |
| 2 independent groups | Independent (Welch) t | **Mann-Whitney U** (Wilcoxon rank-sum) | Same distribution / P(X > Y) = 0.5 |
| 2 paired groups | Paired t | **Wilcoxon signed-rank** | Median difference = 0 |
| 3+ independent groups | One-way ANOVA | **Kruskal-Wallis H** | All groups same distribution |
| 3+ repeated measures | Repeated-measures ANOVA | **Friedman** | Same distribution across conditions |
| Correlation | Pearson r | **Spearman ρ / Kendall τ** | No monotonic association |
| Post-hoc after 3+ groups | Tukey HSD | **Dunn's test** | Pairwise equality |

**What happens as sample size grows?**
- Each rank statistic's sampling distribution converges to a familiar parametric one: **U and W → Normal (Z)**, **Kruskal-Wallis H → χ²(k − 1)**, **Friedman → χ²(k − 1)**. Software switches from exact tables to these approximations for large n.
- Meanwhile, the **CLT** makes the t-test and ANOVA robust to non-normality, so the parametric tests become safe again with large samples (outliers can still be a problem).
- Under normal data, Mann-Whitney is about **95% as efficient** as the t-test (asymptotic relative efficiency = 3/π ≈ 0.955), so you lose little by using it; with heavy tails it can be *more* powerful than t.

---

## 4. Master Cheat Sheet

| Test | Use when | Statistic | df / reference distribution | Effect size | Python (`scipy.stats` unless noted) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Z-test (mean)** | σ known, or n large | (x̄ − μ₀) / (σ/√n) | N(0, 1) | Cohen's d | manual / `statsmodels ztest` |
| **Z-test (proportions)** | A/B conversion rates | (p̂₁ − p̂₂) / SE | N(0, 1) | Diff. in proportions, lift | `statsmodels proportions_ztest` |
| **t-test** | σ unknown, ~normal | (x̄ − μ₀) / (s/√n) | t, n − 1 | Cohen's d | `ttest_1samp`, `ttest_ind`, `ttest_rel` |
| **χ² independence** | 2 categorical variables | Σ (O − E)² / E | χ², (r−1)(c−1) | Cramér's V | `chi2_contingency` |
| **χ² goodness of fit** | Counts vs. expected | Σ (O − E)² / E | χ², k − 1 | Cohen's w | `chisquare` |
| **ANOVA** | 3+ means | MS_between / MS_within | F, (k−1, N−k) | η² | `f_oneway`, `tukey_hsd` |
| **Mann-Whitney U** | 2 indep., non-normal | min(U₁, U₂) | U table / Normal | rank-biserial r | `mannwhitneyu` |
| **Wilcoxon signed-rank** | Paired, non-normal | min(W⁺, W⁻) | W table / Normal | matched-pairs r | `wilcoxon` |
| **Kruskal-Wallis** | 3+ indep., non-normal | H | χ², k − 1 | ε² | `kruskal` |
| **Friedman** | 3+ repeated, non-normal | Q | χ², k − 1 | Kendall's W | `friedmanchisquare` |
| **Levene** | Equal variances? | W | F | — | `levene` |
| **Shapiro-Wilk** | Normality? | W | — | — | `shapiro` |

### One-Line Summaries
- **Z:** means or proportions when σ is known or the sample is large.
- **t:** means when σ is unknown (estimated by s), especially small samples.
- **χ²:** categorical counts (independence, goodness of fit) and a single variance.
- **F:** ratio of variances — ANOVA (3+ means), equal-variance checks, regression overall fit.
- **Mann-Whitney U:** replaces the unpaired t-test when its assumptions fail.
- **Wilcoxon signed-rank:** replaces the paired t-test when its assumptions fail.
- **Kruskal-Wallis:** replaces one-way ANOVA when its assumptions fail.
- **Anomaly detection:** find individual observations far from the rest (fraud, influential outliers).

---

# Part 2 — Foundations

## 5. Core Vocabulary

| Term | Meaning | Population → Sample |
| :--- | :--- | :--- |
| **Population** | Every individual we care about | — |
| **Sample** | The subset we actually measure | — |
| **Parameter** | Fixed, usually unknown number describing the population | μ, σ, σ², p |
| **Statistic** | Number calculated from a sample | x̄, s, s², p̂ |
| **Estimator** | The rule/formula used to compute a statistic | x̄ = Σx / n |
| **Degrees of freedom (df)** | Number of values free to vary after constraints | df |
| **Parametric test** | Assumes a specific population distribution | — |
| **Non-parametric test** | Makes no (or weak) distribution assumptions | — |

<details>
<summary><b>▶ Deep dive: why divide by n − 1?</b></summary>

Once you know x̄, only `n − 1` deviations are free; the last one is forced so the deviations sum to 0. Dividing by `n − 1` (Bessel's correction) makes s² an **unbiased** estimator of σ². Dividing by `n` systematically underestimates the variance.

```python
import numpy as np
data = [4, 8, 6, 5, 7]
np.var(data)          # ÷ n     → population formula (biased for samples)
np.var(data, ddof=1)  # ÷ n − 1 → sample variance s²
# pandas .var() and .std() use ddof=1 by default; numpy uses ddof=0 — a classic gotcha
```

</details>

---

## 6. Point Estimates vs. Sampling Distributions

- A **point estimate** is one number from one sample (x̄, s², p̂) — the best single guess of a parameter. It says **nothing about its own accuracy**.
- A **sampling distribution** is the distribution of that statistic across **all possible samples of size n**. Its spread is the **standard error**.

**Why we need sampling distributions:**
1. **Quantify uncertainty** — standard error = how much the estimate moves from sample to sample.
2. **Hypothesis testing** — p-values are areas under the sampling distribution assuming H₀ is true.
3. **Confidence intervals** — estimate ± critical value × SE.
4. **Comparing estimators** — bias (center) and efficiency (spread).

| Feature | Point Estimate | Sampling Distribution |
| :--- | :--- | :--- |
| What is it? | One calculated number | Distribution of the statistic over all possible samples |
| Comes from | One observed sample | Theory (or simulation) of repeated sampling |
| Shows variability? | No | Yes — its SD is the standard error |
| Answers | "What's my best guess?" | "How much can I trust that guess?" |
| Example | x̄ = 170 cm | x̄ ~ approx. N(μ, σ/√n) |

**Point estimate + sampling distribution = interval estimate (confidence interval).**

<details>
<summary><b>▶ Deep dive: simulate a sampling distribution in Python</b></summary>

```
Population ──► Sample 1 ──► x̄₁ = 169.2
           ──► Sample 2 ──► x̄₂ = 171.5
           ──► Sample 3 ──► x̄₃ = 168.8      ──►  histogram of x̄ = sampling distribution
           ──►   ...    ──►   ...
```

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
population = rng.exponential(scale=10, size=100_000)   # clearly skewed

n = 30
sample_means = [rng.choice(population, n).mean() for _ in range(5_000)]

print(f"Population mean      : {population.mean():.2f}")
print(f"Mean of sample means : {np.mean(sample_means):.2f}")   # ≈ population mean (unbiased)
print(f"Theoretical SE σ/√n  : {population.std() / np.sqrt(n):.2f}")
print(f"Observed SD of x̄     : {np.std(sample_means):.2f}")    # ≈ theoretical SE

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(population, bins=60);   ax[0].set_title("Population (skewed)")
ax[1].hist(sample_means, bins=60); ax[1].set_title(f"Sampling dist. of x̄ (n={n})")
plt.show()
```

The population is skewed but the sample means form a bell curve — the CLT in action.

</details>

---

## 7. Central Limit Theorem & Standard Error

**CLT:** for independent samples of size n from a population with mean μ and finite SD σ,

$$\bar{x} \;\sim\; \text{approx. } N\!\left(\mu,\ \frac{\sigma}{\sqrt{n}}\right) \quad \text{as } n \text{ grows}$$

- Works regardless of population shape; n ≥ 30 is a common rule of thumb (heavily skewed data need more).
- If the population is normal, x̄ is exactly normal for any n.
- **Law of Large Numbers** (different idea): x̄ → μ as n → ∞. LLN says the estimate *converges*; CLT describes the *shape* of its error.

| Statistic | Standard Error |
| :--- | :--- |
| Mean (σ known) | σ / √n |
| Mean (σ unknown) | s / √n |
| Proportion | √( p̂(1 − p̂) / n ) |
| Difference of two means | √( s₁²/n₁ + s₂²/n₂ ) |
| Difference of two proportions | √( p̂₁(1−p̂₁)/n₁ + p̂₂(1−p̂₂)/n₂ ) |

- SE shrinks with √n → **halving SE needs 4× the data.**
- **SD** = spread of individual data points. **SE** = spread of an estimate.

---

## 8. Hypothesis Testing Framework

1. State **H₀** (no effect) and **H₁** (effect exists; one- or two-sided).
2. Choose **α** (commonly 0.05) *before* looking at data.
3. Check assumptions, compute the **test statistic**.
4. Get the **p-value** (or compare with the **critical value**).
5. **p ≤ α → reject H₀**; otherwise **fail to reject H₀** (never "accept/prove H₀").
6. Report the **effect size** and a **confidence interval**, not only the p-value.

| | H₀ true | H₀ false |
| :--- | :--- | :--- |
| **Reject H₀** | **Type I error** (false positive), prob. = α | Correct — **power** = 1 − β |
| **Fail to reject** | Correct | **Type II error** (false negative), prob. = β |

- **p-value:** probability of data at least this extreme **assuming H₀ is true**. It is *not* P(H₀ is true), and not the size of the effect.
- **Power:** probability of detecting a real effect. Increases with larger n, larger effect, larger α, lower variance. Typical target: 80%.
- **One-tailed vs. two-tailed:** use one-tailed only if the direction was decided *before* seeing data and the other direction truly doesn't matter.
- **Statistical vs. practical significance:** with huge n, tiny useless effects become "significant."

---

## 9. Confidence Intervals

$$\text{CI} = \text{point estimate} \pm \text{critical value} \times SE$$

| Parameter | Interval |
| :--- | :--- |
| Mean, σ known | x̄ ± z* · σ/√n |
| Mean, σ unknown | x̄ ± t*₍ₙ₋₁₎ · s/√n |
| Proportion | p̂ ± z* · √(p̂(1−p̂)/n) (Wilson interval is better for small n or p near 0/1) |
| Variance | [ (n−1)s² / χ²_upper , (n−1)s² / χ²_lower ] |
| Anything else | Bootstrap percentile interval |

Common z* values: 90% → 1.645, 95% → 1.960, 99% → 2.576.

**Correct interpretation:** if we repeated the sampling many times, ~95% of intervals built this way would contain the true parameter. It is **not** "95% probability the parameter is in this interval" (that's a Bayesian credible interval).

**Link to tests:** a 95% CI that excludes the null value ⇔ two-sided test rejects H₀ at α = 0.05.

---

## 10. How the Distributions Are Related

```
Z  (standard normal)
│
├── Z₁² + … + Z_k²                     ──►  χ²(k)
├── Z / √(χ²(k) / k)                   ──►  t(k)
└── (χ²(d₁)/d₁) / (χ²(d₂)/d₂)          ──►  F(d₁, d₂)

Special cases:  t(k) → Z as k → ∞      t(k)² = F(1, k)      Z² = χ²(1)
```

| | Z | t | χ² | F |
| :--- | :--- | :--- | :--- | :--- |
| Shape | Symmetric | Symmetric, heavier tails | Right-skewed, ≥ 0 | Right-skewed, ≥ 0 |
| Parameters | none | df | df | d₁, d₂ |
| Mean | 0 | 0 (df > 1) | df | d₂/(d₂−2) |

- A two-sample t-test and a 2-group ANOVA give the same p-value (F = t²).
- A 2×2 χ² test (no continuity correction) and a two-proportion Z-test give the same p-value (χ² = Z²).

---

# Part 3 — Parametric Tests

## 11. Z-Test

**What:** a test whose statistic follows the **standard normal** distribution under H₀.

$$Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

**When to use:**
- Population **σ is known** (rare in practice — e.g., a machine with a long, stable history), **or**
- **Large sample** (n ≥ 30) so s ≈ σ and the CLT applies (t and Z give nearly identical answers), **or**
- Testing **proportions** with enough data (n·p ≥ 10 and n·(1−p) ≥ 10) — this is the workhorse of **A/B testing**.

**Z vs. t in one sentence:** use t whenever σ is estimated from a small sample; Z is fine when σ is known or n is large.

| Variant | Statistic |
| :--- | :--- |
| One-sample mean | (x̄ − μ₀) / (σ/√n) |
| Two-sample mean | (x̄₁ − x̄₂) / √(σ₁²/n₁ + σ₂²/n₂) |
| One proportion | (p̂ − p₀) / √(p₀(1−p₀)/n) |
| Two proportions | (p̂₁ − p̂₂) / √(p̄(1−p̄)(1/n₁ + 1/n₂)), with pooled p̄ = (x₁+x₂)/(n₁+n₂) |

Critical values: two-tailed α = 0.05 → ±1.96; one-tailed α = 0.05 → 1.645.

<details>
<summary><b>▶ Deep dive: one-sample Z-test (σ known)</b></summary>

**Problem:** A bottling machine should fill 500 ml. Its long-run SD is known: σ = 5 ml. A sample of n = 36 bottles has x̄ = 498 ml. Is the machine off target? (α = 0.05, two-tailed)

- H₀: μ = 500  H₁: μ ≠ 500

1. SE = σ/√n = 5/6 = 0.833
2. Z = (498 − 500) / 0.833 = **−2.40**
3. Critical value ±1.96 → |−2.40| > 1.96
4. p-value = 2 × P(Z > 2.40) = **0.0164**
5. **Reject H₀** — the machine is under-filling.

95% CI: 498 ± 1.96 × 0.833 = [496.37, 499.63] ml (excludes 500, consistent with rejecting).

```python
import numpy as np
from scipy import stats

x_bar, mu0, sigma, n = 498, 500, 5, 36
se = sigma / np.sqrt(n)
z = (x_bar - mu0) / se
p = 2 * stats.norm.sf(abs(z))
print(f"z = {z:.2f}, p = {p:.4f}")                      # z = -2.40, p = 0.0164
print("95% CI:", x_bar - 1.96 * se, x_bar + 1.96 * se)

# With raw data (statsmodels uses the sample SD, i.e. large-sample Z):
# from statsmodels.stats.weightstats import ztest
# z, p = ztest(data, value=500)
```

</details>

<details>
<summary><b>▶ Deep dive: two-proportion Z-test (A/B test)</b></summary>

**Problem:** Control (A): 200 conversions / 2000 users = 10%. Variant (B): 260 / 2000 = 13%. Did B improve conversion? (α = 0.05, two-tailed)

- H₀: p_A = p_B  H₁: p_A ≠ p_B

1. Pooled p̄ = (200 + 260) / 4000 = 0.115
2. SE = √(0.115 × 0.885 × (1/2000 + 1/2000)) = 0.01009
3. Z = (0.13 − 0.10) / 0.01009 = **2.97**
4. p-value = **0.0029** → **reject H₀**
5. 95% CI for the difference (unpooled SE) = 0.03 ± 1.96 × 0.01008 = **[1.0, 5.0] percentage points**
6. Relative lift = 0.03 / 0.10 = **+30%**

**Why pooled SE for the test but unpooled for the CI?** The test assumes H₀ (both rates equal), so it pools; the CI doesn't assume that.

```python
import numpy as np
from scipy import stats

x = np.array([200, 260]); n = np.array([2000, 2000])
p1, p2 = x / n
p_pool = x.sum() / n.sum()
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n[0] + 1/n[1]))
z = (p2 - p1) / se_pool
p = 2 * stats.norm.sf(abs(z))
print(f"z = {z:.3f}, p = {p:.4f}")                      # z = 2.974, p = 0.0029

se_unpool = np.sqrt(p1*(1-p1)/n[0] + p2*(1-p2)/n[1])
print("95% CI:", (p2-p1) - 1.96*se_unpool, (p2-p1) + 1.96*se_unpool)

# statsmodels version
# from statsmodels.stats.proportion import proportions_ztest
# z, p = proportions_ztest(count=x, nobs=n)
```

</details>

---

## 12. t-Test / t-Distribution

**What:** Student's t is symmetric and bell-shaped like Z but with **heavier tails**, reflecting the extra uncertainty from estimating σ with s. One parameter: **df**. As df → ∞, t → Z.

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**When to use:** σ **unknown** (estimated by s), especially **small samples** (n < 30), data roughly normal.

| Variant | Use | df |
| :--- | :--- | :--- |
| One-sample | Mean vs. a value | n − 1 |
| Independent (Welch) | Two separate groups — **default choice** | Welch–Satterthwaite approx. |
| Independent (Student / pooled) | Two groups, equal variances | n₁ + n₂ − 2 |
| Paired | Same subjects measured twice | pairs − 1 |

**Assumptions:** independent observations; roughly normal data (or differences, for paired); no extreme outliers. If violated with small n → Mann-Whitney / Wilcoxon.

| df | 2 | 5 | 9 | 20 | 30 | 100 | ∞ (Z) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| t₀.₀₂₅ (95% two-tailed) | 4.303 | 2.776 | 2.262 | 2.086 | 2.042 | 1.984 | 1.960 |

*Fun fact:* developed by William Gosset ("Student") at Guinness for small-sample quality control.

<details>
<summary><b>▶ Deep dive: 95% CI for average adult height (manual + Python)</b></summary>

**Problem:** 10 adults sampled, x̄ = 170 cm, σ unknown. Build a 95% CI for the town's mean height.
> The sample SD isn't given, so we **assume s = 8 cm**.

$$CI = \bar{x} \pm t_{\alpha/2,\,df} \cdot \frac{s}{\sqrt{n}}$$

1. df = 10 − 1 = **9**
2. t₀.₀₂₅,₉ = **2.262**
3. SE = 8 / √10 = 8 / 3.162 = **2.530 cm**
4. ME = 2.262 × 2.530 = **5.72 cm**
5. CI = 170 ± 5.72 = **[164.28, 175.72] cm**

Using Z (1.96) instead would give ±4.96 cm — too narrow, because it ignores the uncertainty in s.

```python
import numpy as np
import scipy.stats as stats

n, sample_mean, sample_std, confidence = 10, 170, 8, 0.95
df = n - 1
se = sample_std / np.sqrt(n)

t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df)
me = t_crit * se
print(f"t* = {t_crit:.3f}, SE = {se:.3f}, ME = {me:.2f}")        # 2.262, 2.530, 5.72
print(f"CI: [{sample_mean - me:.2f}, {sample_mean + me:.2f}]")

ci = stats.t.interval(confidence, df, loc=sample_mean, scale=se)
print(f"scipy: [{ci[0]:.2f}, {ci[1]:.2f}]")                       # [164.28, 175.72]
```

</details>

<details>
<summary><b>▶ Deep dive: one-sample, Welch and paired t-tests in Python</b></summary>

```python
import numpy as np
import scipy.stats as stats

heights = [162, 175, 168, 180, 171, 159, 173, 166, 177, 169]
print(stats.ttest_1samp(heights, popmean=165))            # one-sample

town_a = [170, 172, 168, 175, 169]
town_b = [165, 163, 168, 160, 166]
print(stats.ttest_ind(town_a, town_b, equal_var=False))   # Welch (safe default)

before = [80, 85, 78, 90, 88]
after  = [78, 83, 77, 86, 85]
print(stats.ttest_rel(before, after))                     # paired

# Cohen's d (effect size) for two independent groups
a, b = np.array(town_a), np.array(town_b)
sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
print("Cohen's d:", (a.mean() - b.mean()) / sp)   # 0.2 small, 0.5 medium, 0.8 large

# Visual: t approaches Z
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 400)
plt.plot(x, stats.norm.pdf(x), label="Z")
for d in (1, 5, 30):
    plt.plot(x, stats.t.pdf(x, d), label=f"t, df={d}")
plt.legend(); plt.show()
```

**Why Welch by default?** It stays accurate when variances differ and loses almost nothing when they're equal.

</details>

---

## 13. Chi-Squared Tests

**What:** sum of k squared independent standard normals → χ²(k). Non-negative, right-skewed, mean = df. Count-based tests are **right-tailed** (big χ² = big mismatch).

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

**When to use:**
1. **Independence** — are two categorical variables related? (contingency table)
2. **Goodness of fit** — do counts match an expected distribution?
3. **Homogeneity** — do several populations share the same category proportions?
4. **Single variance** — CI or test for σ².

**Assumptions:** raw **counts** (not percentages); independent observations; **expected** count ≥ 5 in most cells (else Fisher's exact test or merge categories).

<details>
<summary><b>▶ Deep dive: gender vs. political party (test of independence)</b></summary>

H₀: gender and party are independent. α = 0.05.

| Observed | Republican | Democrat | Independent | **Total** |
| :--- | :---: | :---: | :---: | :---: |
| Male | 100 | 70 | 30 | **200** |
| Female | 140 | 60 | 20 | **220** |
| **Total** | **240** | **130** | **50** | **N = 420** |

> ⚠️ Grand total is **420** (not 440, as in an earlier draft).

**Step 1 — Expected counts:** E = (row total × column total) / N
- Male/Rep: 200 × 240 / 420 = 114.29
- Female/Ind: 220 × 50 / 420 = 26.19

| Expected | Rep | Dem | Ind |
| :--- | :---: | :---: | :---: |
| Male | 114.29 | 61.90 | 23.81 |
| Female | 125.71 | 68.10 | 26.19 |

**Step 2 — Contributions (O − E)² / E**

| Cell | O | E | Contribution |
| :--- | :---: | :---: | :---: |
| Male, Rep | 100 | 114.29 | 1.786 |
| Male, Dem | 70 | 61.90 | 1.059 |
| Male, Ind | 30 | 23.81 | 1.610 |
| Female, Rep | 140 | 125.71 | 1.623 |
| Female, Dem | 60 | 68.10 | 0.962 |
| Female, Ind | 20 | 26.19 | 1.463 |
| **Total** | | | **χ² = 8.503** |

**Step 3 — df** = (2 − 1)(3 − 1) = **2**

**Step 4 — Decision:** critical χ²₀.₀₅,₂ = 5.991. 8.503 > 5.991 → **reject H₀** (p = 0.014).

**Step 5 — Effect size (Cramér's V):** √(χ² / (N · (min(r, c) − 1))) = √(8.503 / 420) = **0.142 → weak** association.

```python
import numpy as np
import scipy.stats as stats

observed = np.array([[100, 70, 30],
                     [140, 60, 20]])
chi2, p, df, expected = stats.chi2_contingency(observed)
print(f"chi2 = {chi2:.3f}, p = {p:.4f}, df = {df}")      # 8.503, 0.0142, 2
print(expected.round(2))
print(((observed - expected)**2 / expected).round(3))   # cell contributions
print("critical:", stats.chi2.ppf(0.95, df))            # 5.991
print("Cramér's V:", np.sqrt(chi2 / (observed.sum() * (min(observed.shape) - 1))))

# From raw data:  table = pd.crosstab(df_raw["gender"], df_raw["party"])
# Small 2x2 counts:  stats.fisher_exact(table_2x2)
```
Note: `chi2_contingency` applies Yates' continuity correction only for 2×2 tables.

</details>

<details>
<summary><b>▶ Deep dive: goodness of fit & sample ratio mismatch</b></summary>

**Is a die fair?** 60 rolls, expected 10 per face, df = 6 − 1 = 5.

```python
import scipy.stats as stats
observed = [8, 12, 9, 11, 6, 14]
print(stats.chisquare(observed, f_exp=[10]*6))
```

**Interview favorite — Sample Ratio Mismatch (SRM):** in an A/B test designed as 50/50, you observe 50,600 vs. 49,400 users. A goodness-of-fit test checks whether the split is plausible; a tiny p-value means the assignment/logging is broken and the experiment results can't be trusted.

```python
from scipy import stats
print(stats.chisquare([50_600, 49_400], f_exp=[50_000, 50_000]))   # p ≈ 0.00015 → SRM!
```

</details>

<details>
<summary><b>▶ Deep dive: confidence interval for a variance</b></summary>

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$$

Height example (n = 10, s = 8): (n − 1)s² = 576; χ² values for df = 9 are 19.023 (upper) and 2.700 (lower).
- σ² CI = [576/19.023, 576/2.700] = **[30.28, 213.30]**
- σ CI = **[5.50, 14.60] cm** — not symmetric, because χ² is skewed.

```python
import scipy.stats as stats
n, s = 10, 8; df = n - 1
lo = df * s**2 / stats.chi2.ppf(0.975, df)
hi = df * s**2 / stats.chi2.ppf(0.025, df)
print(f"σ² CI: [{lo:.2f}, {hi:.2f}]  σ CI: [{lo**0.5:.2f}, {hi**0.5:.2f}]")
```

</details>

---

## 14. F-Distribution & ANOVA

**What:** ratio of two independent (χ²/df) variables. Non-negative, right-skewed, two df (numerator d₁, denominator d₂). Under H₀ the ratio is near **1**.

**When to use:**
1. **One-way ANOVA** — compare **3+ group means** in one test.
2. **Equal variances?** — F-test (normal data) or Levene (robust).
3. **Regression** — overall F-test: does the model explain anything?

$$F = \frac{\text{between-group variance}}{\text{within-group variance}} = \frac{MS_{between}}{MS_{within}}$$

- Between = signal (how far group means are from the grand mean). Within = noise.
- **Why not many t-tests?** 4 groups → 6 pairwise tests → P(≥1 false positive) ≈ 1 − 0.95⁶ ≈ **26%**.
- ANOVA only says *some* mean differs → follow with **Tukey HSD** to find which.
- **Assumptions:** independence, normal residuals, equal variances (if violated → Welch's ANOVA; if non-normal → Kruskal-Wallis).

| Source | SS | df | MS | F |
| :--- | :--- | :--- | :--- | :--- |
| Between | Σ nᵢ(x̄ᵢ − x̄)² | k − 1 | SSB/(k−1) | MSB/MSW |
| Within | ΣΣ (x − x̄ᵢ)² | N − k | SSW/(N−k) | |
| Total | SSB + SSW | N − 1 | | |

<details>
<summary><b>▶ Deep dive: F from given mean squares (14.540 / 4.402)</b></summary>

MS_between = 14.540 (df = 3), MS_within = 4.402 (df = 36) → k = 4 groups, N = 40.

1. **F = 14.540 / 4.402 = 3.303** — between-group variance is 3.3× the within-group variance.
2. df₁ = 3, df₂ = 36.
3. Critical F₀.₀₅,₃,₃₆ = 2.866; p = **0.031** < 0.05 → **reject H₀**: not all means are equal.
4. Next step: Tukey HSD.

```python
import scipy.stats as stats
f_stat = 14.540 / 4.402
print(f"F = {f_stat:.3f}")                            # 3.303
print(f"p = {stats.f.sf(f_stat, 3, 36):.4f}")         # 0.0311
print(f"critical = {stats.f.ppf(0.95, 3, 36):.3f}")   # 2.866
```

</details>

<details>
<summary><b>▶ Deep dive: full ANOVA from raw data + Tukey</b></summary>

| Method | Scores | Mean |
| :--- | :--- | :---: |
| A | 23, 25, 21, 24, 22 | 23 |
| B | 27, 29, 26, 28, 30 | 28 |
| C | 22, 24, 23, 25, 21 | 23 |

Grand mean = 24.67
1. SSB = 5(23 − 24.67)² + 5(28 − 24.67)² + 5(23 − 24.67)² = **83.33**
2. SSW = 10 + 10 + 10 = **30**
3. df = 2 and 12 → MSB = 41.67, MSW = 2.5
4. **F = 16.67** vs. critical 3.885 → reject (p ≈ 0.0003)
5. **η² = SSB / SST = 83.33 / 113.33 = 0.735** → 73.5% of variation explained by method

```python
import numpy as np
import scipy.stats as stats

a = [23, 25, 21, 24, 22]; b = [27, 29, 26, 28, 30]; c = [22, 24, 23, 25, 21]
groups = [a, b, c]

print(stats.levene(*groups))              # equal-variance check
print(stats.f_oneway(*groups))            # F = 16.67, p = 0.00034

all_vals = np.concatenate(groups); grand = all_vals.mean()
ssb = sum(len(g) * (np.mean(g) - grand)**2 for g in groups)
ssw = sum(((np.array(g) - np.mean(g))**2).sum() for g in groups)
k, N = len(groups), len(all_vals)
print("F manual:", (ssb/(k-1)) / (ssw/(N-k)), "eta²:", ssb/(ssb+ssw))

print(stats.tukey_hsd(a, b, c))           # which pairs differ

# statsmodels ANOVA table:
# from statsmodels.formula.api import ols; import statsmodels.api as sm
# model = ols("score ~ C(method)", data=df).fit(); sm.stats.anova_lm(model, typ=2)
```

</details>

<details>
<summary><b>▶ Deep dive: F-test for two variances</b></summary>

```python
import numpy as np
import scipy.stats as stats

x = np.array([12.1, 11.8, 12.5, 12.0, 11.6, 12.4])
y = np.array([12.9, 11.2, 13.5, 10.8, 12.7, 11.5])
s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
F = max(s1, s2) / min(s1, s2)                 # larger variance on top
d1 = (len(x) if s1 >= s2 else len(y)) - 1
d2 = (len(y) if s1 >= s2 else len(x)) - 1
print(F, min(1.0, 2 * stats.f.sf(F, d1, d2)))

print(stats.levene(x, y))   # preferred in practice — robust to non-normality
```

</details>

---

# Part 4 — Non-Parametric Tests

## 15. When and Why to Go Non-Parametric

**What:** tests that don't assume a specific population distribution (usually normality). Most work on **ranks** instead of raw values, so they're robust to outliers and skew.

**Use them when:**
- The data are clearly **not normal** and the **sample is small** (CLT can't rescue you).
- The data are **ordinal** (ratings 1–5, pain scales, rankings) — means aren't meaningful.
- There are **extreme outliers** you can't justify removing.
- **Unequal variances / heavy tails** break t-test or ANOVA assumptions.

**Trade-offs:**
- Slightly **less power** than parametric tests *when parametric assumptions hold* (≈ 95% efficiency for Mann-Whitney vs. t).
- Test **distributions / ranks**, not means. Interpreting them as "medians differ" requires the groups' distributions to have the **same shape**.
- Confidence intervals and effect sizes are less familiar to stakeholders.

**As sample size grows:**
- The exact rank distributions become approximately **Normal** (U, W) or **χ²** (H, Friedman) — software uses these large-sample approximations.
- The parametric tests become robust thanks to the CLT, so for large n the choice matters less; pick based on the **question** (means vs. typical values) and on **outliers**.

**Ranking basics (used by all three tests below):**
1. Sort all relevant values from smallest to largest.
2. Assign ranks 1, 2, 3, …
3. **Ties** get the **average** of the ranks they would occupy (e.g., two values tied for 4th and 5th both get 4.5).

---

## 16. Mann-Whitney U Test

*(also called the Wilcoxon rank-sum test)*

**What:** non-parametric alternative to the **independent (unpaired) two-sample t-test**. Tests whether values in one group tend to be larger than in the other — formally H₀: P(X > Y) = 0.5.

**Assumptions:** two independent groups; ordinal or continuous data; independent observations.

**Steps:**
1. Combine both groups into one dataset.
2. Rank all observations together (smallest = 1); average the ranks of ties.
3. Sum the ranks for each group: R₁ and R₂.
4. Compute
$$U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 \qquad U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2$$
   Check: U₁ + U₂ = n₁n₂.
5. **U = min(U₁, U₂)**. The smaller U means less overlap between the groups.
6. **Reject H₀ if U ≤ critical value** from the Mann-Whitney table (note the direction: *small* U is significant, unlike t or F).
7. For large samples (roughly n > 20 per group), use the normal approximation:
$$z = \frac{U - n_1 n_2 / 2}{\sqrt{n_1 n_2 (n_1 + n_2 + 1)/12}}$$

<details>
<summary><b>▶ Deep dive: viral load, new therapy vs. untreated</b></summary>

**Scenario:** A small randomized trial compares viral load (copies/ml) for 7 patients on a new therapy vs. 7 untreated patients. Viral loads are heavily skewed, so a t-test is inappropriate. α = 0.05, two-tailed.
*(Illustrative numbers built to match the structure of the lecture example: U₁ = 41, U₂ = 8.)*

| Treated | 340 | 470 | 960 | 1000 | 1200 | 3100 | 4100 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Untreated** | 1500 | 2100 | 2800 | 3500 | 3900 | 5000 | 7400 |

**Step 1–2 — Combined ranking**

| Value | 340 | 470 | 960 | 1000 | 1200 | 1500 | 2100 | 2800 | 3100 | 3500 | 3900 | 4100 | 5000 | 7400 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Group | T | T | T | T | T | U | U | U | T | U | U | T | U | U |
| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |

**Step 3 — Rank sums:** R₁ (treated) = 1+2+3+4+5+9+12 = **36**; R₂ (untreated) = 6+7+8+10+11+13+14 = **69**. Check: 36 + 69 = 105 = 14·15/2 ✓

**Step 4 — U values** (n₁ = n₂ = 7, n₁n₂ = 49, n(n+1)/2 = 28):
- U₁ = 49 + 28 − 36 = **41**
- U₂ = 49 + 28 − 69 = **8**
- Check: 41 + 8 = 49 ✓

**Step 5 — U = min(41, 8) = 8**

**Step 6 — Decision:** critical value for n₁ = n₂ = 7, α = 0.05 two-tailed is **8**. Since U = 8 ≤ 8, **reject H₀**. Exact p = 0.038.

**Conclusion:** there is evidence of a difference in viral load between treated and untreated patients (treated patients tend to have lower viral load).

**Normal approximation (for illustration):** mean = 24.5, SD = √(49·15/12) = 7.83 → z = (8 − 24.5)/7.83 = −2.11 → p ≈ 0.035.

**Effect size (rank-biserial correlation):** r = 1 − 2U/(n₁n₂) = 1 − 16/49 = **0.67** → large.

```python
import numpy as np
from scipy import stats

treated   = [340, 470, 960, 1000, 1200, 3100, 4100]
untreated = [1500, 2100, 2800, 3500, 3900, 5000, 7400]

# Manual
ranks = stats.rankdata(treated + untreated)          # handles ties by averaging
n1, n2 = len(treated), len(untreated)
R1, R2 = ranks[:n1].sum(), ranks[n1:].sum()
U1 = n1*n2 + n1*(n1+1)/2 - R1
U2 = n1*n2 + n2*(n2+1)/2 - R2
print(R1, R2, U1, U2, "U =", min(U1, U2))            # 36 69 41 8 U = 8

# scipy (reports U for the FIRST sample = R1 - n1(n1+1)/2 = 8; min(U1, U2) is the same)
res = stats.mannwhitneyu(treated, untreated, alternative="two-sided")
print(res)                                           # U = 8, p = 0.0379 (exact)

print("rank-biserial r:", 1 - 2*min(U1, U2)/(n1*n2))
```

⚠️ **Formula convention gotcha:** textbooks define U₁ either as R₁ − n₁(n₁+1)/2 (scipy) or as n₁n₂ + n₁(n₁+1)/2 − R₁ (lecture slides). They just swap labels; **min(U₁, U₂) is identical**.

</details>

---

## 17. Wilcoxon Signed-Rank Test

**What:** non-parametric alternative to the **paired-sample t-test** (and to the one-sample t-test). Used for two related samples, matched pairs, or repeated measurements on the same subjects. H₀: the median of the differences is 0.

**Why not just a sign test?** The sign test uses only the direction (+/−) of each difference. Wilcoxon also uses the **magnitude** (via ranks), so it's more powerful.

**Steps:**
1. For each pair, compute the difference dᵢ = before − after.
2. **Drop zero differences** (and reduce n accordingly).
3. Rank the **absolute** differences |dᵢ| from smallest to largest (average ties).
4. Attach the original sign to each rank.
5. W⁺ = sum of positive ranks, W⁻ = sum of negative ranks. Check: W⁺ + W⁻ = n(n+1)/2.
6. **W = min(W⁺, W⁻)**.
7. **Reject H₀ if W ≤ critical value** from the Wilcoxon table. For large n (> ~25), use
$$z = \frac{W - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}$$

<details>
<summary><b>▶ Deep dive: pain medication before vs. after</b></summary>

**Scenario:** A clinical trial evaluates a pain medication. 8 patients rate pain (0–10) before and after treatment. Pain scores are **ordinal**, so a paired t-test is questionable. α = 0.05, two-tailed.
*(Illustrative data built to match the lecture result W⁻ = 7, W⁺ = 29.)*

- **H₀:** no difference in pain ratings before vs. after (median difference = 0).
- **H₁:** there is a difference.

| Patient | Before | After | d = Before − After | \|d\| | Rank | Signed rank |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 7 | 6 | +1 | 1 | 1 | +1 |
| 2 | 7 | 5 | +2 | 2 | 2 | +2 |
| 3 | 4 | 7 | −3 | 3 | 3 | −3 |
| 4 | 1 | 5 | −4 | 4 | 4 | −4 |
| 5 | 8 | 3 | +5 | 5 | 5 | +5 |
| 6 | 8 | 2 | +6 | 6 | 6 | +6 |
| 7 | 9 | 2 | +7 | 7 | 7 | +7 |
| 8 | 10 | 1 | +9 | 9 | 8 | +8 |

- **W⁻** = 3 + 4 = **7**
- **W⁺** = 1 + 2 + 5 + 6 + 7 + 8 = **29**
- Check: 7 + 29 = 36 = 8·9/2 ✓
- **W = min = 7**

**Decision:** critical value for n = 8, α = 0.05 two-tailed = **3**. Since 7 > 3, we **fail to reject H₀** (exact p = 0.148).

**Conclusion:** insufficient evidence of a difference in pain ratings before and after treatment. (Most patients improved, but two got noticeably worse, and n = 8 gives low power.)

```python
from scipy import stats
import numpy as np

before = [7, 7, 4, 1, 8, 8, 9, 10]
after  = [6, 5, 7, 5, 3, 2, 2, 1]

# Manual
d = np.array(before) - np.array(after)
d = d[d != 0]                                   # drop zeros
ranks = stats.rankdata(np.abs(d))
w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
print(w_plus, w_minus, "W =", min(w_plus, w_minus))   # 29 7 W = 7

# scipy
print(stats.wilcoxon(before, after))            # statistic = 7, p = 0.148

# Compare with the paired t-test
print(stats.ttest_rel(before, after))           # p = 0.127 — same conclusion here
```

</details>

---

## 18. Kruskal-Wallis Test

**What:** non-parametric alternative to **one-way ANOVA** for **3+ independent groups**. H₀: all groups come from the same distribution. Essentially a Mann-Whitney extended to k groups.

**Use when** ANOVA assumptions fail — non-normal data within groups, unequal variances, ordinal outcomes, outliers.

**Steps:**
1. Combine all groups into one dataset.
2. Rank all observations (ties get the average rank).
3. Sum the ranks for each group: Rᵢ.
4. Compute
$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$
5. Compare H to a **χ² distribution with k − 1 df**. **Reject H₀ if H > critical value** (large H = significant, like χ²).
6. If significant, run a post-hoc test (**Dunn's test** with a multiple-comparison correction) to find which groups differ.

<details>
<summary><b>▶ Deep dive: three teaching methods</b></summary>

**Scenario:** A school compares three teaching methods — Traditional Lectures, Interactive Workshops, Online Modules. Students are randomly assigned; scores come from a standardized test. Scores aren't normal within groups and variances differ, so ANOVA assumptions are violated. α = 0.05.
*(Illustrative data.)*

| Traditional | Interactive | Online |
| :-: | :-: | :-: |
| 65 | 82 | 74 |
| 70 | 85 | 76 |
| 72 | 78 | 67 |
| 68 | 88 | 71 |
| 60 | 80 | 69 |

**Step 2 — Rank all 15 scores**

| Score | 60 | 65 | 67 | 68 | 69 | 70 | 71 | 72 | 74 | 76 | 78 | 80 | 82 | 85 | 88 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Group | T | T | O | T | O | T | O | T | O | O | I | I | I | I | I |
| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |

**Step 3 — Rank sums:** R_T = 1+2+4+6+8 = **21**, R_I = 11+12+13+14+15 = **65**, R_O = 3+5+7+9+10 = **34**. Check: 120 = 15·16/2 ✓

**Step 4 — H**

$$H = \frac{12}{15 \cdot 16}\left(\frac{21^2}{5} + \frac{65^2}{5} + \frac{34^2}{5}\right) - 3 \cdot 16 = 0.05 \times 1164.4 - 48 = 10.22$$

**Step 5 — Decision:** df = 3 − 1 = 2, critical χ² = 5.991. 10.22 > 5.991 → **reject H₀** (p = 0.006).

**Conclusion:** at least one teaching method's score distribution differs; Interactive Workshops rank highest. Use Dunn's test to confirm which pairs differ.

*Caveat:* with only 5 per group the χ² approximation is rough; exact tables are preferred for very small samples.

```python
from scipy import stats
import numpy as np

traditional = [65, 70, 72, 68, 60]
interactive = [82, 85, 78, 88, 80]
online      = [74, 76, 67, 71, 69]
groups = [traditional, interactive, online]

# Manual
all_vals = np.concatenate(groups)
ranks = stats.rankdata(all_vals)
N = len(all_vals)
idx = np.cumsum([0] + [len(g) for g in groups])
R = [ranks[idx[i]:idx[i+1]].sum() for i in range(len(groups))]
H = 12 / (N*(N+1)) * sum(r**2 / len(g) for r, g in zip(R, groups)) - 3*(N+1)
print(R, round(H, 2))                                 # [21, 65, 34] 10.22

# scipy (also applies a tie correction when needed)
print(stats.kruskal(*groups))                         # H = 10.22, p = 0.0060
print("critical:", stats.chi2.ppf(0.95, len(groups) - 1))

# Post-hoc (pip install scikit-posthocs)
# import scikit_posthocs as sp
# sp.posthoc_dunn(groups, p_adjust="holm")
```

</details>

---

## 19. Other Non-Parametric Tools

| Test | Replaces / used for | Python |
| :--- | :--- | :--- |
| **Sign test** | Paired data using only +/− signs (very few assumptions) | `stats.binomtest(n_pos, n_nonzero, 0.5)` |
| **Friedman test** | Repeated-measures ANOVA (3+ conditions, same subjects) | `stats.friedmanchisquare(a, b, c)` |
| **Spearman ρ / Kendall τ** | Pearson correlation (monotonic, ranks, outliers) | `stats.spearmanr`, `stats.kendalltau` |
| **Kolmogorov–Smirnov** | Are two samples from the same distribution? (any difference in shape) | `stats.ks_2samp(a, b)` |
| **Fisher's exact** | χ² on small 2×2 tables | `stats.fisher_exact(table)` |
| **McNemar** | Paired yes/no outcomes (before/after) | `statsmodels.stats.contingency_tables.mcnemar` |
| **Dunn's test** | Post-hoc after Kruskal-Wallis | `scikit_posthocs.posthoc_dunn` |
| **Mood's median test** | Do groups share a common median? | `stats.median_test(a, b, c)` |
| **Bootstrap / permutation** | Any statistic, minimal assumptions | see [Section 24](#24-bootstrap--permutation-tests) |

**Summary**
- **Mann-Whitney U** → used in place of the unpaired t-test when its assumptions aren't met.
- **Kruskal-Wallis** → used when the assumptions for one-factor ANOVA are not met.
- **Wilcoxon signed-rank** → used when the paired t-test can't be used because assumptions are violated.

---

# Part 5 — Applied Topics Interviewers Love

## 20. Checking Assumptions

| Assumption | How to check | If violated |
| :--- | :--- | :--- |
| **Normality** | Q-Q plot (best), histogram, Shapiro-Wilk (`stats.shapiro`) | Large n → CLT usually fine; small n → non-parametric, transform (log), or bootstrap |
| **Equal variances** | Levene (`stats.levene`), ratio of largest/smallest SD < ~2 | Welch t-test / Welch ANOVA |
| **Independence** | Study design (randomization, no repeated users, no clustering) | Paired/repeated-measures tests, mixed models, cluster-robust SE |
| **Expected counts ≥ 5** | Look at `expected` from `chi2_contingency` | Fisher's exact, merge categories |
| **No extreme outliers** | Box plot, IQR rule, z-scores | Investigate; rank-based tests; winsorize / trim (and say so) |

⚠️ Normality tests become "significant" for trivial deviations when n is large and have little power when n is small. **Look at the Q-Q plot.**

<details>
<summary><b>▶ Deep dive: assumption-check code</b></summary>

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
data = rng.exponential(10, 40)

print(stats.shapiro(data))                       # small p → evidence of non-normality
stats.probplot(data, dist="norm", plot=plt)      # Q-Q plot: points should hug the line
plt.show()

print(stats.levene(rng.normal(0, 1, 30), rng.normal(0, 3, 30)))   # small p → unequal variances

# Log transform often fixes right skew
print(stats.shapiro(np.log(data)))
```

</details>

---

## 21. Power, Sample Size & Effect Size

**Power = P(reject H₀ | H₁ true) = 1 − β.** Four linked quantities — fix any three and the fourth is determined:

| Quantity | ↑ it → required n |
| :--- | :--- |
| Significance level α (stricter = smaller) | smaller α → **more** n |
| Power (1 − β) | higher power → **more** n |
| Effect size / MDE | bigger effect → **less** n |
| Variance of the metric | more variance → **more** n |

**Sample size per group for comparing two means:**
$$n = \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\,\sigma^2}{\delta^2}$$

**For two proportions:**
$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\,[p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

For α = 0.05, power = 0.80: (1.96 + 0.84)² ≈ **7.85**. Quick rule for means: **n ≈ 16σ²/δ² per group**.

**Effect sizes to know**

| Measure | Used with | Small / Medium / Large |
| :--- | :--- | :--- |
| Cohen's d | t-tests | 0.2 / 0.5 / 0.8 |
| Cohen's h | proportions | 0.2 / 0.5 / 0.8 |
| η² | ANOVA | 0.01 / 0.06 / 0.14 |
| Cramér's V (df = 1) | χ² | 0.1 / 0.3 / 0.5 |
| r (Pearson/Spearman, rank-biserial) | correlation, rank tests | 0.1 / 0.3 / 0.5 |

<details>
<summary><b>▶ Deep dive: A/B test sample size</b></summary>

**Problem:** baseline conversion 10%; we want to detect a lift to 12% (MDE = 2 percentage points) with α = 0.05 (two-sided) and 80% power.

$$n = \frac{7.85 \times (0.10 \times 0.90 + 0.12 \times 0.88)}{0.02^2} = \frac{7.85 \times 0.1956}{0.0004} \approx 3{,}839 \text{ per group}$$

Halving the MDE to 1 pp would need roughly **4×** the users.

```python
import numpy as np
from scipy import stats

p1, p2, alpha, power = 0.10, 0.12, 0.05, 0.80
z_a, z_b = stats.norm.ppf(1 - alpha/2), stats.norm.ppf(power)
n = (z_a + z_b)**2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)**2
print(int(np.ceil(n)))                          # 3839

# statsmodels (uses Cohen's h, gives ≈ 3,835)
# from statsmodels.stats.power import NormalIndPower
# from statsmodels.stats.proportion import proportion_effectsize
# h = proportion_effectsize(p2, p1)
# NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1)
```

</details>

---

## 22. Multiple Testing Corrections

Running many tests inflates false positives. With 20 independent tests at α = 0.05:

$$P(\text{at least one false positive}) = 1 - 0.95^{20} \approx 64\%$$

| Method | Controls | Rule | Character |
| :--- | :--- | :--- | :--- |
| **Bonferroni** | FWER (any false positive) | reject if p ≤ α/m | Simple, very conservative |
| **Holm** | FWER | sort p's; compare p₍ᵢ₎ to α/(m − i + 1), stop at first failure | Uniformly better than Bonferroni |
| **Benjamini–Hochberg** | FDR (expected share of false discoveries) | largest i with p₍ᵢ₎ ≤ (i/m)·α; reject all up to it | More power; standard for many metrics / genomics |

- **FWER** = family-wise error rate; use when any false positive is costly.
- **FDR** = false discovery rate; use when screening many hypotheses.

<details>
<summary><b>▶ Deep dive: 8 metrics example</b></summary>

p-values (sorted): 0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205 (m = 8, α = 0.05)

- **No correction:** 5 significant.
- **Bonferroni** (threshold 0.05/8 = 0.00625): only **0.001** → 1 significant.
- **Holm:** 0.001 × 8 = 0.008 ✓; 0.008 × 7 = 0.056 ✗ → stop → 1 significant.
- **BH:** thresholds (i/8)·0.05 = 0.00625, 0.0125, 0.01875, … → 0.001 ✓, 0.008 ✓, 0.039 > 0.01875 ✗ (and no later p passes its threshold) → **2 significant**.

```python
from scipy import stats
p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205]
print(stats.false_discovery_control(p, method="bh"))   # BH-adjusted p-values
# adjusted: 0.008, 0.032, 0.067, ... → first two < 0.05

# statsmodels covers all methods:
# from statsmodels.stats.multitest import multipletests
# multipletests(p, alpha=0.05, method="holm")   # or "bonferroni", "fdr_bh"
```

</details>

---

## 23. A/B Testing End-to-End

1. **Define the goal & hypothesis** — e.g., "New checkout button increases purchase conversion."
2. **Pick metrics**
   - **Primary (OEC)** — one decision metric (conversion rate).
   - **Guardrails** — must not get worse (latency, refunds, unsubscribes).
   - **Secondary** — for understanding.
3. **Choose the randomization unit** — usually user (not session/pageview) to avoid inconsistent experiences and dependence.
4. **Power analysis** — set α, power, MDE → sample size → test duration (run **full weeks** to cover weekly seasonality).
5. **Run** — don't peek-and-stop; monitor only for bugs.
6. **Sanity checks** — **SRM** (χ² on the split), invariant metrics equal across groups, A/A tests.
7. **Analyze**
   - Proportions → two-proportion Z-test (or χ²).
   - Means (revenue, time) → Welch t-test; heavy tails → bootstrap or Mann-Whitney (note: tests a different hypothesis).
   - Ratio metrics (clicks per session) → **delta method** or bootstrap.
   - Many metrics / segments → multiple-testing correction.
8. **Decide** — combine statistical significance, CI, practical significance (does the lift cover the cost?), guardrails.

<details>
<summary><b>▶ Deep dive: common pitfalls & advanced topics</b></summary>

| Pitfall | What goes wrong | Fix |
| :--- | :--- | :--- |
| **Peeking** | Checking p daily and stopping at first p < 0.05 inflates false positives far above 5% | Fixed horizon; or sequential testing (alpha spending, mSPRT) |
| **SRM** | Unequal split signals a bug; results untrustworthy | χ² goodness-of-fit on counts; investigate before analyzing |
| **Novelty / primacy effect** | Users react to change itself; effect fades or grows | Run longer; look at effect over time / new users only |
| **Network effects / interference** | Treated users affect control users (marketplaces, social) | Cluster or geo randomization, switchback tests |
| **Simpson's paradox** | Aggregate result reverses within segments due to unequal mix | Stratify; keep allocation constant over time |
| **Underpowered test** | "No significant difference" ≠ "no effect" | Report CI; do power analysis first |
| **Slicing until significant** | p-hacking via segments | Pre-register segments; correct for multiple tests |
| **Heavy-tailed metrics** | Revenue outliers blow up variance | Winsorize/cap, log transform, bootstrap |

**Variance reduction — CUPED:** adjust the metric using the same user's pre-experiment value:
Y_adj = Y − θ(X − X̄), θ = cov(X, Y)/var(X). Same expected effect, lower variance → smaller required sample.

**Frequentist vs. Bayesian A/B:**
- Frequentist: p-value, "reject/fail to reject," fixed sample size.
- Bayesian: posterior (e.g., Beta-Binomial for conversion), "P(B > A) = 97%," expected loss; easier to explain, requires a prior.

```python
# Bayesian A/B with Beta(1,1) priors
import numpy as np
rng = np.random.default_rng(0)
a = rng.beta(1 + 200, 1 + 2000 - 200, 100_000)
b = rng.beta(1 + 260, 1 + 2000 - 260, 100_000)
print("P(B > A):", (b > a).mean())
print("Expected lift:", ((b - a) / a).mean())
```

</details>

---

## 24. Bootstrap & Permutation Tests

**Bootstrap** — estimates the **sampling distribution** of any statistic by resampling the data **with replacement** many times. Gives SEs and CIs for medians, percentiles, ratios, model metrics (AUC) — anything without a neat formula.

**Permutation test** — tests H₀ "group labels don't matter" by **shuffling labels** many times and recomputing the statistic. The p-value is the share of shuffles at least as extreme as the observed statistic. Exact under minimal assumptions (exchangeability).

| | Bootstrap | Permutation |
| :--- | :--- | :--- |
| Main output | CI / SE | p-value |
| Resampling | With replacement, within each group | Shuffle labels across groups |
| Simulates | Sampling variability | The null hypothesis |

<details>
<summary><b>▶ Deep dive: code</b></summary>

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.exponential(10, 40)

# Manual bootstrap CI for the median
boot = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(10_000)]
print("95% CI:", np.percentile(boot, [2.5, 97.5]))

# scipy bootstrap (BCa interval by default)
res = stats.bootstrap((x,), np.median, confidence_level=0.95, random_state=0)
print(res.confidence_interval)

# Permutation test for a difference in means
a = [23, 25, 21, 24, 22]
b = [27, 29, 26, 28, 30]
res = stats.permutation_test((a, b), lambda x, y: np.mean(x) - np.mean(y),
                             n_resamples=10_000, random_state=0)
print("p =", res.pvalue)                           # ≈ 0.0079 (exact: 2/252)
```

</details>

---

## 25. Correlation Tests

| | Pearson r | Spearman ρ | Kendall τ |
| :--- | :--- | :--- | :--- |
| Measures | Linear relationship | Monotonic relationship (on ranks) | Monotonic (concordant pairs) |
| Assumptions | ~Normal, linear, no outliers | Ordinal or continuous | Ordinal or continuous; good for small n / many ties |
| Outlier-robust | No | Yes | Yes |

- Test statistic for Pearson: t = r√(n − 2)/√(1 − r²), df = n − 2.
- **Correlation ≠ causation** (confounders, reverse causality).
- r² = share of variance explained in simple linear regression.

<details>
<summary><b>▶ Deep dive: one outlier changes everything</b></summary>

```python
from scipy import stats
x = [1, 2, 3, 4, 5, 100]     # one extreme value
y = [2, 4, 5, 4, 5, 6]
print(stats.pearsonr(x, y))   # r = 0.62, p = 0.19  (distorted by the outlier)
print(stats.spearmanr(x, y))  # ρ = 0.85, p = 0.03  (ranks ignore the magnitude)
print(stats.kendalltau(x, y))
```

</details>

---

# Part 6 — Anomaly Detection

## 26. Anomaly Detection Overview

**Question:** how can we detect observations that are extreme deviations from what we typically see in the data?

**Where it's used:**
- **Fintech / fraud** — unusual transactions, account takeovers.
- **ML model building** — finding influential outliers that distort a model.
- Monitoring (server metrics, sensor failures), manufacturing defects, data-quality checks.

**Outlier vs. anomaly:** an *outlier* is a statistically unusual value; an *anomaly* is an outlier that matters for the business (often generated by a different process, e.g., fraud). Not every outlier is an error — **investigate before deleting**.

**Types of anomalies:**
| Type | Example |
| :--- | :--- |
| **Point** | A single $50,000 purchase on a card that usually spends $50 |
| **Contextual** | 30°C is normal in July, anomalous in January |
| **Collective** | Many small transfers that together look like money laundering |

**Main approaches:**

| Method | Idea | Type | Scale |
| :--- | :--- | :--- | :--- |
| **Z-score** | How many SDs from the mean? | Statistical, univariate | Assumes ~normal |
| **IQR** | Beyond 1.5·IQR from Q1/Q3? | Statistical, univariate | Robust, no normality |
| **LOF** | Is the point in a much sparser region than its neighbors? | Density-based, multivariate | Local anomalies |
| **One-Class SVM** | Learn a boundary around normal data | Boundary-based, multivariate | Complex shapes, slower |
| **Isolation Forest** | Anomalies are easy to isolate with random splits | Tree-based, multivariate | Fast, high-dimensional |

**Setting:** usually **unsupervised** (few or no fraud labels). If you do have labels → treat it as an imbalanced classification problem (precision/recall, PR-AUC, not accuracy).

---

## 27. Z-Score & Modified Z-Score

$$z_i = \frac{x_i - \bar{x}}{s}$$

- Flag |z| > 3 (≈ 0.27% of normal data) or |z| > 2.5 for a looser rule.
- **Assumes roughly normal data.**
- **Weakness — masking:** outliers inflate both the mean and SD, which shrinks their own z-scores. With small n, |z| can't even reach 3 (max |z| ≤ (n−1)/√n).

**Modified Z-score (robust):** uses the median and MAD (median absolute deviation) instead.

$$M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}, \qquad \text{MAD} = \text{median}(|x_i - \tilde{x}|)$$

Flag |M| > 3.5 (Iglewicz & Hoaglin).

<details>
<summary><b>▶ Deep dive: example</b></summary>

Daily transactions: 52, 48, 50, 51, 49, 47, 53, 50, 200

- Mean = 66.67, SD = 50.03 → z(200) = (200 − 66.67)/50.03 = **2.66** → **not flagged at 3!** (masking — and with n = 9 no point can ever exceed |z| = 8/√9 = 2.67)
- Median = 50, MAD = 2 → M(200) = 0.6745 × 150 / 2 = **50.6** → clearly flagged.

```python
import numpy as np
x = np.array([52, 48, 50, 51, 49, 47, 53, 50, 200])

z = (x - x.mean()) / x.std(ddof=1)
print(np.round(z, 2), x[np.abs(z) > 3])          # 200 has z ≈ 2.66 → missed

med = np.median(x)
mad = np.median(np.abs(x - med))
m = 0.6745 * (x - med) / mad
print(np.round(m, 1), x[np.abs(m) > 3.5])        # 200 flagged

# scipy helpers: stats.zscore(x, ddof=1), stats.median_abs_deviation(x)
```

</details>

---

## 28. IQR Method

- Q1 = 25th percentile, Q3 = 75th percentile, **IQR = Q3 − Q1**.
- **Lower fence** = Q1 − 1.5·IQR, **Upper fence** = Q3 + 1.5·IQR. Points outside are outliers (this is exactly the box-plot whisker rule).
- Use 3·IQR for "extreme" outliers.
- **Pros:** no normality assumption, robust (quartiles barely move with outliers).
- **Cons:** univariate; with skewed data it flags many legitimate values in the long tail (consider a log transform first).
- For normal data, 1.5·IQR fences sit at about ±2.7σ (≈ 0.7% flagged).

<details>
<summary><b>▶ Deep dive: example</b></summary>

Same data: 47, 48, 49, 50, 50, 51, 52, 53, 200

- Q1 = 49, Q3 = 52 (numpy default linear interpolation) → IQR = 3
- Fences: 49 − 4.5 = **44.5**, 52 + 4.5 = **56.5**
- 200 > 56.5 → **outlier**

```python
import numpy as np
import pandas as pd

x = pd.Series([52, 48, 50, 51, 49, 47, 53, 50, 200])
q1, q3 = x.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
print(q1, q3, iqr, lower, upper)                # 49.0 52.0 3.0 44.5 56.5
print(x[(x < lower) | (x > upper)])             # 200

# Box plot shows the same rule visually
# x.plot.box()
```
Note: quartile values depend on the interpolation method, so textbook answers can differ slightly.

</details>

---

## 29. Local Outlier Factor (LOF)

**Idea:** compare the **local density** around each point with the density around its k nearest neighbors. A point in a much sparser region than its neighbors is an outlier.

- **LOF ≈ 1** → similar density to neighbors (normal).
- **LOF > 1** (e.g., > 1.5) → lower density than neighbors (outlier).

**Key concepts:**
1. **k-distance** — distance to the k-th nearest neighbor.
2. **Reachability distance** — reach-dist(A, B) = max(k-distance(B), dist(A, B)); smooths out noise.
3. **Local reachability density (lrd)** — inverse of the average reachability distance to neighbors.
4. **LOF(A)** = average of lrd(neighbors) / lrd(A).

**Strengths:** finds **local** outliers — e.g., a point near a dense cluster that would look normal globally next to a sparse cluster.
**Weaknesses:** sensitive to k (`n_neighbors`, commonly 20); O(n²) distance cost; needs **feature scaling**; struggles in very high dimensions.

<details>
<summary><b>▶ Deep dive: code</b></summary>

```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
normal = rng.normal(loc=[50, 50], scale=[5, 5], size=(200, 2))
anomalies = np.array([[90, 90], [10, 85], [85, 15], [50, 95], [5, 5]])
X = StandardScaler().fit_transform(np.vstack([normal, anomalies]))

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.025)
labels = lof.fit_predict(X)                    # -1 = outlier, 1 = inlier
scores = -lof.negative_outlier_factor_         # LOF score (higher = more anomalous)
print(np.where(labels == -1)[0])
print(np.round(scores[-5:], 2))                # the 5 planted anomalies score high

# To score NEW data, use novelty=True, fit on clean training data, then .predict(X_new)
```

</details>

---

## 30. One-Class SVM

**Idea:** train only on (mostly) normal data and learn a **boundary** that encloses it. New points outside the boundary are anomalies. With an RBF kernel, the boundary can take complex, non-linear shapes.

**Key hyperparameters:**
- **`nu`** (0–1) — upper bound on the fraction of training points treated as outliers (and lower bound on support vectors). Roughly your expected contamination rate.
- **`gamma`** — RBF kernel width. High gamma → tight, wiggly boundary (overfits); low → smooth boundary.

**Strengths:** flexible boundary; good for **novelty detection** (train on clean data, flag new deviations).
**Weaknesses:** needs **scaling**; slow on large datasets (roughly O(n²)–O(n³)); sensitive to nu/gamma; hard to interpret.

<details>
<summary><b>▶ Deep dive: code</b></summary>

```python
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
X_train = rng.normal(loc=[50, 50], scale=[5, 5], size=(200, 2))    # normal behavior
X_new = np.array([[51, 49], [90, 90], [10, 85], [48, 53]])

scaler = StandardScaler().fit(X_train)
ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(scaler.transform(X_train))

print(ocsvm.predict(scaler.transform(X_new)))            # expected: [ 1 -1 -1  1]
print(ocsvm.decision_function(scaler.transform(X_new)))  # negative = outside boundary
```

</details>

---

## 31. Isolation Forest & Comparison

**Isolation Forest (bonus, very common in industry):** builds random trees that split on random features at random values. Anomalies are few and different, so they get **isolated in fewer splits** (short path length). Anomaly score comes from the average path length across trees.
- Fast (near-linear), handles high dimensions, no scaling needed, no distance computations.
- Weaker for local anomalies next to dense clusters (where LOF shines).

**Other methods worth naming:** Mahalanobis distance (multivariate z-score; accounts for correlation), DBSCAN (noise points = outliers), autoencoders (high reconstruction error = anomaly), time-series methods (rolling z-score, STL decomposition residuals, Prophet intervals), Grubbs' test (formal test for a single outlier in normal data).

| | Z-score | IQR | LOF | One-Class SVM | Isolation Forest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dimensions | 1 | 1 | Multi | Multi | Multi |
| Distribution assumption | Normal | None | None | None | None |
| Robust to outliers in fit | No (use modified z) | Yes | Yes | Moderately | Yes |
| Needs scaling | — | — | Yes | Yes | No |
| Finds local anomalies | No | No | **Yes** | Somewhat | Weak |
| Speed on big data | Fast | Fast | Slow | Slow | **Fast** |
| Key knob | threshold (3) | multiplier (1.5) | n_neighbors, contamination | nu, gamma | contamination, n_estimators |

**Choosing a threshold (`contamination`)** is a business decision — trade off investigation cost (false positives) against missed fraud (false negatives). If a few labels exist, tune on precision@k or PR-AUC.

<details>
<summary><b>▶ Deep dive: compare all methods on the same data</b></summary>

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
normal = rng.normal(loc=[50, 50], scale=[5, 5], size=(200, 2))
anomalies = np.array([[90, 90], [10, 85], [85, 15], [50, 95], [5, 5]])
X = np.vstack([normal, anomalies])
df = pd.DataFrame(X, columns=["amount", "frequency"])
truth = np.r_[np.zeros(200), np.ones(5)].astype(bool)

# Univariate methods on "amount" only
x = df["amount"]
df["z"] = ((x - x.mean()) / x.std()).abs() > 3
q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
df["iqr"] = (x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)

# Multivariate methods on both features
Xs = StandardScaler().fit_transform(X)
df["lof"] = LocalOutlierFactor(n_neighbors=20, contamination=0.025).fit_predict(Xs) == -1
df["ocsvm"] = OneClassSVM(nu=0.025, gamma="scale").fit(Xs).predict(Xs) == -1
df["iforest"] = IsolationForest(contamination=0.025, random_state=0).fit_predict(Xs) == -1

for m in ["z", "iqr", "lof", "ocsvm", "iforest"]:
    print(f"{m:8s} flagged={df[m].sum():3d}  caught={(df[m] & truth).sum()}/5")
```

**Takeaway:** the planted point (50, 95) has a perfectly normal *amount*, so the univariate z-score and IQR rules on `amount` miss it; multivariate methods (LOF, Isolation Forest) catch it. Real anomalies are often unusual only in **combination** of features.

</details>

**Summary**
- Sometimes we want to detect observations that are different from all others.
- Ways to do it:
  - **Z-score** — number of standard deviations from the mean.
  - **IQR** — beyond 1.5·IQR below the 25th or above the 75th percentile.
  - **LOF** — density around each point; isolated points are outliers.
  - **SVM** — learn a boundary separating normal data from aberrant observations.
- Many more sophisticated methods exist (Isolation Forest, autoencoders, …).

---

# Part 7 — Interview Prep

## 32. Practice Questions

Try answering out loud before expanding.

<details>
<summary><b>Q1. What's the difference between a point estimate and a sampling distribution?</b></summary>

A point estimate is a single number from one sample (x̄ = 170). A sampling distribution describes how that statistic varies across all possible samples; its SD is the standard error. We need it to build confidence intervals and compute p-values.

</details>

<details>
<summary><b>Q2. When do you use a Z-test vs. a t-test?</b></summary>

Z when σ is known or n is large (s ≈ σ, CLT applies) and for proportion tests. t when σ is unknown and estimated by s, especially for small samples. For large n they give nearly identical results.

</details>

<details>
<summary><b>Q3. Explain a p-value to a non-technical stakeholder.</b></summary>

"If the change truly had no effect, how surprising would results like ours be? A p-value of 0.03 means we'd see a difference this large only about 3% of the time by random chance alone." It is not the probability the change works, and it says nothing about how big the effect is.

</details>

<details>
<summary><b>Q4. Your test isn't significant. Can you conclude there's no effect?</b></summary>

No. Absence of evidence isn't evidence of absence — the test may be underpowered. Report the confidence interval: if it's wide and includes meaningful effects, the result is inconclusive. Equivalence testing (TOST) is the formal way to show "no meaningful effect."

</details>

<details>
<summary><b>Q5. Why use ANOVA instead of multiple t-tests?</b></summary>

Multiple t-tests inflate the Type I error (4 groups → 6 tests → ~26% chance of a false positive). ANOVA tests all means at once at α; if significant, follow up with Tukey HSD, which controls the family-wise error rate.

</details>

<details>
<summary><b>Q6. Data are heavily skewed with n = 12 per group. What test do you use?</b></summary>

Mann-Whitney U (independent groups) or Wilcoxon signed-rank (paired). Alternatives: log-transform then t-test (tests the geometric mean), or a permutation/bootstrap approach. Mention that with large n the t-test would be fine thanks to the CLT.

</details>

<details>
<summary><b>Q7. What does Mann-Whitney actually test? Is it a test of medians?</b></summary>

It tests whether one distribution tends to produce larger values than the other (P(X > Y) ≠ 0.5). It's a test of medians only if both distributions have the same shape and differ just by a shift.

</details>

<details>
<summary><b>Q8. How would you test whether a user's device type is related to churn (yes/no)?</b></summary>

χ² test of independence on a device × churn contingency table; check expected counts ≥ 5 (else Fisher's exact / merge categories); report Cramér's V; inspect cell contributions or standardized residuals to see which devices drive the association.

</details>

<details>
<summary><b>Q9. Walk me through designing an A/B test.</b></summary>

Hypothesis → primary metric + guardrails → randomization unit → power analysis (α, power, MDE, baseline variance → n and duration in full weeks) → run without peeking → SRM and A/A sanity checks → analyze (Z-test for proportions, Welch t for means, delta method for ratios, corrections for multiple metrics) → decide using CI, practical significance and guardrails. Mention novelty effects and network interference.

</details>

<details>
<summary><b>Q10. In a 50/50 test you see 50,600 vs. 49,400 users. Problem?</b></summary>

Yes — χ² goodness of fit gives χ² = 14.4, p ≈ 0.00015. That's a sample ratio mismatch: likely a bug in assignment, logging, or bot filtering. Don't trust the results until it's explained.

</details>

<details>
<summary><b>Q11. You test 20 metrics and 1 is significant. What do you conclude?</b></summary>

About 1 false positive is expected by chance (20 × 0.05). Apply a correction (Bonferroni/Holm for FWER, Benjamini–Hochberg for FDR), pre-specify a single primary metric, and treat the rest as exploratory.

</details>

<details>
<summary><b>Q12. How do you get a confidence interval for the median revenue per user?</b></summary>

Bootstrap: resample users with replacement many times, compute the median each time, take the 2.5th and 97.5th percentiles (or BCa interval).

</details>

<details>
<summary><b>Q13. How would you reduce the variance of an A/B test metric?</b></summary>

CUPED (regress out pre-experiment behavior), stratification, winsorizing/capping heavy-tailed metrics, choosing a less noisy metric (conversion instead of revenue), or increasing the sample size.

</details>

<details>
<summary><b>Q14. How would you detect fraudulent transactions without labels?</b></summary>

Start with simple robust rules (modified z-score / IQR per customer, since "normal" is user-specific), then multivariate unsupervised methods — Isolation Forest (fast, scalable), LOF (local anomalies), One-Class SVM (novelty detection on clean data). Engineer features (amount vs. user's history, velocity, geography, time of day), scale when needed, set the contamination threshold based on review capacity, and validate with any available labels (precision@k, PR-AUC). Once labels accumulate, move to supervised imbalanced classification.

</details>

<details>
<summary><b>Q15. Why can the z-score method fail to detect outliers?</b></summary>

Masking: outliers inflate the mean and SD, shrinking their own z-scores; in small samples |z| can't even exceed (n−1)/√n. It also assumes normality. Use the modified z-score (median/MAD) or IQR.

</details>

<details>
<summary><b>Q16. When would you prefer LOF over Isolation Forest?</b></summary>

When anomalies are local — points slightly off a dense cluster that would look normal relative to a sparse cluster elsewhere. Isolation Forest is better for large, high-dimensional data and global anomalies.

</details>

<details>
<summary><b>Q17. Should you remove outliers before training a model?</b></summary>

Only after investigating. Remove clear data errors; keep genuine extreme values (they may be the signal, e.g., fraud). Alternatives: robust models/losses (Huber, tree models), transformations, capping. Check influence with Cook's distance / leverage for linear models. Always document what was removed.

</details>

<details>
<summary><b>Q18. What's Simpson's paradox? Give an example.</b></summary>

A trend in aggregated data reverses within subgroups because group sizes are unbalanced across a confounder. Example: a treatment looks worse overall but better in both mild and severe patient groups because it was given mostly to severe patients. Fix: stratify or control for the confounder.

</details>

<details>
<summary><b>Q19. Explain Type I vs. Type II error with a fraud example.</b></summary>

Type I (false positive): flagging a legitimate transaction as fraud → annoyed customer. Type II (false negative): missing actual fraud → financial loss. Lowering the threshold trades one for the other; the right balance depends on their relative costs.

</details>

<details>
<summary><b>Q20. What is the Central Limit Theorem and why does it matter?</b></summary>

The sampling distribution of the mean approaches normal as n grows, whatever the population's shape (given finite variance). It justifies Z/t-based CIs and tests — including A/B tests on non-normal metrics — when samples are large.

</details>

---

## 33. Common Mistakes

1. Using **Z when σ is unknown** with a small sample → intervals too narrow.
2. **Arithmetic slips in contingency totals** → always check that rows and columns both sum to N.
3. Running χ² on **percentages** instead of counts, or ignoring the **expected-count ≥ 5** rule.
4. Running **many t-tests instead of ANOVA**, or stopping after ANOVA without a **post-hoc** test.
5. Confusing **SD and SE**.
6. Reading **"fail to reject H₀" as "H₀ is true."**
7. Reporting only p-values — always add an **effect size** and **CI**.
8. Misinterpreting a CI as "95% chance μ is in this interval."
9. **Peeking** at A/B tests and stopping early; ignoring **SRM**.
10. **No multiple-testing correction** across many metrics/segments.
11. Treating Mann-Whitney as a test of medians without the same-shape assumption.
12. Forgetting to **drop zero differences** in Wilcoxon; mixing up U-formula conventions.
13. For rank tests, forgetting that **small** U/W is significant but **large** H is significant.
14. Using z-score outlier rules on **skewed** data or tiny samples (masking).
15. Forgetting to **scale features** for LOF / One-Class SVM.
16. **Deleting outliers** without investigating — sometimes they're the fraud you're looking for.
