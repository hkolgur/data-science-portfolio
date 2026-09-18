# A/B Testing & Online Controlled Experiments — Master Notes

> **Purpose:** Interview-ready reference for mid-level Data Scientist / Product DS roles.
> **Base:** Emma Ding (Data Interview Pro) A/B testing curriculum + *Trustworthy Online Controlled Experiments* (Kohavi, Tang, Xu) + Udacity A/B Testing.
> **Refreshed:** August 2026 — cross-checked against current industry practice (CUPED/CUPAC, sequential testing, clustered & switchback designs, delta method, quasi-experiments, AI-product experimentation).
> **Note:** Sections marked 🆕 are material that post-dates the classic 2020–2022 interview canon. Mid-level interviews now regularly probe at least CUPED, sequential testing, SRM, and interference.

---

## Table of Contents

1. [The 7-Step Framework (memorize this)](#1-the-7-step-framework)
2. [Foundations & Causal Framing](#2-foundations--causal-framing)
3. [Step 1 — Idea Selection & Opportunity Sizing](#3-step-1--idea-selection--opportunity-sizing)
4. [Step 2 — Metrics: OEC, Drivers, Guardrails](#4-step-2--metrics-oec-drivers-guardrails)
5. [Step 3 — Statistical Design & Power Analysis](#5-step-3--statistical-design--power-analysis)
6. [Step 4 — Randomization, Assignment & Instrumentation](#6-step-4--randomization-assignment--instrumentation)
7. [Step 5 — Running the Test & Validity Checks](#7-step-5--running-the-test--validity-checks)
8. [Step 6 — Analysis](#8-step-6--analysis)
9. [Step 7 — Decision & Post-Launch](#9-step-7--decision--post-launch)
10. [🆕 Modern Toolkit (what changed since 2022)](#10--modern-toolkit)
11. [Pitfalls, Traps & Twyman's Law](#11-pitfalls-traps--twymans-law)
12. [Interview Question Bank with Model Answers](#12-interview-question-bank)
13. [Cheat Sheet — Formulas & Numbers](#13-cheat-sheet--formulas--numbers)
14. [Corrections Applied to the Original Study Guide](#14-corrections-applied-to-the-original-study-guide)
15. [Glossary & Resources](#15-glossary--resources)

---

## 1. The 7-Step Framework

This is the skeleton for **any** open-ended A/B testing interview question ("How would you test X?"). Say the steps out loud, then go deep where the interviewer pushes.

| # | Step | The one question it answers |
|---|------|------------------------------|
| 1 | **Idea & opportunity sizing** | Is this worth testing at all? |
| 2 | **Metrics** | What does "better" mean, numerically? |
| 3 | **Hypothesis & design** | How much traffic, for how long, split how? |
| 4 | **Randomization & instrumentation** | Who gets what, and can we log it correctly? |
| 5 | **Run & validity checks** | Is the data trustworthy before I look at p-values? |
| 6 | **Analysis** | What is the effect, and how confident am I? |
| 7 | **Decision & post-launch** | Ship / iterate / kill — and what happens after? |

**Interview tip:** Interviewers penalize candidates who jump straight to "run a t-test." The signal they look for is *design thinking* (steps 1–4) and *trust* (step 5). The statistics are table stakes.

---

## 2. Foundations & Causal Framing

### 2.1 What an A/B test actually is

- An **A/B test** (online controlled experiment, OCE) is a randomized controlled trial run on live product traffic. Users are randomly split into a **control** (current experience) and one or more **treatments** (variants). Everything except the variant is held constant.
- Randomization is what buys you **causality**: in expectation, the two groups are identical on every observed *and unobserved* covariate, so any post-treatment difference is attributable to the change.

### 2.2 Potential-outcomes vocabulary (worth 30 seconds in a senior-ish interview)

- Each user $i$ has two potential outcomes: $Y_i(1)$ if treated, $Y_i(0)$ if not. We only ever observe one — the **fundamental problem of causal inference**.
- **ATE (Average Treatment Effect)** = $E[Y_i(1) - Y_i(0)]$. This is what a standard A/B test estimates.
- **CATE / HTE** = the same effect conditional on covariates $X$ — i.e., who benefits more. See §10.6.
- **SUTVA (Stable Unit Treatment Value Assumption)** — two parts:
  1. **No interference**: user $i$'s outcome doesn't depend on user $j$'s assignment.
  2. **No hidden variations of treatment**: "treatment" means the same thing for everyone.
  - SUTVA is the assumption that breaks in social networks and marketplaces (§10.5). Naming SUTVA by name is a strong signal in interviews.

### 2.3 Core vocabulary

| Term | Definition |
|------|-----------|
| **Variant** | A version of the experience. Can be a button color or a full ranking-model swap. |
| **Randomization unit** | The entity randomly assigned to a variant — user, session, page view, device, cookie, geo, or time slice. |
| **Analysis unit** | The grain at which the metric is computed (e.g., per order, per session). **May differ from the randomization unit** — this changes your variance math (§8.3). |
| **Exposure / triggering** | The moment a user actually hits the code path that differs. Only triggered users should be analyzed (§8.6). |
| **Ramp / rollout** | Progressive traffic allocation (1% → 5% → 20% → 50%) used to limit blast radius. |
| **Holdout** | A slice of users deliberately kept on control long-term to measure cumulative/long-run impact. |

**Rule of thumb on randomization unit:** it must be **at least as coarse as** the analysis unit. If you randomize by session but analyze per user, a single user can land in both arms — the contrast is destroyed. **User-level randomization is the default** because it guarantees a consistent experience, which matters both for UX and for validity.

### 2.4 Prerequisites for a trustworthy experiment

- [ ] **A clear, pre-agreed OEC** (§4) that stakeholders signed off on *before* launch.
- [ ] **Engineering ability to isolate the change.** A monolithic redesign of ten things at once cannot attribute causality to any one of them. If the change must be big, use a **factorial design** or test components sequentially.
- [ ] **Sufficient traffic.** Not "thousands" as a magic number — the required $n$ comes out of the power calculation (§5.3). What you need is enough traffic that the runtime implied by that $n$ is measured in weeks, not quarters.
- [ ] **Reliable instrumentation** — exposure logging fires at the moment of exposure, and downstream events join back to the assignment.
- [ ] **Short feedback loop.** If the outcome takes 6 months to realize (e.g., annual renewals), you need a surrogate/driver metric or a different method.

### 2.5 When *not* to A/B test

| Situation | Why it fails | What to do instead |
|---|---|---|
| Very low traffic | Underpowered; runtime is unrealistic | Qualitative research; larger MDE; sequential design; pooled/meta analysis |
| Change affects everyone (pricing, brand, legal, ToS) | No valid control | Quasi-experiment: geo-lift, DiD, synthetic control (§10.7) |
| Strong network/marketplace spillover | SUTVA violated | Cluster, switchback, or two-sided randomization (§10.5) |
| Long-horizon outcome (retention at 12 months) | Test would run forever | Surrogate metrics + long-term holdout |
| Ethically/legally problematic to withhold | Consent, fairness, regulation | Observational causal inference; staged rollout with monitoring |
| One-shot events (Super Bowl ad, launch keynote) | No repetition | Pre/post with control markets |

---

## 3. Step 1 — Idea Selection & Opportunity Sizing

You cannot A/B test every idea; experiment capacity is a scarce resource. Prioritize before you design.

### 3.1 Quantitative sizing

- Use historical/observational data to estimate the **ceiling** of the idea: how many users hit this surface, what's the current conversion, what would a plausible lift be worth?
- Example script: *"3.2M users/month reach the checkout page; 4% abandon at the shipping step. If the new shipping-cost preview recovers even 10% of those, that's ~12.8K incremental orders/month × \$42 AOV ≈ \$537K/month."*
- **Funnel/drop-off analysis** identifies where the biggest leaks are.

### 3.2 Qualitative sourcing

- User research, session replays, support tickets, surveys, usability studies, sales/CS feedback.
- Qualitative work generates *hypotheses*; quantitative sizing *ranks* them.

### 3.3 Prioritization frameworks (name-drop one)

- **ICE** — Impact × Confidence × Ease.
- **PIE** — Potential × Importance × Ease.
- **RICE** — Reach × Impact × Confidence ÷ Effort.
- The point isn't the acronym; it's that you're weighing expected value against experiment cost and opportunity cost.

### 3.4 Reality check to quote in interviews

Across mature experimentation programs, **roughly 70–90% of shipped ideas fail to produce a statistically significant positive effect on the target metric.** Microsoft/Bing has reported success rates around one-third; other platforms are lower. This is the strongest argument for testing rather than shipping on intuition — and a great line to drop when asked "why bother experimenting?"

---

## 4. Step 2 — Metrics: OEC, Drivers, Guardrails

This is where mid-level candidates most often get separated from junior ones.

### 4.1 The metric hierarchy

| Layer | Also called | Role | Example (food-delivery app) |
|---|---|---|---|
| **Goal metric** | North Star, success metric | What the org ultimately cares about; slow-moving, hard to move in one test | Monthly active purchasers; user lifetime value |
| **Driver metric** | Signpost, surrogate, proxy, indirect | Shorter-term, more sensitive, causally believed to *lead to* the goal | Orders per user per week; search→order conversion |
| **Guardrail metric** | Invariant (trust type) | Must not degrade; protects business and validates the test | p95 latency, crash rate, refund rate, unsubscribes, SRM |
| **Diagnostic / debug** | Secondary | Explains *why* the OEC moved | CTR by module, funnel step rates |

### 4.2 The OEC (Overall Evaluation Criterion)

- A **single** quantitative measure — or a deliberately weighted composite — that decides success. Kohavi's framing: it is the experiment's *decision rule*.
- Properties of a good OEC:
  - **Aligned** with long-term business value, not just short-term clicks.
  - **Sensitive** enough to move within a normal experiment window.
  - **Measurable and timely** within the test duration.
  - **Not gameable.** Optimizing raw clicks invites clickbait; that's why Bing's OEC includes sessions-per-user and successful-task signals, not just CTR.
  - **Agreed cross-functionally before launch** — otherwise every stakeholder reads the results differently and you get decision paralysis.
- Composite example: $\text{OEC} = w_1 \cdot \text{(normalized engagement)} + w_2 \cdot \text{(normalized revenue)} - w_3 \cdot \text{(normalized complaint rate)}$.
- **Classic cautionary tale:** hiding house rules on Airbnb increased bookings (the OEC) but tanked review ratings (a guardrail) — the test was scrapped. Short-term OEC wins can destroy long-term value.

### 4.3 Guardrail metrics — two distinct flavors

1. **Trust-related guardrails (invariants).** Should be *identical* across arms by design. If they move, you have a bug, not a finding.
   - Sample Ratio (SRM) — the single most important one (§7.2).
   - Cache-hit rate, ratio of users per variant, pre-experiment metric balance, bot traffic share.
2. **Organizational guardrails.** Real business metrics you're not willing to sacrifice.
   - Page load / p95 latency, crash rate, error rate, uninstall rate, unsubscribe rate, support-ticket volume, refund rate, ad revenue.

**Discipline:** be selective. Every extra guardrail is another chance for a false alarm. Ten guardrails at α = 0.05 gives you roughly a 40% chance of at least one spurious red flag ($1 - 0.95^{10} \approx 0.40$).

### 4.4 Metric types and why it matters statistically

| Type | Example | Test / variance treatment |
|---|---|---|
| **Proportion / binary** | conversion rate, D1 retention | Two-proportion z-test; $\sigma^2 = p(1-p)$ |
| **Continuous, per randomization unit** | revenue per user, sessions per user | Welch's t-test / CLT z-test |
| **Ratio (analysis unit ≠ randomization unit)** | CTR = clicks/impressions, AOV = revenue/order | **Delta method** or clustered SEs (§8.3) — *not* a naive t-test |
| **Count** | # searches | Often log-transform or use Poisson/negative-binomial framing |
| **Percentile** | p95 latency | Bootstrap or quantile-specific methods; means hide tail regressions |
| **Heavy-tailed / skewed** | revenue with whales | Winsorize/cap, bootstrap, or transform; report both capped and uncapped |

### 4.5 Metric sensitivity ladder

When the OEC is too insensitive to move, climb *down* the ladder toward more sensitive proxies — but keep the goal metric as a guardrail so you don't optimize a proxy into a local maximum.

`Revenue (insensitive) → Orders → Add-to-cart → Product-detail views → Search CTR (very sensitive)`

---

## 5. Step 3 — Statistical Design & Power Analysis

### 5.1 Hypotheses

- **Null $H_0$:** the variant has no effect on the OEC — $\mu_T - \mu_C = 0$. Observed differences are sampling noise.
- **Alternative $H_1$:** $\mu_T - \mu_C \neq 0$ (two-sided is the industry default; you care about regressions too, and one-sided tests invite motivated reasoning).

### 5.2 The four parameters

| Symbol | Name | Meaning | Industry default |
|---|---|---|---|
| $\alpha$ | Significance level | P(reject $H_0$ \| $H_0$ true) = **Type I error / false positive** | 0.05 (convention, *not* a law — tighten under multiple testing, loosen to 0.10 for cheap exploratory tests) |
| $\beta$ | Type II error rate | P(fail to reject $H_0$ \| $H_1$ true) = **false negative** | 0.20 |
| $1-\beta$ | **Power** | P(detect a true effect of size MDE) | 0.80 (0.90 for high-stakes/expensive changes) |
| MDE | Minimum Detectable Effect | Smallest effect that would actually change the business decision | Set by product/finance, not by the DS alone |

**Type I vs Type II in plain English (say it this way):** A Type I error ships a feature that does nothing — you pay maintenance costs forever for zero value. A Type II error kills a feature that works — you lose the upside. Which is worse depends on the cost of the change; that asymmetry is what should drive your α/β choices, not habit.

**MDE is a business input, not a statistical one.** Frame it as: *"What's the smallest lift that would justify the engineering, maintenance, and opportunity cost of shipping this?"* If the answer is "0.5% lift on revenue," that's your MDE.

### 5.3 Sample size

**Per-group sample size**, two-sided test, comparing two means:

$$n_{\text{per group}} \approx \frac{2\sigma^2\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2}{\delta^2}$$

- $\sigma^2$ = pre-experiment (baseline) variance of the metric, estimated from historical data.
- $\delta$ = the **absolute** difference you want to detect (= MDE expressed in metric units).
- $z_{1-\alpha/2} = 1.96$ at α = 0.05; $z_{1-\beta} = 0.84$ at 80% power.

**Kohavi's rule of 16.** At α = 0.05 and power = 0.80:

$$2(1.96 + 0.84)^2 = 2(2.80)^2 = 15.68 \approx 16 \quad \Rightarrow \quad \boxed{n_{\text{per group}} \approx \frac{16\sigma^2}{\delta^2}}$$

Memorize this. It's the fastest way to do sample sizing on a whiteboard.

**For a proportion metric**, substitute $\sigma^2 = p(1-p)$ where $p$ is the baseline rate:

$$n_{\text{per group}} \approx \frac{16\,p(1-p)}{\delta^2}$$

**Worked example.** Baseline conversion $p = 5\%$, you want to detect a **relative** 5% lift → absolute $\delta = 0.05 \times 0.05 = 0.0025$.

$$n \approx \frac{16 \times 0.05 \times 0.95}{0.0025^2} = \frac{16 \times 0.0475}{0.00000625} = \frac{0.76}{0.00000625} \approx 121{,}600 \text{ per group}$$

≈ 243K users total. At 40K eligible users/day, that's ~6 days of traffic → round up to **14 days** to cover two full weekly cycles.

**Critical relationship — get this right:**

> Sample size scales with $1/\delta^2$. Halving the MDE **quadruples** the required sample. This is *quadratic (inverse-square)*, **not exponential** — a very common misstatement and an easy thing for an interviewer to catch.

| Change | Effect on required $n$ |
|---|---|
| MDE halved | ×4 |
| MDE ÷ 10 | ×100 |
| Variance doubled | ×2 |
| Power 80% → 90% | ×1.34 (since $(1.96+1.28)^2/(1.96+0.84)^2$) |
| α 0.05 → 0.01 | ×1.49 |

### 5.4 Duration

Runtime $\approx \dfrac{n_{\text{per group}} \times \text{number of arms}}{\text{eligible units per day} \times \text{traffic allocation}}$ — then **round up** to satisfy all of the following:

- **Full weekly cycles.** Behavior on a Tuesday afternoon differs from Saturday night. Always run in **multiples of 7 days**; 1–2 weeks is the common floor, 2–4 weeks typical. Ending mid-week bakes in day-of-week bias.
- **Novelty effect.** Users engage with something *because it's new*; the spike decays. Longer runs separate genuine lift from curiosity.
- **Primacy effect (change aversion).** The mirror image — habituated users resist a changed interface, then adapt. Early results look artificially bad.
- **Cookie/ID churn.** On web, cookie-based IDs churn; very long tests accumulate identity drift and dilute the effect.
- **Seasonality & external events.** Avoid Black Friday, holidays, major launches, or bracket them symmetrically across arms.
- **Business cycle.** For purchase behavior, run at least one full purchase cycle.

**Trade-off to state explicitly:** longer runtime buys power and long-run signal but costs iteration velocity and (for a losing variant) real user harm. That tension is why variance reduction and sequential testing exist (§10.1–10.2).

**Detecting novelty/primacy while running:**
- Plot the treatment effect **by day since first exposure** — a decaying curve suggests novelty, a rising one suggests primacy.
- Compare **new users vs existing users**: new users have no prior habit, so a gap between the two groups is diagnostic.
- Restrict analysis to first-time users as a clean read.

### 5.5 Traffic allocation

- **50/50 is the most powerful split** for a fixed total sample (variance of the difference is minimized at equal allocation).
- Unequal splits (90/10) are used for **risk management** during ramp, not for statistical reasons. You pay a power penalty: a 90/10 split needs ~2.8× the total traffic of 50/50 for the same power.
- **Ramp plan** for a risky change: 1% (1 day, guardrails only) → 5% → 20% → 50% (full measurement period). Only the final, stable allocation period should be used for the primary read — mixing ramp phases creates a **Simpson's paradox** trap (§11).
- **Multiple variants (A/B/n):** each additional arm splits your traffic and triggers the multiple-comparisons problem (§8.5). Keep arms to 2–4 unless you have enormous traffic.

---

## 6. Step 4 — Randomization, Assignment & Instrumentation

### 6.1 Choosing the randomization unit

| Unit | When to use | Trade-off |
|---|---|---|
| **User / account ID** | Default. Any UX-visible change. | Consistent experience; requires login or stable ID |
| **Cookie / device** | Logged-out web traffic | Churns; same human counted multiple times; cross-device leakage |
| **Session** | Change is self-contained within a session, no memory | Higher power (more units) but inconsistent UX; invalid for anything users remember |
| **Page view / request** | Backend-only changes invisible to users (e.g., caching) | Highest power; only valid when truly invisible |
| **Cluster (social graph, household, team)** | Network effects present | Fewer effective units → much lower power |
| **Geo (city, DMA, region)** | Marketplace or marketing tests | Very few units; high variance; needs synthetic-control style analysis |
| **Time slice (switchback)** | Marketplace policies: pricing, dispatch, matching | Handles interference; needs washout periods; temporal autocorrelation |

**Hashing:** assignment is usually `hash(user_id + experiment_salt) % 100`. The **salt per experiment** is what makes concurrent experiments independent. A poor hash function (e.g., FNV in older systems) can correlate assignments across experiments; cryptographic hashes (MD5) or high-quality non-crypto hashes (SpookyHash, MurmurHash) are standard.

### 6.2 Concurrent experiments & layers

- Large orgs run hundreds of simultaneous tests. **Layered / overlapping experiment design** (Google's terminology) lets one user be in many experiments at once, as long as those experiments are in different *layers* and don't touch the same UI surface.
- **Interaction effects** between concurrent experiments are real but rare in practice; platforms typically detect them via automated pairwise interaction scans rather than blocking overlap.
- Mutually exclusive groups are reserved for experiments that would visibly collide.

### 6.3 Instrumentation checklist

- [ ] **Exposure event fires at the moment of exposure**, not at page load. Logging assignment for users who never saw the change dilutes the effect (§8.6).
- [ ] Assignment is **sticky** — the same user gets the same variant across sessions and devices where possible.
- [ ] Downstream events carry the experiment/variant ID or can be reliably joined to it.
- [ ] Both arms are instrumented **identically**. Adding a tracking pixel only to treatment is a classic source of SRM.
- [ ] Bots and internal traffic are filtered *consistently* across arms. (Bot traffic can be >50% of raw traffic on some properties — filtering it asymmetrically is fatal.)
- [ ] Late-arriving/delayed events are handled the same way in both arms.

---

## 7. Step 5 — Running the Test & Validity Checks

**Do these before you look at a single p-value.** In interviews, volunteering this section unprompted is a strong differentiator.

### 7.1 A/A tests

- Run an experiment where both arms are identical. Purpose:
  - Verify the platform's **Type I error rate is actually ~5%** — over many A/A tests, p-values should be uniformly distributed.
  - Detect pre-existing bias between buckets, broken hashing, or logging asymmetries.
  - Empirically estimate metric variance for future power calculations.
- 🆕 A/A testing is also the standard sanity check for a *new design*: if you switch to cluster or switchback randomization, run an A/A under the new design. If outcomes drift with no treatment, your design hasn't actually removed the interference.

### 7.2 Sample Ratio Mismatch (SRM) — the #1 trust guardrail

**Definition:** the observed split between arms differs from the designed split by more than chance allows.

**Chi-square goodness-of-fit test:**

1. Expected counts under a 50/50 design: $E_C = E_T = \dfrac{N_C + N_T}{2}$
2. Statistic: $\chi^2 = \sum \dfrac{(O - E)^2}{E} = \dfrac{(N_C - E_C)^2}{E_C} + \dfrac{(N_T - E_T)^2}{E_T}$
3. Compare against $\chi^2$ with $k - 1$ degrees of freedom ($k$ = number of arms; df = 1 for a standard A/B).
4. **Convention: flag SRM if p < 0.0005 (some platforms use 0.001).** The threshold is deliberately far stricter than 0.05 because platforms run this check on every experiment continuously, and because a *true* SRM is catastrophic while a false alarm is merely annoying.

**If SRM fires: stop. Do not interpret the results.** Even a tiny imbalance implies a non-random mechanism, which means the arms differ on unobserved characteristics too — and that bias is usually far larger than the effect you're trying to measure.

**Common root causes (know these):**
- Buggy or mis-salted randomization / bucketing.
- **Asymmetric filtering** — e.g., bot filters or "exclude users with errors" applied differently across arms.
- Treatment crashes or times out, so treated users never log the downstream event (survivorship).
- **Redirect-based tests** — treatment is redirected to a new URL and loses users to latency/drop-off; control isn't.
- Data pipeline joins losing rows for one variant.
- **Residual/carryover effects** — reusing buckets from a prior experiment without re-randomizing.
- Ramp changes mid-experiment (traffic allocation changed but analysis window didn't).
- Delayed logging on one platform (e.g., iOS batches events, Android doesn't).

**Diagnosing:** segment the SRM by day, platform, browser, country, and new-vs-returning. The segment where the ratio breaks usually reveals the bug.

### 7.3 Other during-experiment monitoring

- **Guardrail alerting** on latency, crash rate, error rate — check within hours, not at the end.
- **Kill-switch criteria** defined in advance: e.g., "roll back immediately if crash rate rises >10% or revenue drops >5% with p < 0.01."
- **Do not stop early on a significant OEC** under a fixed-horizon design — that's peeking, and it inflates false positives dramatically (§10.2). Stopping early for *harm* (guardrail breach) is different and always allowed.

---

## 8. Step 6 — Analysis

### 8.1 Choosing the test

| Metric | Test | Notes |
|---|---|---|
| Proportion (conversion, retention) | Two-proportion z-test; chi-square test of independence | Equivalent for 2×2 |
| Mean, large n | Two-sample z-test (CLT) or Welch's t-test | Welch's (unequal variance) is the safe default — do **not** assume equal variance |
| Mean, heavy-tailed / outliers | Winsorized/capped mean + t-test, or bootstrap | Report both capped and raw |
| Ratio, analysis unit ≠ randomization unit | **Delta method** or cluster-robust SEs | §8.3 |
| Distribution shift, non-normal, small n | Mann–Whitney U / permutation test | Tests stochastic dominance, not means — say so |
| Percentiles (p95 latency) | Bootstrap / quantile regression | Means can look fine while the tail regresses |
| Count data | Poisson / negative binomial, or log-transform | Overdispersion is the norm |

**Practical note:** with $n$ in the tens of thousands, the CLT makes the z-test and t-test numerically identical. Don't burn interview time on "t vs z" — spend it on variance structure.

### 8.2 Interpreting the output correctly

- **p-value:** P(observing data this extreme or more | $H_0$ is true). It is **not** the probability that $H_0$ is true, and **not** the probability the effect is real.
- **"Not significant" ≠ "no effect."** It means insufficient evidence — often insufficient *power*. Report the confidence interval so the reader can see what effects remain plausible.
- **Confidence interval** is the more useful object for decisions: a 95% CI of `[+0.1%, +4.0%]` is "significant" but tells you the true effect could be trivially small. **Always report CIs, not just p-values.**
- Report **relative lift** with CI (e.g., "+2.3% [+0.8%, +3.8%]") — stakeholders think in relative terms.

### 8.3 🆕 Ratio metrics, the delta method & clustered standard errors

This is the single most common *technical* gap in mid-level candidates.

**The problem:** you randomize by **user**, but the metric is per **session**, per **order**, or per **impression** (CTR = clicks/impressions, AOV = revenue/order). Sessions from the same user are correlated. Treating each session as an independent observation **underestimates the variance**, inflating the false-positive rate. Run A/A simulations under this mistake and the p-value distribution shows a telltale bump near 0.

**Two equivalent fixes:**

1. **Delta method.** Approximate the variance of the ratio $R = \bar{X}/\bar{Y}$ via a first-order Taylor expansion:

   $$\mathrm{Var}(R) \approx \frac{1}{n}\left(\frac{\mathrm{Var}(X)}{\bar{Y}^2} - \frac{2\bar{X}\,\mathrm{Cov}(X,Y)}{\bar{Y}^3} + \frac{\bar{X}^2\,\mathrm{Var}(Y)}{\bar{Y}^4}\right)$$

   where $X$ and $Y$ are the per-user aggregated numerator and denominator, and $n$ is the number of users (randomization units).

2. **Cluster-robust standard errors.** Regress the outcome on the treatment indicator with standard errors clustered at the randomization unit. **These two give the same answer** — delta-method variance equals OLS variance with cluster-robust SEs. Mentioning this equivalence is a strong signal.

**Alternative:** nonparametric **bootstrap at the user level** (resample users, not sessions). Slower but assumption-light; good cross-check for heavy tails.

**Interview answer, compressed:** *"Because the randomization unit is the user but the analysis unit is the session, observations within a user are correlated. I'd use the delta method — or equivalently, cluster-robust standard errors at the user level — to avoid understating variance and inflating false positives."*

### 8.4 Segmentation & heterogeneity

- Standard cuts: platform (iOS/Android/web), new vs returning, country/locale, browser, device tier, usage decile, acquisition channel.
- **Pre-register your segments.** Unlimited post-hoc slicing is p-hacking. If you must explore, treat findings as *hypothesis-generating* and confirm in a follow-up test.
- Segment analysis is what surfaces bugs ("the win is entirely on Android; iOS is flat — is the iOS build even shipping the variant?").
- 🆕 For principled heterogeneity, use HTE methods (§10.6) rather than eyeballing subgroups.

### 8.5 Multiple testing

Every additional metric or arm inflates the family-wise false-positive rate. With $m$ independent tests at α: $P(\text{≥1 false positive}) = 1 - (1-\alpha)^m$. At $m = 20$, that's **64%**.

| Correction | Method | When to use |
|---|---|---|
| **Bonferroni** | Use $\alpha/m$ | Few tests, need strict FWER control. Simple but very conservative — kills power |
| **Šidák** | $1-(1-\alpha)^{1/m}$ | Slightly less conservative than Bonferroni |
| **Holm–Bonferroni** | Step-down | Strictly more powerful than Bonferroni, same FWER guarantee — prefer it |
| **Benjamini–Hochberg (FDR)** | Controls $E\left[\frac{\text{false positives}}{\text{rejections}}\right]$ | **The industry default when scoring many metrics.** Much more powerful; accepts a controlled proportion of false discoveries |

**Practical convention used by real platforms:** don't correct everything uniformly. Apply **no correction to the single pre-registered OEC**, apply **FDR (BH) across the broad secondary metric panel**, and use **loose thresholds on guardrails** (you *want* to be sensitive to harm). State this tiered approach in an interview — it shows you've thought past the textbook.

### 8.6 🆕 Triggering & dilution

- If only 8% of users ever reach the changed surface, including the other 92% in the analysis **dilutes the effect by ~12×** and destroys your power.
- **Analyze only triggered users** (those who actually hit the code path), but compute the *counterfactual trigger* in control too — i.e., users who *would have* triggered. Comparing triggered-treatment against all-control is a broken comparison.
- Then **translate back**: overall business impact = triggered-population effect × trigger rate. Report both numbers.
- Triggering is also a free variance-reduction technique — it's often the single biggest sensitivity win available.

---

## 9. Step 7 — Decision & Post-Launch

### 9.1 The decision matrix (statistical × practical significance)

The correct version compares the **confidence interval** to the **MDE / practical significance boundary**, not just the point estimate.

| # | Confidence interval vs 0 and MDE | Reading | Recommendation |
|---|---|---|---|
| 1 | Entire CI **above** MDE | Clear, meaningful win | **Ship.** |
| 2 | CI excludes 0, but **entirely below** MDE | Statistically significant, practically trivial. Typical of huge samples where a 0.02% lift clears p < 0.05 | **Don't ship** (or ship only if maintenance cost ≈ 0). The lift doesn't repay tech debt + opportunity cost |
| 3 | CI excludes 0 and **straddles** MDE | Real effect; unclear if it's big enough | **Inconclusive → extend or re-run** with more power, or make a judgment call with stakeholders |
| 4 | CI includes 0 but is **narrow and within ±MDE** | Confidently flat | **Don't ship.** Genuine null result — this is real information, log it |
| 5 | CI includes 0 and is **wide**, extending past MDE | Underpowered; you learned nothing | **Inconclusive → increase sample / reduce variance / re-run.** Do *not* call this "no effect" |
| 6 | Entire CI **below** 0 | Real harm | **Roll back**, and investigate why the hypothesis was wrong |

**Key line for interviews:** *"Statistical significance tells me the effect is real. Practical significance tells me it's worth it. I need both, and I read them off the confidence interval, not the p-value alone."*

### 9.2 Conflicting metrics

Metrics regularly move in opposite directions — e.g., engagement +4%, ad revenue −1.5%.

Resolution order:
1. **Return to the pre-registered OEC.** This is precisely why you agreed on one up front — it pre-commits the trade-off.
2. **Monetize both movements** into a common unit. If +4% engagement is historically worth \$X/user/year in retention and the ad loss is \$Y, compare X and Y directly.
3. **Check the time horizon.** Short-term revenue loss may be a deliberate investment in retention; validate with a long-term holdout.
4. **Escalate with a clear recommendation**, not a shrug. Present: effect sizes with CIs, the monetized net, the assumptions behind the monetization, and your recommendation.
5. **Segment.** Sometimes the loss is concentrated in a segment you can carve out.

### 9.3 Costs that a positive lift must clear

- **Engineering maintenance / tech debt** — every shipped variant is permanent code, dependencies, and future bug surface.
- **Opportunity cost** — the team maintaining a 0.1% winner isn't building the next 5% winner.
- **Complexity cost to users** — feature bloat degrades the product even when each feature "wins."
- **Operational cost** — infra, support load, compliance review.

### 9.4 Post-launch

- **Long-term holdout.** Keep 1–5% of users on control for weeks-to-months after launch to measure the true cumulative effect. Effects commonly decay — Microsoft and others report that measured lift at 3 months is often materially smaller than the 2-week read.
- **Short- vs long-term divergence.** Track 1/3/6/12-month cohorts. Novelty fades; competitors adapt; users habituate.
- **User fatigue.** The canonical example: increasing push notifications lifts 14-day conversion but drives uninstalls at 6 months. Always pair an engagement-forcing change with a long-horizon retention guardrail.
- **Feed the loop.** Document **every** result, especially failures. Use them to (a) recalibrate baseline variance $\sigma^2$ for future power calcs, (b) validate whether driver metrics actually predict goal metrics, (c) build institutional memory so the same idea isn't re-tested in 18 months.
- **Meta-analysis across experiments** — pooling hundreds of past tests is how mature orgs learn what *categories* of change work, and how they validate their OEC.

---

## 10. 🆕 Modern Toolkit

*Everything in this section post-dates the standard 2020–2022 interview material. You don't need to implement all of it, but you should be able to explain what each is, when to reach for it, and its main trade-off. Naming two or three of these correctly is what makes a mid-level answer sound current rather than textbook.*

### 10.1 Variance reduction — the biggest practical lever

Since $n \propto \sigma^2$, **halving variance halves the required sample size** (or halves runtime). This is where modern platforms compete.

#### CUPED (Controlled-experiment Using Pre-Experiment Data)

- Introduced by Microsoft (Deng et al., 2013); now standard at Netflix, Meta, Airbnb, Booking, and built into essentially every commercial platform (Statsig, Eppo/Datadog, GrowthBook, Optimizely).
- **Mechanism:** adjust each unit's outcome using a pre-experiment covariate $X$ (usually the *same metric* measured before the experiment, which is typically its best predictor):

  $$Y_{\text{CUPED}} = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)}$$

- Variance reduction achieved: $\mathrm{Var}(Y_{\text{CUPED}}) = \mathrm{Var}(Y)\,(1 - \rho^2)$ where $\rho = \mathrm{corr}(Y, X)$. So $\rho = 0.7$ → ~50% variance reduction.
- **Unbiased**, because $E[X - \bar{X}] = 0$ by construction. Same expected lift, tighter CI.
- **Hard requirement:** the covariate must be **unaffected by treatment**. Using an in-experiment covariate that treatment influences introduces bias. Pre-period data is the safe default.
- **Estimate $\theta$ on pooled data across all arms**, not per-arm, to avoid treatment–covariate interaction contaminating the adjustment.
- **When CUPED doesn't help:** new users (no pre-period history), brand-new features/metrics, metrics with low pre/post autocorrelation, or fewer than ~2 weeks of clean pre-experiment data.
- **Typical gains:** ~20% CI reduction on many metrics; Instacart has reported a *median 66% variance reduction* on a key metric using covariate adjustment, cutting runtime by two-thirds. Marketplace benchmarks put plain CUPED around 21% CI reduction and CUPAC around 38%.
- **Equivalence to name-drop:** CUPED is closely related to **ANCOVA / regression adjustment** (Lin 2013). Modern framing (Deng et al., 2023) presents it as a general **augmentation framework** that extends to ratio and percentile metrics and can use in-experiment data too.

#### Other variance-reduction techniques

| Technique | Idea | Notes |
|---|---|---|
| **CUPAC** | CUPED with an **ML-predicted** control covariate instead of a raw pre-period metric (DoorDash) | Strongest empirical performer in recent benchmarks (~38% CI reduction) |
| **CURE / multivariate CUPED** | Multiple covariates, warehouse-native | Handles new users and non-autocorrelated metrics |
| **Stratification / blocking** | Randomize within strata (country, platform, usage decile) | Removes between-stratum variance up front; post-stratification does it at analysis time |
| **Outlier capping / winsorization** | Cap the top 0.1–1% of a heavy-tailed metric | ~37% CI reduction in some benchmarks; must be applied identically to both arms and pre-registered |
| **Triggering** | Analyze only exposed users (§8.6) | Often the largest single win |
| **Variance-reducing metric design** | Use bounded/capped metrics; log transforms; "did user do X ≥1 time" instead of raw counts | Free, but changes what you're measuring |
| **Doubly robust / DML estimators** | Combine outcome model + propensity | Marginal gains over CUPED in RCTs; more relevant for quasi-experiments |

### 10.2 Sequential testing & the peeking problem

**The problem:** repeatedly checking a fixed-horizon test and stopping when p < 0.05 inflates Type I error massively — with continuous monitoring, an A/A test will *eventually* cross the threshold with probability approaching 1.

**The solutions:**

| Approach | How it works | Used by |
|---|---|---|
| **Fixed-horizon + discipline** | Pre-commit to $n$; don't look | Still the default when traffic is ample |
| **Group Sequential Testing (GST)** | A pre-scheduled finite number of interim looks with alpha-spending boundaries (O'Brien–Fleming, Pocock) | Clinical trials; Spotify; more powerful than AVI at a fixed sample size |
| **Always Valid Inference (AVI) / confidence sequences** | p-values and CIs valid at **any** stopping time, unlimited peeking | Optimizely (Johari et al.), Eppo (GAVI), GrowthBook (asymptotic confidence sequences) |
| **mSPRT / SPRT** | Mixture sequential probability ratio test — the original always-valid test | Optimizely, various |
| **Corrected-alpha approaches** | Adjust α for the number of looks | Statsig |

**Trade-offs to state:** sequential methods trade **power for flexibility** — AVI has lower power than a fixed-horizon test at any given $n$, and requires more total samples if you run to the planned end. They also **bias effect-size estimates upward** when you stop early (you stop precisely on a lucky high reading — the "winner's curse"). Report a de-biased or shrunk estimate for shipped-early winners.

**Business value:** early stopping for **futility** is arguably more valuable than early stopping for wins — given 70–90% of experiments fail, killing losers fast is the bigger throughput unlock.

### 10.3 Bayesian A/B testing

- Reports **P(treatment > control)** and **expected loss** instead of p-values. Much easier for stakeholders to act on: *"87% probability the variant is better; expected loss if we ship and we're wrong is 0.02%."*
- Requires a **prior** (usually weakly informative or empirical-Bayes from historical experiments).
- **Genuine advantages:** natural handling of multiple arms; decision-theoretic stopping rules; intuitive communication; shrinkage across many metrics.
- **Honest caveat for interviews:** Bayesian methods are *not* automatically immune to peeking. Naive continuous monitoring of a posterior probability with a fixed threshold still inflates error rates unless the decision rule is properly constructed. Don't repeat the myth.
- Most large platforms offer **both** frequentist and Bayesian engines and let teams choose.

### 10.4 Multi-armed bandits & adaptive designs

- **Bandits** (ε-greedy, Thompson sampling, UCB) dynamically shift traffic to better-performing arms — optimizing *cumulative reward* during the test rather than *learning* a clean effect estimate.
- **Use when:** short-lived optimization (headline selection, promo creative, homepage modules), many arms, and you care about earnings-while-learning.
- **Don't use when:** you need an unbiased, well-estimated ATE for a permanent product decision, when effects take time to materialize, or when you need clean segment-level reads. Adaptive allocation breaks naive inference.
- **Contextual bandits** personalize the arm per user — the bridge to §10.6.

### 10.5 Interference, network effects & marketplace designs

**When SUTVA breaks:**

| Setting | What happens | Direction of bias |
|---|---|---|
| **Social network** | Treatment "leaks" to control friends (a treated user shares content that control users see) | **Underestimates** the true effect (control is partially treated) |
| **Two-sided marketplace** | Arms compete for the same finite supply (drivers, listings, inventory) — treatment "steals" demand from control | **Overestimates** the effect (cannibalization is counted as gain) |

**Design remedies:**

| Design | How | Where used |
|---|---|---|
| **Cluster randomization** | Randomize graph communities / clusters rather than individuals | LinkedIn, Facebook |
| **Ego-cluster randomization** | Randomize an ego + their immediate neighborhood | LinkedIn |
| **Geo randomization / region split** | Randomize cities or regions | Lyft, Uber, DoorDash, marketing lift |
| **Switchback (time-based)** | Whole system alternates between control and treatment over time windows | DoorDash, Lyft, Uber — for pricing, dispatch, matching |
| **Two-sided / multiple randomization designs** | Randomize both supply and demand sides | Airbnb, Amazon |

**Switchback specifics (worth knowing precisely):**
- Randomize treatment at the **cluster × time-period** level. Effective sample size is the number of *periods*, not users — so power is dramatically lower than user-level tests.
- **Carryover / temporal spillover** is the main threat: a ride started under treatment finishes under control. Fix with **washout periods** (discard data at the start of each window).
- Switching too fast → carryover bias. Switching too slow → too few periods → low power. That's the core bias–variance trade-off of the design.
- Airbnb-style markets where users deliberate for days are **poor** switchback candidates — the market doesn't clear fast enough.

**Validation:** always run an **A/A test under the new design**. If outcomes drift with no treatment applied, the design hasn't removed the interference.

### 10.6 Heterogeneous treatment effects (HTE / CATE)

- The ATE hides who wins and who loses. **CATE** $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ estimates the effect as a function of user features.
- **Methods:** meta-learners (**S-learner, T-learner, X-learner**, typically over LightGBM/XGBoost), **causal forests / generalized random forests** (Wager & Athey), **DR-learner**, **double machine learning (DML)**. Standard libraries: `EconML` (Microsoft), `CausalML` (Uber), `grf` (R).
- **Evaluation:** **Qini coefficient**, uplift/cumulative-gain curves, **RATE** (rank-weighted ATE), not accuracy.
- **The four-quadrant uplift framing** (very interviewable): *persuadables* (respond only if treated), *sure things*, *lost causes*, and **sleeping dogs** (treatment actively harms them — the reason blanket rollouts can be worse than targeted ones).
- **Uses:** targeted rollout, personalization policies, discovering that an average-null result masks a big win for one segment and a loss for another.
- **Caveat to state:** HTE estimates are noisy and prone to false discovery. Pre-register, validate on held-out data, and confirm discovered segments with a **follow-up confirmatory experiment**.

### 10.7 When you can't randomize — quasi-experiments

| Method | Setup | Key assumption |
|---|---|---|
| **Difference-in-Differences (DiD)** | Compare pre/post change in treated vs untreated group | **Parallel trends** — absent treatment, both groups would have moved together. Validate on pre-period |
| **Synthetic Control (SCM)** | Build a weighted composite of untreated units that reproduces the treated unit's pre-period trajectory | Good pre-period fit; donor pool not itself treated |
| **Synthetic DiD** | Hybrid; often lower MDE than either alone | As above |
| **Geo-lift** | Treat some cities/DMAs, synthesize controls from the rest | Meta's `GeoLift`; used heavily in marketing incrementality post-privacy-changes |
| **Interrupted Time Series / CausalImpact** | Bayesian structural time series counterfactual | No coincident shocks |
| **Regression Discontinuity (RDD)** | Exploit a threshold rule (e.g., loyalty tier cutoff) | Continuity at the cutoff; local effect only |
| **Instrumental Variables / encouragement design** | Randomize *encouragement* to use a feature rather than the feature itself | Valid instrument; estimates **LATE**, not ATE |
| **Propensity score matching** | Match treated to similar untreated users | No unobserved confounders — the weakest assumption of the set |

**Practical geo-lift design guidance:** aim for ≥20 geo units (10 absolute minimum), ≥25 pre-treatment periods (ideally 52 weeks to capture seasonality), daily rather than weekly granularity, and run at least one full purchase cycle. Validate with **placebo tests** — apply the method to untreated units and confirm you find no effect.

**Say the trade-off:** quasi-experiments are assumption-dependent and less powerful than RCTs. Use them when randomization is infeasible, and always run robustness/placebo checks.

### 10.8 The platform & tooling landscape (2026)

- **Commercial:** Statsig, Eppo (acquired by Datadog in 2025; integrated Datadog Experiments launched 2026), Optimizely, LaunchDarkly, Amplitude, Kameleoon, Monetate, VWO.
- **Open-source / warehouse-native:** GrowthBook, plus in-house platforms at Microsoft (ExP), Netflix (XP), Airbnb (ERF), Uber (XP), Booking.
- **The consolidation trend:** experimentation is merging with **observability** — experiment results sitting next to logs, traces, latency, and LLM telemetry, so teams can distinguish "the feature didn't work" from "the integration was broken / latency regressed."
- **Warehouse-native** (compute in Snowflake/BigQuery/Databricks rather than vendor-side) is now the default enterprise architecture — it keeps metric definitions in one place and avoids data duplication.
- **Standard feature set to expect:** frequentist + Bayesian engines, sequential testing, CUPED, multiple-testing correction, automated SRM detection, guardrails, long-running holdouts, and dimension/segment scorecards.

### 10.9 🆕 Experimentation for AI/LLM products

Increasingly asked about; a differentiator if you're interviewing at any AI-adjacent company.

- **Two-layer evaluation.** *Offline evals* (fixed golden datasets, LLM-as-judge scorers, regression suites) gate a change before it reaches users; *online A/B tests* measure real user impact. Modern platforms link the two: change a prompt/model → run against an eval set → ship as an experiment.
- **What varies:** prompt, model version, retrieval config, tool surface, decoding parameters (temperature, top-p), context window strategy, agent scaffolding.
- **Non-determinism** is the core complication. Fix seeds where possible; otherwise treat output variance as an additional variance component, which raises your required sample size.
- **Metrics that matter:** task success/completion rate, thumbs-up rate, retry/regeneration rate, conversation length (ambiguous — could mean engagement *or* failure), escalation-to-human rate, **latency and cost per request** (near-universal guardrails), and safety/refusal rates.
- **Cost is a first-class guardrail** in a way it isn't for UI tests — a better model that triples inference cost may not be shippable.
- **Novelty effects are severe** for AI features; long-term holdouts matter more, not less.
- **Agent-based simulation** (LLM agents simulating users to pre-screen designs) is an emerging complement for low-traffic surfaces — directionally useful for triage, **not** a replacement for real-user experimentation.
- **Governance:** in regulated contexts, experiment runs are expected to emit audit artifacts (model cards, dataset cards, evaluation summaries) as a byproduct.

### 10.10 🆕 Privacy & measurement constraints

- Cookie deprecation, ATT/IDFA, and consent regimes have degraded individual-level tracking, which is precisely why **geo-lift and synthetic-control methods have surged** in marketing measurement.
- **Consent-mode and privacy-by-design** architectures mean some traffic is unmeasurable — check that consent status doesn't differ across arms (a subtle SRM source).
- Server-side / full-stack testing is now preferred over client-side DOM manipulation: better tracking fidelity, no flicker, and it lets you test architecture rather than just UI.

---

## 11. Pitfalls, Traps & Twyman's Law

> **Twyman's Law:** *Any figure that looks interesting or different is usually wrong.* A +30% revenue lift is a bug until proven otherwise.

| Pitfall | What goes wrong | Defense |
|---|---|---|
| **Peeking / early stopping** | Type I error inflates far above 5% | Fixed horizon, or a proper sequential method |
| **SRM ignored** | Entire result is biased | Automated chi-square check, p < 0.0005 |
| **HARKing / p-hacking** | Slice until something is significant | Pre-register OEC, hypothesis, segments, and duration |
| **Multiple comparisons uncorrected** | 20 metrics → 64% chance of a false win | BH-FDR on secondary panel |
| **Simpson's paradox** | Aggregating across ramp phases or unequal segment mixes reverses the sign | Analyze one stable allocation period; segment-weight consistently |
| **Novelty / primacy** | Two-week read doesn't generalize | Effect-by-day plots; new vs existing user split; long-term holdout |
| **Dilution** | Untriggered users wash out the effect | Trigger analysis with counterfactual triggering in control |
| **Randomization ≠ analysis unit** | Understated variance, false positives | Delta method / clustered SEs |
| **Survivorship in logging** | Crashed treatment users never log outcomes | Instrument exposure before the risky code path; check SRM |
| **Outliers / whales** | One user's \$50K order flips the result | Winsorize (pre-registered), bootstrap, report both |
| **Metric gaming** | Team optimizes the proxy, destroys the goal | Pair every driver metric with goal + guardrail |
| **Change aversion mistaken for failure** | Kill a good feature too early | Longer runtime; new-user analysis |
| **Interaction with concurrent tests** | Two experiments touch the same surface | Layered design; interaction scans |
| **Assuming p > 0.05 means "no effect"** | Underpowered null read reported as flat | Report the CI; check whether MDE was achievable |
| **Carryover from a previous experiment** | Buckets not re-randomized; residual effects | Re-randomize between experiments; run A/A |
| **Seasonality / external shocks** | Holiday spike attributed to the feature | Full-week runs; check both arms move together |
| **Bot traffic** | Can be >50% of raw traffic on some properties | Filter identically across arms, before analysis |

---

## 12. Interview Question Bank

*Answer structure for open-ended design questions: **Clarify → Metrics → Hypothesis → Design → Analysis → Decision → Risks.** Spend ~20% of your time clarifying. Always state assumptions out loud.*

### 12.1 Conceptual

**Q: What is statistical power, and what affects it?**
Power = P(detect a true effect of at least the MDE) = $1-\beta$. It increases with larger sample size, larger true effect, lower metric variance, higher α, and a one-sided rather than two-sided test. Industry default is 80%.

**Q: Explain Type I vs Type II error in business terms.**
Type I: ship something that does nothing — you pay permanent maintenance and tech-debt cost for zero return. Type II: kill something that works — you forgo the upside. Which is costlier depends on the change; a cheap UI tweak tolerates Type I risk far better than a costly platform migration.

**Q: What does a p-value of 0.03 mean?**
If the null were true, there's a 3% chance of seeing a difference this large or larger. It is *not* a 97% probability the feature works, and it says nothing about effect size — which is why I'd report the confidence interval alongside it.

**Q: Your test isn't significant. Is there no effect?**
Not necessarily. Non-significance means insufficient evidence, which is often insufficient power. I'd look at the CI: if it's narrow and tightly around zero, that's a real null and useful information. If it's wide and includes effects larger than the MDE, the test was underpowered and I'd either extend it, apply variance reduction, or accept that we can't resolve an effect this small at our traffic.

**Q: Why run for two weeks instead of stopping at day 3 when it hit significance?**
Three reasons: (1) peeking inflates the false-positive rate well above 5%; (2) day-of-week effects mean a partial week isn't representative; (3) novelty effects make early reads systematically optimistic. If we genuinely need early stopping, I'd design for it up front with a group sequential test or always-valid inference rather than peeking at a fixed-horizon test.

**Q: How do you pick the MDE?**
It's a business decision, not a statistical one. I'd ask product and finance: what's the smallest lift that would justify shipping, given engineering cost, maintenance, and opportunity cost? Then I check feasibility — if that MDE requires 6 months of traffic, we renegotiate: accept a larger MDE, apply CUPED, use a more sensitive driver metric, or don't test it.

**Q: When would you not run an A/B test?**
Insufficient traffic; the change affects everyone (pricing, ToS, brand); strong network/marketplace interference; outcomes that take months to realize; or ethical/legal barriers to withholding. In each case I'd name the alternative — quasi-experiment, cluster/switchback design, surrogate metric, or staged rollout with monitoring.

### 12.2 Design

**Q: How would you test a new recommendation algorithm on a marketplace?**
Clarify whether supply is constrained. If yes, user-level randomization violates SUTVA — treatment users get better recommendations and consume inventory that control users would have booked, so I'd *overestimate* the effect via cannibalization. I'd propose geo/cluster randomization or a switchback design depending on how fast the market clears, accept the power hit, validate with an A/A test under the new design, and use OEC = bookings per user with supply utilization and seller-side metrics as guardrails.

**Q: Test adding a "buy now" button to product pages.**
- *Clarify:* placement, all users or a segment, mobile/web, does it skip cart?
- *Metrics:* OEC = orders per user (or revenue per user). Drivers: PDP→purchase conversion, time-to-purchase. Guardrails: AOV (fast checkout may reduce basket size), return/refund rate, cart abandonment, latency.
- *Design:* user-level randomization, 50/50, trigger on PDP view. Power for a 1% relative lift on conversion → compute $n$ via rule of 16 → runtime rounded to full weeks.
- *Analysis:* delta method if I analyze per-session; segment by new/returning and platform; BH-FDR on the secondary panel.
- *Risk:* cannibalization of cart-based multi-item orders — that's exactly why AOV is a guardrail, and it's the trade-off I'd monetize if the two conflict.

**Q: How would you test a change to push-notification frequency?**
Flag the horizon problem immediately: this is a change where short-term conversion and long-term retention diverge. Two-week test on conversion is not sufficient. I'd run at least 4 weeks, make uninstall rate / notification opt-out / DAU-at-30-days explicit guardrails, and set up a long-term holdout to measure the 3–6 month retention effect before a full rollout.

**Q: Your randomization is by user but the metric is per-session. Problem?**
Yes — the analysis unit is finer than the randomization unit, so sessions within a user are correlated. A naive t-test treating sessions as independent understates variance and inflates false positives. Fix with the delta method, or equivalently OLS with cluster-robust standard errors at the user level.

### 12.3 Diagnostics & troubleshooting

**Q: You designed a 50/50 split but observe 50.6/49.4 on 2M users. What now?**
Run a chi-square goodness-of-fit test with df = 1. At that sample size, that imbalance is almost certainly significant at p < 0.0005 — an SRM. I'd stop and not interpret results. Then I'd segment the ratio by day, platform, browser, country, and new-vs-returning to localize the bug, and check the usual suspects: asymmetric bot/error filtering, treatment-side crashes suppressing logs, redirect drop-off, pipeline join loss, or bucket carryover from a prior experiment.

**Q: Engagement is up 4% but ad revenue is down 1.5%. Ship?**
First, back to the pre-registered OEC — that's what it's for. If the OEC doesn't settle it, monetize both: convert the engagement lift into expected retention/LTV value and compare directly to the ad revenue loss, stating my assumptions. I'd also check whether the revenue loss is concentrated in a segment we could exclude, and whether the trade is a deliberate long-term investment we should validate with a holdout. Then I'd give a clear recommendation with the CIs and the net number, not just present the tension.

**Q: Your treatment shows +25% revenue. Reaction?**
Twyman's Law — I'd assume it's a bug first. Check SRM, look for outliers/whales driving it, verify logging symmetry, check whether the effect is concentrated in one day or one segment, confirm the metric definition is identical across arms, and run an A/A on the same buckets. Only after all of that would I believe a lift that large.

**Q: The effect was +3% in the test but only +0.5% after launch. Why?**
Candidate explanations: novelty decay; primacy in the opposite direction; the test population was triggered-only while launch is diluted across everyone; winner's curse from stopping early or from selecting the best of many variants; seasonality differences; or interference that inflated the test estimate. I'd diagnose with the long-term holdout and by comparing triggered vs overall populations.

### 12.4 Modern-topic questions (increasingly common)

**Q: How would you make an underpowered test feasible?**
In order of leverage: (1) trigger analysis — restrict to exposed users; (2) CUPED or regression adjustment using pre-period data, typically 20%+ variance reduction; (3) outlier capping on heavy-tailed metrics; (4) stratified randomization; (5) switch to a more sensitive driver metric with the goal metric as a guardrail; (6) increase allocation to 50/50; (7) accept a larger MDE; (8) sequential design to stop early on futility. Only after those would I extend runtime.

**Q: What is CUPED and when does it fail?**
[See §10.1.] It fails for new users with no pre-period, for brand-new metrics, when pre/post correlation is weak, with <2 weeks of clean history, and — critically — it becomes *biased* if the covariate is influenced by the treatment.

**Q: Frequentist or Bayesian?**
Depends on the org and the decision. Frequentist is the default, well-understood, and easy to standardize across hundreds of tests. Bayesian communicates better to stakeholders (P(better), expected loss) and handles many arms naturally, but needs a prior and — contrary to a common myth — is not automatically peeking-safe. Most platforms offer both; I'd match whatever the team has standardized on.

**Q: Can you stop a test early?**
Only if you designed for it. Group sequential testing pre-schedules interim looks with alpha-spending boundaries; always-valid inference / confidence sequences permit unlimited peeking at a power cost. Both bias the effect estimate upward when you stop early, so I'd report a shrunk estimate. Stopping early for *harm* on a guardrail is always allowed regardless of design.

---

## 13. Cheat Sheet — Formulas & Numbers

### Numbers to have memorized

| Quantity | Value |
|---|---|
| $z_{1-\alpha/2}$ at α = 0.05 (two-sided) | 1.96 |
| $z_{1-\beta}$ at 80% power | 0.84 |
| $z_{1-\beta}$ at 90% power | 1.28 |
| $2(1.96+0.84)^2$ | 15.68 ≈ **16** |
| $2(1.96+1.28)^2$ | 21.0 (for 90% power) |
| SRM alert threshold | p < 0.0005 |
| Standard α / β | 0.05 / 0.20 |
| Typical minimum runtime | 7 days; 14 days preferred |
| Experiment success rate in mature programs | 10–33% |

### Formulas

**Sample size per group (means):**
$$n \approx \frac{2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2}{\delta^2} \;\;\xrightarrow{\alpha=0.05,\,1-\beta=0.8}\;\; n \approx \frac{16\sigma^2}{\delta^2}$$

**Sample size per group (proportions):**
$$n \approx \frac{16\,p(1-p)}{\delta^2}$$

**Two-proportion z-test:**
$$z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_T}+\frac{1}{n_C}\right)}}, \qquad \hat{p} = \frac{x_T + x_C}{n_T + n_C}$$

**Welch's t-statistic:**
$$t = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}$$

**95% CI for a difference in means:**
$$(\bar{X}_T - \bar{X}_C) \pm 1.96\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}$$

**SRM chi-square (2 arms, df = 1):**
$$\chi^2 = \frac{(N_C - E)^2}{E} + \frac{(N_T - E)^2}{E}, \qquad E = \frac{N_C+N_T}{2}$$

**CUPED-adjusted metric and its variance:**
$$Y_{\text{cuped}} = Y - \theta(X - \bar{X}),\quad \theta = \frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},\quad \mathrm{Var}(Y_{\text{cuped}}) = \mathrm{Var}(Y)(1-\rho^2)$$

**Delta method for a ratio $R = \bar{X}/\bar{Y}$:**
$$\mathrm{Var}(R) \approx \frac{1}{n}\left(\frac{\mathrm{Var}(X)}{\bar{Y}^2} - \frac{2\bar{X}\mathrm{Cov}(X,Y)}{\bar{Y}^3} + \frac{\bar{X}^2\mathrm{Var}(Y)}{\bar{Y}^4}\right)$$

**Family-wise error rate with $m$ independent tests:**
$$\text{FWER} = 1-(1-\alpha)^m$$

**Benjamini–Hochberg:** sort p-values ascending; find the largest $k$ such that $p_{(k)} \le \frac{k}{m}q$; reject $H_{(1)} \dots H_{(k)}$.

---

## 14. Corrections Applied to the Original Study Guide

*Fixes made to the source `ab_testing_study_guide.md`, listed so you don't re-learn the errors.*

| # | Original claim | Correction |
|---|---|---|
| 1 | "Setting a smaller MDE requires a**n exponentially** larger sample size" | **Quadratic (inverse-square)**, not exponential. $n \propto 1/\delta^2$: halving MDE quadruples $n$. This is a commonly caught error |
| 2 | "The standard industry threshold is **strictly capped** at α = 0.05" | 0.05 is a **convention**, not a cap. It's tightened under multiple-testing correction and loosened (0.10) for cheap exploratory tests. Presenting it as a hard rule is a red flag |
| 3 | "Secure Adequate Sample **Volatility**… a baseline standard requires **thousands** of units" | Terminology error (should be *volume*), and "thousands" is not a real standard. Required $n$ comes from the power calculation; typical web tests need tens of thousands to millions per arm |
| 4 | "the **total** required sample size ($n$) **per group**" | Self-contradictory. The formula gives $n$ **per group**; total = $n \times$ number of arms |
| 5 | Sample-size formula given without the α/β shortcut | Added **Kohavi's rule of 16** ($n \approx 16\sigma^2/\delta^2$) and the proportions form $\sigma^2 = p(1-p)$ — the version you'd actually use on a whiteboard |
| 6 | MDE used in the denominator without units clarified | Clarified **absolute vs relative** MDE. Nearly all sample-size mistakes in interviews come from mixing these up |
| 7 | SRM threshold stated as p < 0.001 only | Industry standard is **p < 0.0005** at most large platforms (0.001 also seen). Added df = $k-1$ for $k$ arms |
| 8 | Decision matrix based on point estimate vs MDE | Rewritten as a **6-case confidence-interval framework**. The original 4-case table can't distinguish "confidently flat" from "underpowered," which are completely different decisions |
| 9 | "Underpowered Test… High variance / Large observed mean shift" | Reframed: you diagnose underpowered tests by a **wide CI extending beyond the MDE**, not by the point estimate |
| 10 | "Novelty… this initial curiosity **fade**" | Typo; also expanded with **how to detect** novelty/primacy (effect-by-day curves, new vs existing users) |
| 11 | Chi-square SRM formula given, but no root-cause list | Added the full diagnostic list — this is what interviewers actually probe |
| 12 | Trailing disclaimer: *"For medical advice or diagnosis, consult a professional"* | Removed — irrelevant boilerplate that appears to be a copy-paste artifact |
| 13 | LaTeX rendering errors (`$lpha$`, `$eta$`, `$	ext{...}$`) | Fixed — the backslashes were stripped from `\alpha`, `\beta`, `\text{}` |

### Major gaps that were missing entirely and have been added

`A/A tests` · `guardrail metrics & metric hierarchy (goal/driver/guardrail)` · `randomization unit vs analysis unit` · `ratio metrics, delta method, clustered SEs` · `triggering & dilution` · `multiple-testing correction (Bonferroni/Holm/BH-FDR)` · `network effects & SUTVA violations` · `cluster / geo / switchback designs` · `variance reduction (CUPED, CUPAC, stratification, winsorization)` · `sequential testing & the peeking problem` · `Bayesian methods & bandits` · `HTE/CATE & uplift modeling` · `quasi-experiments (DiD, synthetic control, geo-lift, RDD, IV)` · `Simpson's paradox` · `long-term holdouts` · `opportunity sizing & prioritization` · `AI/LLM product experimentation`

---

## 15. Glossary & Resources

### Glossary

| Term | Definition |
|---|---|
| **ATE / CATE / LATE** | Average / Conditional Average / Local Average Treatment Effect |
| **AVI** | Always Valid Inference — p-values valid under unlimited peeking |
| **CUPED** | Controlled-experiment Using Pre-Experiment Data — variance reduction via pre-period covariate |
| **CUPAC** | CUPED with an ML-predicted control covariate |
| **Delta method** | Taylor-expansion approximation for the variance of a ratio |
| **Dilution** | Effect shrinkage from including users who never saw the change |
| **FDR** | False Discovery Rate — $E[\text{false positives}/\text{rejections}]$ |
| **FWER** | Family-Wise Error Rate — P(≥1 false positive across a family of tests) |
| **GST** | Group Sequential Testing — pre-scheduled interim looks with alpha spending |
| **Guardrail** | Metric that must not degrade; may be a trust invariant or a business constraint |
| **HTE** | Heterogeneous Treatment Effect |
| **MDE** | Minimum Detectable Effect |
| **Novelty effect** | Temporary engagement lift caused by newness |
| **OEC** | Overall Evaluation Criterion — the single success metric |
| **Primacy effect** | Temporary degradation from user resistance to change (change aversion) |
| **Qini** | Evaluation metric for uplift/CATE model ranking quality |
| **SRM** | Sample Ratio Mismatch |
| **SUTVA** | Stable Unit Treatment Value Assumption (no interference, no hidden treatment variants) |
| **Switchback** | Time-sliced randomization of an entire system |
| **Triggering** | Restricting analysis to users who actually reached the changed code path |
| **Twyman's Law** | Any figure that looks interesting is usually wrong |
| **Winner's curse** | Upward bias in effect size when selecting/stopping on the best result |

### Resources

**Books**
- *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing* — Kohavi, Tang, Xu. The single canonical reference; chapters 1–4, 6–7, 17–23 map almost exactly onto this document.
- *Statistical Methods in Online A/B Testing* — Georgi Georgiev.
- *Experimentation Works* — Stefan Thomke (organizational/cultural angle).

**Courses & channels**
- Emma Ding / Data Interview Pro — YouTube A/B testing playlist and the A/B testing cheat sheet at emmading.com. Best interview-shaped framing.
- Udacity A/B Testing (by Google) — free, classic, still the best structured intro.

**Engineering blogs worth skimming before an interview**
- Microsoft ExP (patterns of trustworthy experimentation), Netflix TechBlog, Airbnb, Uber, DoorDash (switchback), LinkedIn (network interference), Spotify Engineering (sequential testing), Booking.com.
- Platform docs are unusually good primary sources for modern methods: Statsig, Eppo/Datadog, GrowthBook, Optimizely.

**Key papers, if you want depth**
- Deng et al. (2013) — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED).
- Deng et al. (2018) — *Applying the Delta Method in Metric Analytics*.
- Johari et al. (2017/2022) — *Peeking at A/B Tests* / *Always Valid Inference*.
- Wager & Athey (2018) — *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*.
- Holtz et al. (2025) — *Reducing Interference Bias in Online Marketplace Experiments Using Cluster Randomization* (Airbnb).
- Deng et al. (2023) — *From Augmentation to Decomposition: A New Look at CUPED*.

---

### Study plan (2 weeks before an interview)

| Day | Focus |
|---|---|
| 1–2 | §1–§5. Be able to recite the 7 steps and do a sample-size calculation from memory |
| 3–4 | §6–§8. SRM, delta method, multiple testing, triggering |
| 5 | §9. Practice the decision matrix on 5 fake result sets |
| 6–7 | §10.1–10.3. CUPED, sequential testing, Bayesian — enough to explain and critique |
| 8 | §10.5–10.7. Interference, HTE, quasi-experiments |
| 9–10 | §12. Say the answers out loud. Record yourself. Cut the filler |
| 11–12 | Product-sense practice: 10 "how would you test X" prompts, 15 min each, full framework |
| 13–14 | §11 + §13. Pitfalls and formula recall. Skim Kohavi ch. 1–4 |

---

*Notes compiled August 2026. Sources: Emma Ding / Data Interview Pro; Kohavi, Tang & Xu; Microsoft ExP; Spotify, Netflix, DoorDash, LinkedIn, Airbnb and Instacart engineering publications; Statsig / Eppo / GrowthBook / Optimizely documentation; and recent arXiv/KDD work on variance reduction, sequential testing, and interference.*
