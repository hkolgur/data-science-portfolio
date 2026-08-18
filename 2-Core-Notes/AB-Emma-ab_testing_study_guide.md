# Comprehensive A/B Testing Study Guide & Interview Checklist
*Based on Emma Ding's Fundamentals Curriculum and Industry Best Practices*

---

## 1. Experiment Prerequisites & Core Mechanics
Before launching an experiment, a data scientist must verify that the business problem and engineering infrastructure meet the structural prerequisites for a trustworthy online controlled experiment.

### Key Concepts & Definitions
*   **A/B Test / Controlled Experiment**: An experiment in which all elements are held constant except for one variable. It formally compares a control group (current version) against a treatment group (new feature or variant) to establish a clear causal relationship.
*   **Variants**: Different versions of a product experience. These can range from minor UI modifications (e.g., changing the color of a checkout button) to highly complex back-end alterations (e.g., structural changes to search ranking or recommendation algorithms).
*   **Randomization Unit**: The specific entity (the "who" or "what") that is uniquely and randomly allocated to either the control or treatment group. While individual **users** are the most common unit, other examples include sessions, page views, devices, or geographic regions.

### Critical Prerequisites Checklist
*   [ ] **Define the Overall Evaluation Criteria (OEC):** Establish a single, quantitative metric or a heavily weighted composite index that serves as the gold standard for success. The OEC must be aligned upon by cross-functional stakeholders and be reliably measurable (e.g., *Revenue per user per month*).
*   [ ] **Ensure Engineering Ease of Change:** Changes must be simple enough to cleanly isolate and deploy as separate variants. High-complexity monolithic changes (e.g., redesigning an entire e-commerce ecosystem all at once) introduce too many compounding variables, making it impossible to isolate causal factors.
*   [ ] **Secure Adequate Sample Volatility:** The experiment environment must yield a massive volume of randomization units. A baseline standard requires **thousands** of units; a larger sample size naturally drives down standard error, enabling the test to confidently detect small, incremental metric shifts.

---

## 2. Statistical Design & Power Analysis
Designing the experiment requires translating a business question into an isolated statistical framework, ensuring the experiment runs long enough to bypass environmental noise.

### Hypothesis Formulation
Every experiment evaluates two competing, mutually exclusive statistical positions:
*   **Null Hypothesis ($H_0$):** The variant and the control experience have identical impacts on the OEC; any observed difference is purely due to random sampling noise ($\mu_{treatment} - \mu_{control} = 0$).
*   **Alternative Hypothesis ($H_1$):** The variant creates a genuine, mathematically distinct impact on the OEC ($\mu_{treatment} - \mu_{control} \neq 0$).

### Core Statistical Parameters
*   **Significance Level ($lpha$):** The probability of committing a Type I error (rejecting the null hypothesis when it is actually true—a "false positive"). The standard industry threshold is strictly capped at $lpha = 0.05$. If Engineering cost is high to conduct test then business may keep alpha little high vs if the error is very critical then alpha will be set to very low.
*   **Statistical Power ($1-eta$):** The probability of correctly rejecting the null hypothesis when a true effect exists (avoiding a Type II "false negative"). The standard industry baseline is targeted at $80\%$ ($eta = 0.20$).
*   **Minimum Detectable Effect (MDE):** MDE is smallest change your test can reliably spot based on your sample size and noise level..The smallest metric lift or absolute shift that the business considers practically meaningful to detect. Setting a smaller MDE requires an exponentially larger sample size to achieve sufficient statistical power.
   - What is MDE?
         1. It is a statistical number.
         2. It shows the smallest change your test can find.
         3. It depends on your sample size and data noise
   - If you cut MDE in half , we need 4x sample size. (1%-5% for high traffic sites, Email and Messaging  Range: 8% to 12%)

### Sample Size Calculation Template
For a standard two-sample t-test comparing two proportions with equal variance, the total required sample size ($n$) per group is calculated using the following mathematical architecture:

$$n \approx \frac{2 \sigma^2 (Z_{1-\alpha/2} + Z_{1-\beta})^2}{\text{MDE}^2}$$

Where:
*   $\sigma^2$ represents the historical baseline variance of the core metric.
*   $Z_{1-\alpha/2}$ represents the critical value for a two-tailed significance threshold (at $lpha = 0.05$, $Z_{0.975} = 1.96$).
*   $Z_{1-\beta}$ represents the critical value for the targeted statistical power (at $1-eta = 0.80$, $Z_{0.80} = 0.84$).

### Runtime & Duration Factors
Never stop an experiment early the moment it crosses a significance boundary. The total runtime duration must be predetermined and strictly adhere to:
*   **Seasonality / Day-of-Week Effects:** Experiments must run in full-week increments (typically **14 to 28 days**) to capture cyclic weekly patterns. User behavior on Monday afternoon completely differs from user behavior on Saturday night; ending a test at 10 days introduces fatal bias.
*   **Novelty Effects:** Users frequently interact with a new feature heavily at first simply because it is novel. Over a few weeks, this initial curiosity fade. Running a test over an extended duration isolates true long-term behavioral changes from temporary novelty spikes.
*   **Primacy Effects:** Opposite to novelty, users may initially resist or struggle with a changed interface because they are habituated to the old system. A longer runtime gives users space to learn the new workflow.

---

## 3. Running the Experiment & Validity Checks
Data collection is an active engineering process. Before looking at p-values, you must perform deep validation checks to prove the underlying data collection process is completely uncompromised.

### Engineering Instrumentation
*   **Log Verification:** Coordinate heavily with platform engineers to audit data pipelines. Ensure that tracking tags fire accurately the exact millisecond a user is exposed to a variant, and that downstream event logs capture interactions without data loss.

### Sample Ratio Mismatch (SRM) Analysis
An SRM is the ultimate indicator of a fundamentally flawed experiment design or corrupted deployment pipeline. It occurs when the actual observed sample split between your groups deviates significantly from the intended design allocation (e.g., a planned 50/50 split ends up as 49.2/50.8 over a massive sample).

#### How to Mathematically Detect SRM (The Chi-Square Goodness-of-Fit Test)
If an experiment plans a perfect 50/50 allocation split, and collects $N_{control}$ and $N_{treatment}$ samples, we evaluate whether the deviation is due to pure chance:

1.  **Calculate Expected Counts:** 
    $$E_{control} = E_{treatment} = \frac{N_{control} + N_{treatment}}{2}$$
2.  **Compute the Chi-Square ($\chi^2$) Statistic:**
    $$\chi^2 = \sum \frac{(O - E)^2}{E} = \frac{(N_{control} - E_{control})^2}{E_{control}} + \frac{(N_{treatment} - E_{treatment})^2}{E_{treatment}}$$
3.  **Evaluate:** Compare the result against a Chi-Square distribution with $1$ Degree of Freedom. If the resulting $p	ext{-value} < 0.001$, an SRM is officially confirmed. You must discard the results, investigate the log architecture, and find the root deployment error.

---

## 4. Analyzing Results & Making Launch Decisions
Once data integrity is verified, a data scientist must transform raw statistical outputs into a definitive business recommendation.

### The Decision Matrix Framework
When moving from statistical results to an active launch decision, use a strict dual-boundary evaluation model:

| Scenario | Statistical Significance ($p < 0.05$) | Practical Significance ($\Delta > 	ext{MDE}$) | Strategic Product Recommendation |
| :--- | :--- | :--- | :--- |
| **Ideal Winner** | **Yes** (Confident effect) | **Yes** (Outweighs all overhead) | **Launch Immediately:** The feature provides an uncompromised, high-value return that easily justifies deployment. |
| **Statistical Noise** | **No** (Cannot reject $H_0$) | **No** (Negligible movement) | **Do Not Launch:** Retain the baseline control. The proposed change demonstrates no proven value. |
| **Underpowered Test** | **No** (High variance) | **Yes** (Large observed mean shift) | **Inconclusive:** The mean shift looks promising, but high noise limits statistical confidence. Increase sample size or re-run the test. |
| **The "Tricky" Case** | **Yes** (Confident effect) | **No** (Below economic utility boundary) | **Do Not Launch:** Commonly occurs in massive samples where tiny, irrelevant metric changes become statistically significant. The benefit does not cover the overhead. |

### Navigating Complex Trade-offs
*   **Conflicting Metrics:** In real-world tech environments, metrics regularly move in completely opposite directions (e.g., User App Engagement rises by $+4\%$, but Ad Click Revenue drops by $-1.5\%$). To resolve this, map both movements back to the single predefined **OEC** or assign concrete financial values to both movements to see if the net business impact is positive.
*   **Engineering Maintenance Overhead:** Every single line of additional code introduces permanent technical debt, complex code dependencies, and potential performance bugs. If a feature's metric lift is positive but fails to clear the practical significance boundary, the long-term cost of maintaining that code outweighs the minor optimization gain.
*   **Opportunity Costs:** Deploying engineering squads to launch and maintain a low-impact feature strips those resources away from building entirely new, high-impact product ideas.

---

## 5. Post-Launch Monitoring & Continuous Iteration
A definitive launch decision does not conclude the responsibilities of a data scientist. Long-term validation ensures the experimental results accurately scale.

### Post-Launch Verifications
*   **Short-Term vs. Long-Term Divergence:** Track users continuously over 3, 6, and 12-month cohorts post-launch. It is incredibly common for an initial metric lift to completely decay over time as the novelty wears off or as competitors adapt.
*   **User Fatigue Analysis:** Continually monitor for negative secondary behavioral shifts. For instance, increasing the volume of push notifications might show a short-term conversion spike in a 14-day test, but create severe user fatigue and accelerate long-term app uninstalls over a 6-month period.
*   **Feeding the Experimentation Loop:** Every experimental outcome—especially complete failures—must be meticulously documented. Use these insights to re-calibrate your primary metric definitions, adjust historical variance baselines ($\sigma^2$), and formulate more precise hypotheses for future optimization iterations.

---
*Disclaimer: This is for informational purposes only. For medical advice or diagnosis, consult a professional. AI responses may include mistakes.*
