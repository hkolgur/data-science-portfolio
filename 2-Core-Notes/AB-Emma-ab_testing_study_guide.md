# Comprehensive A/B Testing Study Guide & Interview Checklist
*Based on Emma Ding's Fundamentals Curriculum and Industry Best Practices*

---

## 1. Experiment Prerequisites & Core Mechanics
Before launching an experiment, a data scientist must verify that the business problem and engineering infrastructure meet the structural prerequisites for a trustworthy online controlled experiment.
-Questions to ask Before any A/B Test.
   a. How sample population look like and what are the customer segments for target product
   b. For current problem can we find a solution using Data Analysis/Historical data etc
   c. Do we want to test single or multiple variants
   d. Can we truly randomize Control and Treatment groups without bias
   e. During entire duration of test can we ensure integrity of control and varient

### Key Concepts & Definitions
*   **A/B Test / Controlled Experiment**: An experiment in which all elements are held constant except for one variable. It formally compares a control group (current version) against a treatment group (new feature or variant) to establish a clear causal relationship.
*   **Variants**: Different versions of a product experience. These can range from minor UI modifications (e.g., changing the color of a checkout button) to highly complex back-end alterations (e.g., structural changes to search ranking or recommendation algorithms).
*   **Randomization Unit**: The specific entity (the "who" or "what") that is uniquely and randomly allocated to either the control or treatment group. While individual **users** are the most common unit, other examples include sessions, page views, devices, or geographic regions.

### Critical Prerequisites Checklist
*   [ ] **Define the Overall Evaluation Criteria (OEC):** Establish a single, quantitative metric or a heavily weighted composite index that serves as the gold standard for success. The OEC must be aligned upon by cross-functional stakeholders and be reliably measurable. Example metrics:  , 
   - *Revenue per user per month*
   - Click-Through Rate (CTR) for usage CTR =  # total_clicks * 100 % / (# total_clicks +  # total_views)
   -  Click-Through Probability (CTP) for impact CTP= #people with at least 1 click * 100% / #unique visitors per page   . In CTP multiple clicks are counted as one. ex due to impatience if user clicks button multiple times in same session
   -  Conversion Rate CR = # converted /(#converted + #notconverted)

     
*   [ ] **Ensure Engineering Ease of Change:** Changes must be simple enough to cleanly isolate and deploy as separate variants. High-complexity monolithic changes (e.g., redesigning an entire e-commerce ecosystem all at once) introduce too many compounding variables, making it impossible to isolate causal factors.
*   [ ] **Secure Adequate Sample Volatility:** The experiment environment must yield a massive volume of randomization units. A baseline standard requires **thousands** of units; a larger sample size naturally drives down standard error, enabling the test to confidently detect small, incremental metric shifts.

---

## 2. Statistical Design & Power Analysis
Designing the experiment requires translating a business question into an isolated statistical framework, ensuring the experiment runs long enough to bypass environmental noise.

### Hypothesis Formulation
Every experiment evaluates two competing, mutually exclusive statistical positions:
For Sample size calculation with Continuous Metric:
*   **Null Hypothesis ($H_0$):** The variant and the control experience have identical impacts on the OEC; any observed difference is purely due to random sampling noise ($\mu_{treatment} - \mu_{control} = 0$).
*   **Alternative Hypothesis ($H_1$):** The variant creates a genuine, mathematically distinct impact on the OEC ($\mu_{treatment} - \mu_{control} \neq 0$).

For Sample size calculation with Binay Metric:
*   **Null Hypothesis ($H_0$):** The variant and the control experience have identical impacts on the OEC; any observed difference is purely due to random sampling noise ($p_{treatment} - p_{control} = 0$).
*   **Alternative Hypothesis ($H_1$):** The variant creates a genuine, mathematically distinct impact on the OEC ($p_{treatment} - p_{control} \neq 0$).
  
### Core Statistical Parameters- Power Analysis 
*   **Significance Level ($lpha$):** The probability of committing a Type I error (rejecting the null hypothesis when it is actually true—a "false positive"). The standard industry threshold is strictly capped at $lpha = 0.05$. If Engineering cost is high to conduct test then business may keep alpha little high vs if the error is very critical then alpha will be set to very low.
*   **Statistical Power ($1-eta$):** The probability of correctly rejecting the null hypothesis when a true effect exists (avoiding a Type II "false negative"). The standard industry baseline is targeted at $80\%$ ($eta = 0.20$).
*   **Minimum Detectable Effect (MDE):** MDE is smallest change your test can reliably spot based on your sample size and noise level..The smallest metric lift or absolute shift that the business considers practically meaningful to detect. Setting a smaller MDE requires an exponentially larger sample size to achieve sufficient statistical power.
   - What is MDE?
         1. It is a statistical number.
         2. It shows the smallest change your test can find.
         3. It depends on your sample size and data noise
   - If you cut MDE in half , we need 4x sample size. (1%-5% for high traffic sites, Email and Messaging  Range: 8% to 12%)
 
**Power Analysis**
* **$\alpha$:** Probability of Type I Error, Significance Level (False Positive).False ALaram
* **$\beta$:** Probability of Type II Error (False Negative). Missed opportunity
* **$(1 - \beta)$:** Power of the test
* **$\delta$:** Minimum Detectable Effect

### Sample Size Calculation Template
1. Primary Metric is in form of Proportions or Averages.Eg. Mean Order Amount , Mean CLick Thru Rate .
#### Sample Size Calculation for Proportions/Averages 

when we use means , we use Central Limit Theorem . Mean of the sample sizes drawn follow normal distribution for both control and treatment group and hence the difference of means also follow normal distribution.

$$\bar{X}_{con} \sim N(\mu_{con}, \sigma_{con}^2)$$

$$\bar{X}_{exp} \sim N(\mu_{exp}, \sigma_{exp}^2)$$

$$\bar{X}_{con} - \bar{X}_{exp} \sim N\left(\mu_{con} - \mu_{exp}, \frac{\sigma_{con}^2}{N_{con}} + \frac{\sigma_{exp}^2}{N_{exp}}\right)$$

\sigma_{exp}^2 and \sigma_{con}^2 can be obtained from A/A testing. We already have alpha, Beta, delta.

For a standard two-sample t-test comparing two proportions with equal variance, the total required sample size ($n$) per group is calculated using the following mathematical architecture:

$$n \approx \frac{2 \sigma^2 (Z_{1-\alpha/2} + Z_{1-\beta})^2}{\text{MDE}^2}$$

Where:
*   $\sigma^2$ represents the historical baseline variance of the core metric.
*   $Z_{1-\alpha/2}$ represents the critical value for a two-tailed significance threshold (at $lpha = 0.05$, $Z_{0.975} = 1.96$).
*   $Z_{1-\beta}$ represents the critical value for the targeted statistical power (at $1-eta = 0.80$, $Z_{0.80} = 0.84$).

2. Primary Metric is Binay. Eg.Conversion or No-conversion ,Click or No-Click,
   
   #### Sample Size Calculation for Binomial Proportions (CTR)

When a metric has two outcomes (like Click vs. No Click), we treat user responses as independent Bernoulli Trials. The click events follow a binomial distribution based on sample size (impressions) and success probabilities ($p_{con}$ for Control, $p_{exp}$ for Experimental).

To find the required sample size per group ($N$) for a two-sided test with a chosen significance level, power, and Minimum Detectable Effect ($\delta$), use the formula below:

$$N = \frac{(\sqrt{2\bar{p}\bar{q}} \cdot z_{1-\frac{\alpha}{2}} + \sqrt{p_{con}q_{con} + p_{exp}q_{exp}} \cdot z_{1-\beta})^2}{\delta^2}$$

*Note: Use A/A testing to estimate baseline values for $\bar{p}$ and $\bar{q}$.*

### Runtime & Duration Factors

**Duration = N / #visitors per day**  

-If this is 14 then run for 2 weeks

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

Analyzing A/B test results with Python includes:

1. Choosing an appropriate statistical test
2. Calculating the test statistics (T)
3. Calculating the p-value of the test statistics
4. Reject or fail to reject the statistical hypothesis (statistical significance)
5. Calculate the margin of error (external validity of the experiment)
6. Calculate confidence interval (external validity and practical significance of the experiment)

#### Choosing an appropriate statistical test
Appropriate statistical test that is usually categorized in **parametric and non-parametric** tests. 
The choice of the test depends on the following factors:format of the primary metric (underlying pdf),sample size (for CLT),nature of the statistical hypothesis (show that a relationship between two groups merely exists or identify the type of relationship between the groups)

The most popular **parametric** tests that are used in A/B testing are:
   -a. 2 Sample T-test (when N < 30, metric follows student-t distribution, and you want to identify whether there exist a   
    relationship and the type of relationship between control and experimental groups)
   -b. 2 Sample Z-test (when N > 30, metric follows asymptotic Normal distribution and you want to identify whether there exist a        relationship and the type of relationship between control and experimental groups)

The most popular non-parametric tests that are used in A/B testing are:
   -a. Fishers Exact test (small N, identify and you want to identify whether there exist a relationship between control and
     experimental groups)
   -b. Chi-Squared test (large N, identify and you want to identify whether there exist a relationship between control and
      experimental groups)
   -c. Wilcoxon Rank Sum/Mann Whitney test (small N or large N, skewed sampling distributions, testing the difference in medians
     between control and experimental groups)

### 2-Sample T-Test

If you want to test whether there is a statistically significant difference between the control and experimental groups’ metrics that are in the form of averages (e.g., average purchase amount), the metric follows a Student-t distribution. When the sample size is smaller than 30, you can use a 2-sample T-test to test the following hypothesis:

$$
\begin{cases}
H_0: \mu_{con} = \mu_{exp} \\
H_1: \mu_{con} \neq \mu_{exp}
\end{cases}
$$

Or alternatively written as:

$$
\begin{cases}
H_0: \mu_{con} - \mu_{exp} = 0 \\
H_1: \mu_{con} - \mu_{exp} \neq 0
\end{cases}
$$

* **Control Group:** The sampling distribution of means follows a Student-t distribution with degrees of freedom: $N_{con} - 1$.
  $$ \hat{\mu}_{con} \sim t(N_{con} - 1) $$
* **Experimental Group:** The sampling distribution of means follows a Student-t distribution with degrees of freedom: $N_{exp} - 1$.
  $$ \hat{\mu}_{exp} \sim t(N_{exp} - 1) $$

*Note: $N_{con}$ and $N_{exp}$ represent the total number of users in the Control and Experimental groups, respectively.*

#### Pooled Variance
An estimate for the **pooled variance** of the two samples can be calculated as follows:

$$ \hat{S}^2_{pooled} = \frac{(N_{con} - 1) * \sigma_{con}^2 + (N_{exp} - 1) * \sigma_{exp}^2}{N_{con} + N_{exp} - 2} * \left( \frac{1}{N_{con}} + \frac{1}{N_{exp}} \right) $$

Where $\sigma^2_{con}$ and $\sigma^2_{exp}$ are the sample variances of the Control and Experimental groups, respectively. 

#### Test Statistics
Consequently, the **test statistic** ($T$) of the 2-sample T-test with the hypothesis stated earlier can be calculated as follows:

$$ T = \frac{\hat{\mu}_{con} - \hat{\mu}_{exp}}{\sqrt{\hat{S}^2_{pooled}}} $$

#### Statistical Significance and P-Value
In order to test the **statistical significance** of the observed difference between sample means, we need to calculate the **p-value** of our test statistic. 

The p-value is the probability of observing values at least as extreme as the observed value when this is due to random chance. Stated differently, the p-value is the probability of obtaining an effect at least as extreme as the one in your sample data, assuming the null hypothesis ($H_0$) is true. 

Then the p-value of the test statistic can be calculated as follows:

$$
\begin{aligned}
p_{value} &= \Pr[t \leq -T \text{ or } t \geq T] \\
&= 2 * \Pr[t \geq T]
\end{aligned}
$$

The interpretation of a p-value is dependent on the chosen significance level, **alpha** ($\alpha$), which was chosen before running the test during the *power analysis*. If the calculated p-value appears to be smaller than or equal to alpha (e.g., $0.05$ for a $5\%$ significance level) we can reject the null hypothesis and state that there is a statistically significant difference between the primary metrics of the Control and Experimental groups.

#### Confidence Interval
Finally, to determine how accurate the obtained results are and also to comment about the practical significance of the obtained results, you can compute the **Confidence Interval** (CI) of your test by using the following formula:

$$ CI = \left[ (\hat{\mu}_{con} - \hat{\mu}_{exp}) - t_{1 - \frac{\alpha}{2}} * SE, \ (\hat{\mu}_{con} - \hat{\mu}_{exp}) + t_{1 - \frac{\alpha}{2}} * SE \right] $$

where the t_(1-alpha/2) is the critical value of the test corresponding to the two-sided t-test with alpha significance level and can be found using the t-table.

```python

import numpy as np
from scipy.stats import t

N_con = 20
df_con = N_con - 1 # degrees of freedom of Control 
N_exp = 20
df_exp = N_exp - 1 # degrees of freedom of Experimental 

# Significance level
alpha = 0.05

# data of control group with t-distribution
X_con = np.random.standard_t(df_con,N_con)
# data of experimental group with t-distribution
X_exp = np.random.standard_t(df_exp,N_exp)

# mean of control
mu_con = np.mean(X_con)
# mean of experimental
mu_exp = np.mean(X_exp)

# variance of control
sigma_sqr_con = np.var(X_con)
#variance of control
sigma_sqr_exp = np.var(X_exp)

# pooled variance
pooled_variance_t_test = ((N_con-1)*sigma_sqr_con + (N_exp -1) * sigma_sqr_exp)/(N_con + N_exp-2)*(1/N_con + 1/N_exp)

# Standard Error
SE = np.sqrt(pooled_variance_t_test)

# Test Statistics
T = (mu_con-mu_exp)/SE

# Critical value for two sided 2 sample t-test
t_crit = t.ppf(1-alpha/2, N_con + N_exp - 2)

# P-value of the two sided T-test using t-distribution and its symmetric property
p_value = t.sf(T, N_con + N_exp - 2)*2

# Margin of Error
margin_error = t_crit * SE
# Confidence Interval
CI = [(mu_con-mu_exp) - margin_error, (mu_con-mu_exp) + margin_error]

print("T-score: ", T)
print("T-critical: ", t_crit)
print("P_value: ", p_value)
print("Confidence Interval of 2 sample Z-test: ", np.round(CI,2))
view raw 2 Sample t-test for Means.py hosted with ❤ by GitHub
```

### 📝 2-Sample Z-Test Notes

#### 🔍 Overview
* Use to test statistically significant differences between **Control** and **Experimental** groups.
* Applies to metrics in the form of **averages** (e.g., average purchase amount) or **proportions** (e.g., CTR).
* Requires the metric to follow a **Normal distribution**, **OR**
* Requires a **sample size > 30** so the **Central Limit Theorem (CLT)** applies.
* CLT ensures sampling distributions are asymptotically Normal.

---

#### 📌 Case 1: Z-test for Comparing Proportions (2-Sided)
* Use when primary metrics are **proportions** (e.g., Click-Through Rate).
* Assumes that events (like clicks) occur **independently**.
* Tests for any significant difference between the two groups.

##### 📋 Hypotheses
$$
\begin{cases}
H_0: p_{con} = p_{exp} \\
H_1: p_{con} \neq p_{exp}
\end{cases}
$$

* Where each click event is a random variable taking two values: **1 (success)** or **0 (failure)**.
* Follows a **Bernoulli distribution** where $p_{con}$ and $p_{exp}$ are the click probabilities.
  
$$X_{con} \sim \text{Bern}(p_{con})$$
$$X_{exp} \sim \text{Bern}(p_{exp})$$

Hence after collecting interaction data, calculate the estimated probabilities for each group:

$$\hat{p}_{con} = \frac{X_{con}}{N_{con}} = \frac{\\#\text{clicks}_{con}}{\\#\text{impressions}_{con}}$$

$$\hat{p}_{exp} = \frac{X_{exp}}{N_{exp}} = \frac{\\#\text{clicks}_{exp}}{\\#\text{impressions}_{exp}}$$


##### 🤝 Pooled Probability and Variance
To test the difference between these probabilities, calculate the **pooled probability of success** and the **pooled variance**:

$$\hat{p}_{pooled} = \frac{X_{con} + X_{exp}}{N_{con} + N_{exp}} = \frac{\\#\text{clicks}_{con} + \\#\text{clicks}_{exp}}{\\#\text{impressions}_{con} + \\#\text{impressions}_{exp}}$$


$$\hat{S}^2_{pooled} = \hat{p}_{pooled}(1 - \hat{p}_{pooled}) \times \left(\frac{1}{N_{con}} + \frac{1}{N_{exp}}\right)$$
##### 📉 Standard Error and Test Statistic
The **Standard Error (SE)** is the square root of the estimated pooled variance:

$$SE = \sqrt{\hat{S}^2_{pooled}}$$

Using the standard error, calculate the final **test statistic ($T$)** for the 2-sample Z-test:

$$T = \frac{\hat{p}_{con} - \hat{p}_{exp}}{\sqrt{\hat{S}^2_{pooled}}}$$
##### 📊 P-Value and Confidence Interval
Calculate the **p-value** of the test statistic to determine statistical significance:

$$
\begin{aligned}
p_{value} &= \Pr[Z \leq -T \text{ or } Z \geq T] \\
&= 2 \times \Pr[Z \geq T]
\end{aligned}
$$

Compute the **Confidence Interval (CI)** for the difference between the two proportions:

$$CI = \left[ (\hat{p}_{con} - \hat{p}_{exp}) - z_{1-\frac{\alpha}{2}} \times SE, \,\, (\hat{p}_{con} - \hat{p}_{exp}) + z_{1-\frac{\alpha}{2}} \times SE \right]$$

* **$z_{1-\frac{\alpha}{2}}$**: The critical value from the Z-table for a two-sided test at the $\alpha$ significance level.


#### 📌 Case 2: Z-test for Comparing Means (2-Sided)
* Use to test whether there is a statistically significant difference between the Control and Experimental groups’ metrics that are in the form of **averages** (e.g., average purchase amount).

##### 📋 Hypotheses
$$
\begin{cases}
H_0: \mu_{con} = \mu_{exp} \\
H_1: \mu_{con} \neq \mu_{exp}
\end{cases}
$$

**Alternatively stated as:**
$$
\begin{cases}
H_0: \mu_{con} - \mu_{exp} = 0 \\
H_1: \mu_{con} - \mu_{exp} \neq 0
\end{cases}
$$

##### 📊 Sampling Distributions
The sampling distribution of the means for both groups follows a **Normal distribution**:

$$\hat{\mu}_{con} \sim N\left(\mu_{con}, \frac{\sigma_{con}^2}{N_{con}}\right)$$

$$\hat{\mu}_{exp} \sim N\left(\mu_{exp}, \frac{\sigma_{exp}^2}{N_{exp}}\right)$$

The difference between the two sample means also follows a Normal distribution:

$$\hat{\mu}_{con} - \hat{\mu}_{exp} \sim N\left(\mu_{con} - \mu_{exp}, \,\, \frac{\sigma_{con}^2}{N_{con}} + \frac{\sigma_{exp}^2}{N_{exp}}\right)$$

---

##### 🧮 Test Statistic ($T$)
The test statistic of the 2-sample Z-test for the difference in means is calculated as follows:

$$T = \frac{\hat{\mu}_{con} - \hat{\mu}_{exp}}{\sqrt{\frac{\sigma_{con}^2}{N_{con}} + \frac{\sigma_{exp}^2}{N_{exp}}}} \sim N(0, 1)$$

##### 📉 Standard Error ($SE$)
The standard error is equal to the square root of the estimate of the pooled variance:

$$SE = \sqrt{\frac{\sigma_{con}^2}{N_{con}} + \frac{\sigma_{exp}^2}{N_{exp}}}$$

##### 📈 P-Value
The p-value of this test statistic is calculated as follows:

$$
\begin{aligned}
p_{value} &= \Pr[Z \leq -T \text{ or } Z \geq T] \\
&= 2 \times \Pr[Z \geq T]
\end{aligned}
$$

##### 🔒 Confidence Interval ($CI$)
Finally, compute the confidence interval of the test as follows:

$$CI = \left[ (\hat{\mu}_{con} - \hat{\mu}_{exp}) - z_{1-\frac{\alpha}{2}} \times SE, \,\, (\hat{\mu}_{con} - \hat{\mu}_{exp}) + z_{1-\frac{\alpha}{2}} \times SE \right]$$

```python

import numpy as np
from scipy.stats import norm

N_con = 60
N_exp = 60

# Significance Level
alpha = 0.05

X_A = np.random.randint(100, size = N_con)
X_B = np.random.randint(100, size = N_exp)

# Calculating means of control and experimental groups
mu_con = np.mean(X_A)
mu_exp = np.mean(X_B)

variance_con = np.var(X_A)
variance_exp = np.var(X_B)

# Pooled Variance
pooled_variance = np.sqrt(variance_con/N_con + variance_exp/N_exp)

# Test statistics
T = (mu_con-mu_exp)/np.sqrt(variance_con/N_con + variance_exp/N_exp)

# two sided test and using symmetry property of Normal distibution so we multiple with 2
p_value = norm.sf(T)*2

# Z-critical value
Z_crit  = norm.ppf(1-alpha/2)

# Margin of error
m = Z_crit*pooled_variance

# Confidence Interval
CI = [(mu_con - mu_exp) - m, (mu_con - mu_exp) + m]


print("Test Statistics stat: ", T)
print("Z-critical: ", Z_crit)
print("P_value: ", p_value)
print("Confidence Interval of 2 sample Z-test for proportions: ", np.round(CI,2))

import matplotlib.pyplot as plt
z = np.arange(-3,3,  0.1)
plt.plot(z, norm.pdf(z), label = 'Standard Normal Distribution',color = 'purple',linewidth = 2.5)
plt.fill_between(z[z>Z_crit], norm.pdf(z[z>Z_crit]), label = 'Right Rejection Region',color ='y' )
plt.fill_between(z[z<(-1)*Z_crit], norm.pdf(z[z<(-1)*Z_crit]), label = 'Left Rejection Region',color ='y' )
plt.title("Two Sample Z-test rejection region")
plt.legend()
plt.show()
```
### 📊 Chi-Squared Test

#### 🔍 Overview
* Use to test whether there is a statistically significant difference between the **Control** and **Experimental** groups' performance metrics (e.g., conversions).
* Use when you do not need to know the specific direction or nature of the relationship (i.e., which one is strictly better).
* The metric must be a **binary variable** (e.g., conversion vs. no conversion, click vs. no click).

##### 📋 Hypotheses
$$
\begin{cases}
H_0: CR_{con} = CR_{exp} \\
H_1: CR_{con} \neq CR_{exp}
\end{cases}
$$

**Alternatively stated as:**
$$
\begin{cases}
H_0: CR_{con} - CR_{exp} = 0 \\
H_1: CR_{con} - CR_{exp} \neq 0
\end{cases}
$$

---

#### 📅 Contingency Tables
The collected data is represented in a table where **O** corresponds to **Observed** values and **T** corresponds to **Theoretical (Expected)** values.

**Theoretical / Expected Values ($T$):**

| Group | # Sessions | # Converted | # Not Converted |
| :--- | :--- | :--- | :--- |
| **Control** | $N_{con}$ | $T_{con,1}$ | $T_{con,0}$ |
| **Experimental** | $N_{exp}$ | $T_{exp,1}$ | $T_{exp,0}$ |

**Observed Values ($O$):**

| Group | # Sessions | # Converted | # Not Converted |
| :--- | :--- | :--- | :--- |
| **Control** | $N_{con}$ | $O_{con,1}$ | $O_{con,0}$ |
| **Experimental** | $N_{exp}$ | $O_{exp,1}$ | $O_{exp,0}$ |

---

#### 🧮 Test Statistic ($T$)
The general test statistic of the Chi-2 ($\chi^2$) test is expressed as:

$$T = \sum_{i} \frac{(Observed_i - Expected_i)^2}{Expected_i}$$

* **$Observed$** corresponds to your collected data.
* **$Expected$** corresponds to the theoretical values.
* **$i$** can take values of $0$ (no conversion) and $1$ (conversion). 

Each of these factors has a separate denominator. Expanded for exactly two groups, the formula is:

$$T = \frac{(O_{con,1} - T_{con,1})^2}{T_{con,1}} + \frac{(O_{con,0} - T_{con,0})^2}{T_{con,0}} + \frac{(O_{exp,1} - T_{exp,1})^2}{T_{exp,1}} + \frac{(O_{exp,0} - T_{exp,0})^2}{T_{exp,0}}$$

##### 💡 Calculating Expected Values
The expected value is equal to the number of times each version of the product is viewed, multiplied by the pooled probability of it leading to a conversion (or click in the case of CTR).

⚠️ **Important Note:** Since the Chi-2 test is a non-parametric test, its Standard Error ($SE$) and Confidence Interval ($CI$) cannot be calculated in the standard way as done in the parametric Z-test or T-test.

```python

import numpy as np
from scipy.stats import chi2

O = np.array([86, 83, 5810,3920])
T = np.array([105,65,5781, 3841])

# Squared_relative_distance

def calculate_D(O,T):
    D_sum = 0
    for i in range(len(O)):
        D_sum += (O[i] - T[i])**2/T[i]
    return(D_sum)

D = calculate_D(O,T)
p_value = chi2.sf(D, df = 1)


import matplotlib.pyplot as plt
# Step 1: pick a x-axis range like in case of z-test (-3,3,0.1)
d = np.arange(0,5,0.1)
# Step 2: drawing the initial pdf of chi-2 with df = 1 and x-axis d range we just created
plt.plot(d, chi2.pdf(d, df = 1), color = "purple")
# Step 3: filling in the rejection region
plt.fill_between(d[d>D], chi2.pdf(d[d>D], df = 1), color = "y")
# Step 4: adding title
plt.title("Two Sample Chi-2 Test rejection region")
# Step 5: showing the plt graph
plt.show()
```

Standard Error and Confidence Interval for Non-parametric Tests:

In the case of parametric tests, the calculation of Standard Error and Confidence Interval is straightforward. However, in the case of Non-parametric tests, the calculation is no longer straightforward. To calculate the Standard Error and the Confidence Interval of a non-parametric statistical test that aims to compare the sample means or sample medians of control and experimental groups, one needs to use resampling techniques such as Bootstrapping and Boostrap Quantile method, respectively.


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
