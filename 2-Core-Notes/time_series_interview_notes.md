# Time Series — Data Science Interview Notes

> Quick-reference notes for data science interviews. Part 1 covers the fundamentals (components, visualization, feature engineering, stationarity) based on my class notes. Part 2 fills in what interviewers usually ask next: forecasting models, validation, metrics, and common Q&A.

---

## Table of Contents

**Part 1 — Fundamentals**
1. [What is Time Series Data?](#1-what-is-time-series-data)
2. [Components of a Time Series](#2-components-of-a-time-series)
3. [Decomposition (Additive vs Multiplicative)](#3-decomposition-additive-vs-multiplicative)
4. [Visualization & Exploration](#4-visualization--exploration)
5. [Autocorrelation: ACF & PACF](#5-autocorrelation-acf--pacf)
6. [Feature Engineering](#6-feature-engineering)
7. [Stationarity](#7-stationarity)
8. [Testing for Stationarity (ADF & KPSS)](#8-testing-for-stationarity-adf--kpss)
9. [Making a Series Stationary](#9-making-a-series-stationary)

**Part 2 — Beyond the Basics**

10. [Forecasting Models Overview](#10-forecasting-models-overview)
11. [ARIMA Family](#11-arima-family)
12. [Validation: Train/Test Splits Done Right](#12-validation-traintest-splits-done-right)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Residual Diagnostics](#14-residual-diagnostics)
15. [Common Pitfalls](#15-common-pitfalls)
16. [Interview Q&A](#16-interview-qa)
17. [Cheat Sheet](#17-cheat-sheet)

---

# Part 1 — Fundamentals

## 1. What is Time Series Data?

A **time series** is a sequence of observations recorded in chronological order, usually at equally spaced intervals (hourly, daily, monthly…).

**Key characteristics**
| Characteristic | Meaning |
|---|---|
| Sequential order | Order matters — you can't shuffle rows like in normal tabular ML |
| Temporal dependence | Current value often depends on past values (autocorrelation) |
| Equally spaced intervals | Ideal scenario; real data often has gaps that need handling |

**Analysis vs Forecasting**
- **Time series analysis** — describing and understanding patterns (trend, seasonality, anomalies) using descriptive/inferential statistics.
- **Time series forecasting** — using historical values to predict future values.

**Real-world examples**
- Finance: stock prices, S&P 500, currency rates
- Retail: sales/demand forecasting, inventory planning
- Weather prediction, electricity demand
- Software engineering: response times, error rates, CPU/memory utilization, network traffic, website visits, feature usage

> **Example scenario (good to talk through in interviews):** An online multiplayer game logs players per hour. You want to plan server capacity, sell ad slots and schedule maintenance.
> - **Seasonality** → peak hours (e.g., 7 PM–2 AM, weekends) → scale up servers; schedule maintenance in troughs
> - **Cycles** → periods where advertisers pull back (e.g., economic downturn)
> - **Trend** → is the user base growing overall? Plan long-term capacity
> - **Residuals** → unexplained spikes/drops (e.g., server outage) → anomaly detection

---

## 2. Components of a Time Series

| Component | Type | Description | Example |
|---|---|---|---|
| **Level** | Systematic | Baseline average value of the series | Average daily users ≈ 10k |
| **Trend** | Systematic | Long-term increase/decrease; need not be linear | Growing user base |
| **Seasonality** | Systematic | Repeating pattern with a **fixed, known period** | Higher sales every December |
| **Cyclicity** | Systematic | Rises/falls with **no fixed period**, usually longer (multi-year) | Business/economic cycles, housing market (6–10 yrs) |
| **Noise / Residual** | Non-systematic | Random, unpredictable variation left over | Natural disasters, one-off outages |

**Seasonality vs Cyclicity — a classic interview question**
- Seasonality: fixed & known frequency (daily, weekly, yearly), tied to the calendar.
- Cyclicity: variable length, not tied to the calendar, typically longer and harder to predict.

**Why each matters**
- Trend → reveals the underlying trajectory
- Seasonality → accounting for it lets you see other patterns it might hide
- Residuals → surface anomalies, unexpected events, or data-collection errors

---

## 3. Decomposition (Additive vs Multiplicative)

Decomposition splits a series into **trend-cycle + seasonal + residual**.

| Model | Formula | Use when |
|---|---|---|
| **Additive** | $y_t = T_t + S_t + R_t$ | Seasonal swings are roughly **constant** in size |
| **Multiplicative** | $y_t = T_t \times S_t \times R_t$ | Seasonal swings **grow with the level** of the series |

> Tip: A log transform turns multiplicative into additive: $\log y_t = \log T_t + \log S_t + \log R_t$

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Classical decomposition
result = seasonal_decompose(df["sales"], model="multiplicative", period=12)
result.plot()

# STL — more robust (handles outliers, seasonality can change over time)
stl = STL(df["sales"], period=12, robust=True).fit()
stl.plot()
```

---

## 4. Visualization & Exploration

**Why visualize?** Get a first feel for the data, reveal patterns and anomalies, build intuition for model choice, and communicate findings.

| Plot | What it shows | What to look for |
|---|---|---|
| **Time plot** | Values vs time, joined by lines | Trend, seasonality, outliers, structural breaks, changing variance |
| **Seasonal plot** | Each season (e.g., year) overlaid on the same axis (e.g., months) | Whether seasonal pattern is consistent year to year; unusual seasons |
| **Seasonal subseries plot** | Mini time plot per season (all Januaries together, etc.) | Changes *within* a particular season over time |
| **Multiple seasonal periods** | e.g., weekly pattern inside hourly data | Daily + weekly seasonality at once (common in electricity/web traffic) |
| **Scatterplot** | One series vs another (e.g., demand vs temperature) | Relationships, non-linearity |
| **Scatterplot matrix** | All pairs of variables | Quick overview of pairwise relationships |
| **Lag plot** | $y_t$ vs $y_{t-k}$ for several k | Autocorrelation at each lag |
| **ACF plot (correlogram)** | Autocorrelation vs lag | Trend, seasonality, stationarity, model order |

**Reading a time plot — example observations**
- Increasing trend
- Strong seasonality that **grows with the level** → multiplicative
- Sudden drop at the start of each year → calendar effect worth encoding as a feature

**Correlation caution**
- Correlation only measures **linear** relationships.
- Very different relationships can share the same coefficient (Anscombe's quartet: all ≈ 0.82).
- A low correlation does **not** mean no relationship — e.g., electricity demand vs temperature is U-shaped (heating when cold, AC when hot). Always plot the data.

---

## 5. Autocorrelation: ACF & PACF

- **Correlation** — linear relationship between two variables.
- **Autocorrelation** — linear relationship between a series and its own **lagged** values.
- $r_k$ = correlation between $y_t$ and $y_{t-k}$. The set of all $r_k$ is the **Autocorrelation Function (ACF)**.
- **PACF (Partial ACF)** — correlation between $y_t$ and $y_{t-k}$ **after removing the effect of the intermediate lags** $1 … k-1$.

**Reading ACF patterns**
| Pattern in ACF | Meaning |
|---|---|
| Large positive values that decay **slowly** | Trend (non-stationary) |
| Spikes at regular intervals (lag 12, 24… for monthly) | Seasonality |
| Slow decay + "scalloped" shape | Trend **and** seasonality |
| All bars inside confidence band | White noise — nothing left to model |

**Stationary vs non-stationary**
| | Stationary | Non-stationary |
|---|---|---|
| ACF | Drops quickly to ~0 | Decays slowly |
| PACF | Cuts off sharply after a few lags | Decays slowly |

```python
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_vals = acf(df["y"], nlags=24)
plot_acf(df["y"], lags=40)
plot_pacf(df["y"], lags=40, method="ywm")
```

---

## 6. Feature Engineering

Goal: extract useful signal so that general ML models (linear regression, XGBoost, LightGBM, etc.) can learn temporal patterns.

### 6.1 Descriptive statistics
Mean, median, mode, std, quantiles, min/max — used for outlier detection, scaling, understanding the distribution.

### 6.2 Lag features
Past values used as inputs: $y_{t-1}, y_{t-2}, …, y_{t-k}$.
- Example: to predict today's temperature, use temperatures from 1, 2, 3… days ago.
- Pick lags using ACF/PACF and domain knowledge (e.g., lag 7 for weekly, lag 365 for yearly patterns in daily data).

### 6.3 Window statistics
| Type | How it works | Use when |
|---|---|---|
| **Rolling (sliding) window** | Stats over the last *n* observations (e.g., 7-day moving average); window slides forward | Recent data matters more |
| **Expanding window** | Stats over **all** observations up to time *t*; window grows | All history is equally relevant |
| **EWM (exponentially weighted)** | Weighted average, recent points weigh more | Want smoothness + recency |

### 6.4 Date-time features
Hour, day of week, day of month, month, quarter, year, week of year, is_weekend, is_month_end.

### 6.5 Timestamp decomposition / special events
Flag special dates: holidays (Diwali, Christmas), promotions, Black Friday, days-before-holiday. Sales of seasonal items spike around these.

### 6.6 Cyclical encoding
Hour 23 and hour 0 are neighbors, but as integers they look far apart. Encode with sine/cosine:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

```python
df = df.sort_values("date").set_index("date")

# Lags
for k in [1, 7, 14]:
    df[f"lag_{k}"] = df["y"].shift(k)

# Rolling & expanding — ALWAYS shift(1) first to avoid leakage
df["roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
df["roll_std_7"]  = df["y"].shift(1).rolling(7).std()
df["expanding_mean"] = df["y"].shift(1).expanding().mean()
df["ewm_7"] = df["y"].shift(1).ewm(span=7).mean()

# Date-time features
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# Cyclical encoding
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

> ⚠️ **Data leakage:** a rolling mean that includes the current value `y_t` leaks the target. Use `.shift(1)` before `.rolling()`.

---

## 7. Stationarity

A series is **stationary** when its statistical properties — **mean, variance and autocovariance** — do not change over time.

In practice, a stationary series has:
- No trend (constant mean)
- Constant variance
- No seasonality
- Autocorrelation that depends only on the lag, not on *when* you look

> ⚠️ Note: "no seasonality" alone is **not** enough — a series with a trend or growing variance is still non-stationary.

**Types**
| Type | Definition |
|---|---|
| **Strict (strong) stationarity** | The entire joint distribution is unchanged by shifts in time (rarely verifiable in practice) |
| **Weak (covariance) stationarity** | Constant mean, constant variance, autocovariance depends only on lag — what we usually mean |
| **Trend stationary** | Stationary after removing a deterministic trend |
| **Difference stationary** | Stationary after differencing (e.g., random walk) |

**Why it matters**
- Classical models (AR, MA, ARMA/ARIMA) assume stationarity (ARIMA handles it via the "I" = differencing step).
- Patterns learned from the past are only reliable for the future if the statistics stay stable.
- Regressing one non-stationary series on another can produce **spurious correlations**.

**How to check**
1. **Visual** — time plot (trend? seasonality? changing spread?), rolling mean/std plot
2. **ACF/PACF** — slow decay → non-stationary
3. **Statistical tests** — ADF and KPSS (below)

---

## 8. Testing for Stationarity (ADF & KPSS)

A **unit root** is a property of processes like a random walk ($y_t = y_{t-1} + \epsilon_t$) that makes them non-stationary — shocks never die out.

### ADF — Augmented Dickey-Fuller
| | |
|---|---|
| **H₀** | Series has a unit root → **non-stationary** |
| **H₁** | No unit root → stationary |
| **Reject H₀ when** | p-value < 0.05 (equivalently test statistic < critical value; more negative = stronger evidence) |
| **Conclusion if rejected** | Stationary ✅ |

### KPSS — Kwiatkowski-Phillips-Schmidt-Shin
| | |
|---|---|
| **H₀** | Series is **stationary** (around a level or deterministic trend) |
| **H₁** | Series has a unit root → non-stationary |
| **Reject H₀ when** | p-value < 0.05 (test statistic > critical value) |
| **Conclusion if NOT rejected** | Stationary ✅ |

> 🔑 **Memory trick:** The hypotheses are **reversed**. ADF's null = "non-stationary"; KPSS's null = "stationary". For ADF you want a **small** p-value; for KPSS you want a **large** p-value.

### Using both tests together
| ADF says | KPSS says | Conclusion | Action |
|---|---|---|---|
| Stationary | Stationary | **Stationary** | None |
| Non-stationary | Non-stationary | **Non-stationary** | Difference / transform |
| Non-stationary | Stationary | **Trend stationary** | Remove the trend (detrend), retest |
| Stationary | Non-stationary | **Difference stationary** | Difference, retest |

```python
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series):
    stat, pvalue, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF stat={stat:.3f}, p={pvalue:.4f}, crit={crit}")
    print("Stationary" if pvalue < 0.05 else "Non-stationary")

def kpss_test(series, regression="c"):   # "c" = level, "ct" = trend
    stat, pvalue, lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    print(f"KPSS stat={stat:.3f}, p={pvalue:.4f}, crit={crit}")
    print("Stationary" if pvalue >= 0.05 else "Non-stationary")
```

> Note: statsmodels' KPSS p-value is interpolated from a table and capped at 0.01–0.10, so you'll often see warnings at the boundaries.

---

## 9. Making a Series Stationary

| Method | Removes | How | Pros | Cons |
|---|---|---|---|---|
| **Differencing** | Trend (& seasonality with seasonal differencing) | $y'_t = y_t - y_{t-1}$; seasonal: $y_t - y_{t-m}$ | Simple, very effective, used by ARIMA | Lose interpretability; over-differencing adds artificial negative autocorrelation |
| **Subtract rolling mean** | Time-varying mean | $y_t - \text{rolling\_mean}_t$ | Easy with pandas `.rolling()` | Variance may remain; loses first *n* points |
| **Detrending** | Deterministic trend | Fit linear/polynomial trend, subtract it | Keeps original scale & interpretability | Must pick the right trend form |
| **Log transform** | Growing variance, exponential growth | $\log(y_t)$ | Stabilizes variance; multiplicative → additive | Needs positive values |
| **Box-Cox** | Non-constant variance, skew | Power transform with parameter λ | Flexible; λ chosen automatically | Needs positive values; harder to interpret |

**Box-Cox λ values**
| λ | Effect |
|---|---|
| 0 | Natural log |
| 0.5 | Square-root-like |
| 1 | No real change (just shifted) |
| < 1 | Reduces right (positive) skew |
| > 1 | Addresses left (negative) skew |

**Typical workflow:** log/Box-Cox to fix variance → difference to fix trend → seasonal difference if needed → re-run ADF/KPSS.

```python
import numpy as np
from scipy.stats import boxcox

df["log_y"] = np.log(df["y"])
df["diff_1"] = df["y"].diff()               # first difference
df["seasonal_diff"] = df["y"].diff(12)      # seasonal difference (monthly data)
df["y_minus_roll"] = df["y"] - df["y"].rolling(12).mean()
df["boxcox_y"], lam = boxcox(df["y"])       # y must be > 0
```

> Remember to **invert** the transformations on your forecasts (cumulative sum for differencing, `exp` for log, inverse Box-Cox) before reporting results.

---

# Part 2 — Beyond the Basics

## 10. Forecasting Models Overview

| Category | Models | Notes |
|---|---|---|
| **Baselines** | Naive ($\hat y_{t+1} = y_t$), Seasonal naive ($\hat y_{t+1} = y_{t+1-m}$), Mean, Drift | **Always** compare against these first |
| **Smoothing** | Moving average, Simple Exponential Smoothing (level), Holt (level + trend), Holt-Winters (level + trend + seasonality) | Fast, interpretable, strong for short horizons |
| **Statistical** | AR, MA, ARMA, ARIMA, SARIMA, SARIMAX (with exogenous vars), VAR (multivariate) | Need stationarity (or differencing) |
| **Decomposition-based** | Prophet, STL + model | Prophet handles holidays, multiple seasonalities, missing data |
| **ML on features** | Linear regression, Random Forest, XGBoost, LightGBM | Uses lag/rolling/date features from Section 6; can't extrapolate trends well (trees) |
| **Deep learning** | RNN, LSTM, GRU, TCN, Transformers (TFT, N-BEATS, PatchTST) | Useful with lots of data or many related series |

**Exponential smoothing in one line:** $\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t$, with $0 < \alpha < 1$. Higher α → reacts faster to recent changes.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
forecast = model.forecast(12)
```

---

## 11. ARIMA Family

**ARIMA(p, d, q)**
- **AR(p)** — AutoRegressive: regression on the last *p* values
  $y_t = c + \phi_1 y_{t-1} + … + \phi_p y_{t-p} + \epsilon_t$
- **I(d)** — Integrated: number of times the series is differenced to become stationary
- **MA(q)** — Moving Average: regression on the last *q* forecast **errors**
  $y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + … + \theta_q \epsilon_{t-q}$

**SARIMA(p,d,q)(P,D,Q)ₘ** — adds seasonal AR/differencing/MA terms with season length *m*.
**SARIMAX** — SARIMA + external regressors (price, promotions, temperature…).

### Choosing p and q from ACF/PACF (after making series stationary)
| Model | ACF | PACF |
|---|---|---|
| **AR(p)** | Tails off gradually | **Cuts off after lag p** |
| **MA(q)** | **Cuts off after lag q** | Tails off gradually |
| **ARMA(p,q)** | Tails off | Tails off |

**Choosing d:** number of differences needed to pass ADF/KPSS (usually 0, 1 or 2).
**In practice:** grid search / `auto_arima` (pmdarima) and pick the lowest **AIC/BIC**.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(model.summary())
pred = model.get_forecast(steps=12)
mean, conf_int = pred.predicted_mean, pred.conf_int()
```

---

## 12. Validation: Train/Test Splits Done Right

❌ **Never** use random K-fold / shuffled splits — that trains on the future to predict the past (leakage).

✅ **Split by time:** train on earlier data, test on later data.

**Walk-forward / rolling-origin validation**
```
Fold 1: [train-----][test]
Fold 2: [train---------][test]
Fold 3: [train-------------][test]
```
- **Expanding window** — training set keeps growing
- **Sliding window** — fixed-size training set that moves forward (good if old data becomes irrelevant)
- Add a **gap** between train and test if features use future-dependent info or there's reporting lag

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
```

---

## 13. Evaluation Metrics

| Metric | Formula (idea) | Pros | Cons |
|---|---|---|---|
| **MAE** | mean(\|y − ŷ\|) | Interpretable, robust to outliers | Scale-dependent |
| **RMSE** | √mean((y − ŷ)²) | Penalizes large errors | Sensitive to outliers; scale-dependent |
| **MAPE** | mean(\|y − ŷ\| / \|y\|) × 100 | Scale-free, business-friendly (%) | Undefined/explodes near 0; penalizes over-forecasts more |
| **sMAPE** | uses (\|y\| + \|ŷ\|)/2 in denominator | Bounded, handles near-0 better | Still asymmetric, less intuitive |
| **MASE** | MAE / MAE of naive forecast | Scale-free; < 1 means you beat the naive baseline | Less known by business audiences |
| **WAPE** | Σ\|y − ŷ\| / Σ\|y\| | Good for retail/demand with many low-volume items | Aggregated view |

> Interview tip: mention that you'd pick the metric based on business cost (e.g., under-forecasting inventory vs over-forecasting), and always report performance vs a **naive baseline**.

---

## 14. Residual Diagnostics

A good model's residuals should look like **white noise**:
- Mean ≈ 0 (otherwise the forecasts are biased)
- No autocorrelation (otherwise there's signal left to model)
- Constant variance
- (Nice to have) Normally distributed → valid prediction intervals

**Checks**
- Plot residuals over time
- ACF of residuals — all bars within the confidence band
- **Ljung-Box test** — H₀: no autocorrelation up to lag k. You want p-value **> 0.05**.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(model.resid, lags=[10, 20], return_df=True))
```

---

## 15. Common Pitfalls

1. **Shuffling data** or random train/test splits → leakage.
2. **Rolling/lag features computed without `shift`** → target leakage.
3. **Scaling/normalizing on the full dataset** before splitting → fit scalers on train only.
4. **Not comparing to a naive baseline** → a "good" model may be worse than "tomorrow = today".
5. **Forgetting to invert transforms** (differencing, log, Box-Cox) before evaluating.
6. **Over-differencing** → ACF at lag 1 becomes strongly negative; variance increases.
7. **Ignoring missing timestamps** — reindex to a full date range and then impute (forward fill, interpolation) deliberately.
8. **Using correlation alone** to judge relationships — plot the data.
9. **Tree models can't extrapolate** trends beyond training range → detrend first or use linear/statistical models.
10. **Ignoring structural breaks** (COVID, pricing change) → consider dropping, flagging, or modeling them.

```python
# Fill missing timestamps
df = df.asfreq("D")                       # reindex to daily frequency
df["y"] = df["y"].interpolate("time")     # or .ffill()
```

---

## 16. Interview Q&A

**Q1. What makes time series different from regular regression data?**
Observations aren't independent — they're ordered and autocorrelated. This affects how you split data (by time), build features (lags, windows), and evaluate (walk-forward).

**Q2. Difference between seasonality and cyclicity?**
Seasonality has a fixed, known period tied to the calendar (weekly, yearly). Cycles have variable, usually longer periods (economic cycles).

**Q3. What is stationarity and why do we need it?**
Constant mean, variance and autocovariance over time. Classical models assume it; stable statistics make past patterns useful for predicting the future and avoid spurious regressions.

**Q4. How do you check for stationarity?**
Visual inspection (time plot, rolling mean/std), ACF (slow decay = non-stationary), and ADF + KPSS tests.

**Q5. ADF vs KPSS?**
ADF null = unit root (non-stationary); KPSS null = stationary. Use both; the combination tells you whether the series is stationary, trend-stationary, or difference-stationary.

**Q6. How do you make a series stationary?**
Log/Box-Cox for variance, differencing (regular and seasonal) for trend/seasonality, detrending for deterministic trends, then retest.

**Q7. What's the difference between ACF and PACF?**
ACF = total correlation with lag k (includes indirect effects through intermediate lags). PACF = direct correlation with lag k after removing intermediate lags. PACF cut-off → AR order; ACF cut-off → MA order.

**Q8. Explain ARIMA parameters.**
p = number of autoregressive lags, d = number of differences, q = number of lagged error terms. SARIMA adds seasonal (P, D, Q, m).

**Q9. How would you validate a forecasting model?**
Time-based split, walk-forward (expanding or sliding window) cross-validation, compare against naive/seasonal-naive baseline, check residuals are white noise.

**Q10. Which metric would you use?**
Depends on business context. MAE for interpretability, RMSE if large errors are costly, MAPE for % communication (not when values near 0), MASE for comparing across series.

**Q11. Can you use XGBoost for time series?**
Yes — reframe as supervised learning with lag, rolling and date features. Watch for leakage, and remember trees can't extrapolate beyond the target range seen in training.

**Q12. When would you choose ARIMA vs ML vs deep learning?**
ARIMA/ETS: single series, limited data, need interpretability. ML (LightGBM): many features/exogenous variables, many series. Deep learning: large datasets, many related series, complex patterns.

**Q13. How do you handle multiple seasonalities (e.g., daily + weekly in hourly data)?**
Fourier terms, Prophet, TBATS, MSTL decomposition, or date features (hour, day-of-week) in ML models.

**Q14. How do you forecast multiple steps ahead?**
- **Recursive:** predict t+1, feed it back to predict t+2 (errors accumulate)
- **Direct:** separate model per horizon
- **Multi-output:** one model outputs the whole horizon

**Q15. What is a random walk and why does it matter?**
$y_t = y_{t-1} + \epsilon_t$. Non-stationary (unit root); the best forecast is the last value (naive). Stock prices are often close to a random walk — so a naive baseline is hard to beat.

**Q16. Walk through the multiplayer-game scenario.**
Decompose hourly players → seasonality (daily/weekly peaks for server scaling and maintenance windows), trend (capacity planning), cycles (ad demand), residuals (outage/anomaly detection). Then forecast with a seasonal model or LightGBM with hour/day features, validated walk-forward.

---

## 17. Cheat Sheet

```
EDA:         time plot → seasonal plot → decomposition → ACF/PACF
Stationary?  ADF p < 0.05 ✅   |   KPSS p > 0.05 ✅
Fix:         variance → log / Box-Cox ; trend → diff(1) ; seasonality → diff(m)
ACF/PACF:    AR(p) → PACF cuts at p ; MA(q) → ACF cuts at q
Features:    lags, rolling (shift first!), expanding, date parts, holidays, sin/cos
Split:       by time, walk-forward CV, never shuffle
Baseline:    naive / seasonal naive → beat it or it's not worth it
Metrics:     MAE, RMSE, MAPE (not near 0), MASE (<1 beats naive)
Residuals:   white noise → Ljung-Box p > 0.05
```

**Key Python libraries:** `pandas`, `statsmodels`, `pmdarima`, `prophet`, `sktime`, `darts`, `scikit-learn`, `lightgbm`

**Good references**
- *Forecasting: Principles and Practice* — Rob Hyndman & George Athanasopoulos (free online: otexts.com/fpp3)
- statsmodels time series docs
