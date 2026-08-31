# Pandas EDA, Preprocessing, & Modeling Cheat Sheet

## 1. Initial Data Inspection
```python
df.info()                    # Memory usage, data types, and non-null counts
df.describe(include='all')   # Stats for both numerical and categorical columns
df.shape                     # (rows, columns) tuple
df.head(5) / df.tail(5)      # View first or last 5 rows
df['col'].value_counts()     # Count unique values (useful for target classes)
df.nunique()                 # Count of unique values per column
df.duplicated().sum()        # check for duplicates
df[df.duplicated()]          #Shows all duplicate rows (keeps the first occurrence hidden)
tesla_df['Date'] = pd.to_datetime(tesla_df['Date']) # convert to datetime field
tesla_df.set_index('Date', inplace=True) # set Date column as index of the dataframe
tesla_df.sort_index(ascending=True, inplace=True) #sort the df based on this Date index

```

## 2. Handling Missing Data (NaNs)
```python
df.isna().sum()              # Count missing values per column
df=df.dropna(axis=0) # Any col has missing values ,drop all those rows 
df=df.dropna(subset=['col']) # Drop rows if missing in specific column
df['col'] = df['col'].fillna(df['col'].median()) # Impute using median
df = df.ffill() #fill nan with prev known value until new value comes.Forward fill/ time-series
```

## 3. Data Transformation & Cleaning
```python
df.drop_duplicates(inplace=True)               # Remove duplicate rows
df['col'] = df['col'].astype('int64')          # Convert data types
df.rename(columns={'old':'new'}, inplace=True) # Rename columns
df.drop(columns=['col1', 'col2'], inplace=True)# Drop columns
```
## EDA

#Basic EDA commands 

# Seaborn Plotting Cheat Sheet

Seaborn groups plots by **statistical purpose**  into 3 Master Functions (`relplot`, `displot`, `catplot`).
Using these figure-level functions is recommended because changing the plot type is as simple as updating the `kind` parameter.

---

## 🚀 The Big Three (Master Functions)

### 1. Relational Plots (`sns.relplot`)
*Use case: Analyzing relationships between continuous variables.*
*   `kind="scatter"` (Default): Spots trends and clusters.
*   `kind="line"`: Tracking changes over time or continuous intervals.

```python
import seaborn as sns

sns.relplot(data=df, x="col_x", y="col_y", kind="scatter", hue="category")
```

### 2. Distribution Plots (`sns.displot`)
*Use case: Visualizing data spread, shape, and frequency.*
*   `kind="hist"` (Default): Classic binned frequency counts.
*   `kind="kde"`: Smooth probability density curve.
*   `kind="ecdf"`: Cumulative proportions.

```python
sns.displot(data=df, x="col_x", kind="hist", kde=True) # kde=True adds the line overlay
```

### 3. Categorical Plots (`sns.catplot`)
*Use case: Comparing numeric metrics across different categorical groups.*
*   **Scatter styles:** `kind="strip"` (default), `kind="swarm"` (no overlap)
*   **Distribution styles:** `kind="box"`, `kind="violin"`, `kind="boxen"`
*   **Estimation styles:** `kind="bar"` (means), `kind="count"` (frequencies), `kind="point"`

```python
sns.catplot(data=df, x="category_col", y="numeric_col", kind="box")
```

---

## 🎛️ Special Purpose Plots (Standalone Functions)

These plots do not route through the master functions and must be called directly.

### Regression Plots (Trend Lines)
Fits a linear regression model automatically over a scatter plot.
```python
sns.lmplot(data=df, x="col_x", y="col_y", hue="category")
```

### Matrix Plots (Grids)
Requires data to be structured as a matrix (e.g., a `.corr()` matrix).
```python
# Heatmap: Visualizes 2D grid intensities
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

# Clustermap: Heatmap + hierarchical clustering dendrograms
sns.clustermap(df.corr())
```

---

## 💡 Quick Tips for Parameters
*   `data`: Your Pandas DataFrame (`data=df`).
*   `x`, `y`: Column names as strings (`x="age"`).
*   `hue`: Groups data visually using different **colors**.
*   `size`: Groups data visually using different **marker sizes**.
*   `style`: Groups data visually using different **marker styles** (dots, crosses) or line styles (dashed, solid).
*   `row`, `col`: Splits the data into a grid of **subplots** based on a categorical variable (Only works in `relplot`, `displot`, `catplot`, and `lmplot`).

```python
plt.figure(figsize=(8,6))
#Histogram .continuous numerical values distributed across ranges
sns.histplot(data=df,x='bmi',bins='fd',kde=True) #'fd'-IQRbase, 'scott' sdev based
plt.axvline(df['bmi'].mean(), color='red', linestyle='--', label='Mean') #to see vertical line on mean
plt.axvline(df['bmi'].median(), color='blue', linestyle=':', label='Median')
plt.legend()

#categorical groups /count frequency/class-imbalance . count plot only needs x axis categorial variable and y axis is automatically counted
sns.countplot(data=df, x='gender_col', hue='target_col') 

#sum/mean/median of numeric col(y) vs cat col(x).estimator="mean".Bplot needs x axis categorial variable and y axis aggregate metric.default mean
sns.barplot(data=df,x='category_col', y='numeric_value_col')

#pie chart.need list as numbers, names 
counts=df['col'].value_counts()
plt.pie(counts,labels=counts.index,startangle=90,autopct='%1.1f%%')

#relationship between 2 numeric cols
sns.scatterplot(data=df,x='age',y='expenses',hue='smoker')
 
#box plot 
num_cols=df.select_dtypes(exclude='O').columns.to_list()
for col in num_cols:
  plt.figure(figsize=(6,4))
  sns.boxplot(data=df[col])
  plt.show()

#Pair Plot
# Plots everything vs everything. Warning: Filter to 4-5 core columns on large dfs!
sns.pairplot(data=df[['age', 'bmi', 'expenses', 'target_col']], hue='target_col')

# 9. Missing Data Patterns (Visual Heatmap)
# Dark lines highlight exactly where missing values are located across rows
sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()

#heatmap with corr
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
```
## 4. Advanced Column Manipulation (Lists & Strings)
```python
import ast
# 1. Convert string representation of lists "['A', 'B']" into real lists
df['list_col'] = df['string_col'].apply(ast.literal_eval)

# 2. Explode a column of lists into individual rows. uid 101 hobbies [reading,dancing,surfing] -> becomes 3 rows for 101 
df_exploded = df.explode('list_col')

# 3. String cleanup
df['text'] = df['text'].str.strip().str.lower()
```

## 5. Imputing and  Encoding
```python

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  #when not in pipeline use drop='first'
# for all 3 above line do fit_transform on train and only transform on testdata 

# Multi-Label Encoding (List of tags/types per row)
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(df['list_col']), columns=mlb.classes_, index=df.index)
df = pd.concat([df, one_hot], axis=1)

# Ordinal Encoding (Mapping ordered categories)
df['size_encoded'] = df['size'].map({'Small': 1, 'Medium': 2, 'Large': 3})
```

## 6. Outliers & Scaling
```python
# 1. IQR Outlier Filter
# Clean, fast, and completely safe  
for col in ['col1', 'col2']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

# 2. Feature Scaling (Scikit-Learn)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['col1', 'col2']] = scaler.fit_transform(X_train[['col1', 'col2']])
X_test[['col1', 'col2']] = scaler.transform(X_test[['col1', 'col2']])
```

## 7. Aggregation & Grouping
```python
df.groupby('category_col')['numeric_col'].mean()
df.groupby('category_col').agg({'numeric_col': ['mean', 'std'], 'text_col': 'count'})
pd.crosstab(df['col1'], df['col2']) # Frequency table comparing two variables
```

## 8. Train-Test Split & Class Imbalance (SMOTE)
```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. Separate features and target 
X = df.drop(columns=['target_column']) 
y = df['target_column'] 

# 2. Split dataset (Stratify preserves minority class balance in both sets) 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y 
) 

# Divide the dataset into numerical and categorical columns 
numerical_columns = X_train.select_dtypes(exclude='O').columns.to_list() 
categorical_columns = X_train.select_dtypes(include='O').columns.to_list() 

# Separate numerical and categorical columns for train and test data 
train_numerical = X_train[numerical_columns] 
test_numerical = X_test[numerical_columns] 
train_categorical = X_train[categorical_columns] 
test_categorical = X_test[categorical_columns] 

# Apply StandardScaler to numerical columns in train and test data 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 

# Apply StandardScaler on train data (fit and transform) 
train_numerical_scaled = scaler.fit_transform(train_numerical) 
# Apply the scaler on test data (transform only) 
test_numerical_scaled = scaler.transform(test_numerical) 

# Import OneHotEncoder from sklearn.preprocessing 
from sklearn.preprocessing import OneHotEncoder 
# FIX 1: Added sparse_output=False so it outputs a normal numpy array
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')   #when not in pipeline use drop='first'

# Apply OneHotEncoder on train data (fit and transform) 
train_categorical_encoded = encoder.fit_transform(train_categorical) 
# Apply the encoder on test data (transform only) 
test_categorical_encoded = encoder.transform(test_categorical) 

# Combine numerical and categorical columns in train and test data 
import numpy as np
# FIX 2: Wrapped the arrays in an extra set of parentheses (a tuple)
combined_scaled_train = np.hstack((train_numerical_scaled, train_categorical_encoded)) 
combined_scaled_test = np.hstack((test_numerical_scaled, test_categorical_encoded)) 

# 3. Apply SMOTE to training data only. NEVER RUN ON TEST DATA 
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42) 
X_train_res, y_train_res = smote.fit_resample(combined_scaled_train, y_train)

# 4. Fit the model using the SMOTE-resampled training data
m1.fit(X_train_res, y_train_res)

# 5. Predict on the scaled test data 
# (Notice we use 'combined_scaled_test' which was NOT altered by SMOTE)
pred = m1.predict(combined_scaled_test)
```

## 9. Basic Modeling & Evaluation
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res) # Using resampled data from SMOTE

# 2. Predict on test data
y_pred = model.predict(X_test)

# 3. Print Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
## 10.Pipeline
```python
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X=df.drop(columns=['y'])
y=df['y']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,stratify=y,random_state=42)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()
num_cols = X_train.select_dtypes(include=['number']).columns.to_list()

cat_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])  #when not in pipeline use drop='first'
num_transformer=Pipeline(steps=[
    ('Imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])
nb_preprocessor=ColumnTransformer(transformers=
                                  [('num',num_transformer,num_cols),
                                  ('cat',cat_transformer,cat_cols)],
                                  remainder='drop')
nb_pipe=Pipeline(steps=[("preprocessor",nb_preprocessor),
                        ("smote",SMOTE(random_state=42)),  #smote woks only with imblearn.pipeline not regular pipeline
                        ("nb_classifier",GaussianNB())])

nb_pipe.fit(X_train,y_train) #does Both fit and transform
nb_pipe_pred=nb_pipe.predict(X_test) #predict passes X_test through transform
print(classification_report(y_test,nb_pipe_pred))

#Hyper parameter tuning using Grid search
nb_param_grid = {
    'nb_classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
}
# 3. Initialize Grid Search
# Pass the entire pipeline as the estimator
grid_search = GridSearchCV(
    estimator=clf_pipeline, 
    param_grid=nb_param_grid, 
    cv=5,                     # 5-fold cross-validation
    scoring='accuracy', 
    n_jobs=-1                 # Use all available CPU cores
)
# 4. Fit on Training Data
# This safely runs the full cross-validated grid search without leakage
grid_search.fit(X_train, y_train)
# 5. Review Results & Predict
print(f"Best Parameters: {grid_search.best_params_}")

# grid_search automatically uses the best found model configuration to predict
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

## Same for Trees:
#Assemble the Master Pipeline
tree_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)), #smote woks only with imblearn.pipeline not regular pipeline
    ("tree_clf", RandomForestClassifier(random_state=42)) # Named 'tree_clf'
])
#  Define the Tree-Specific Hyperparameter Grid
# Format: pipelineStepName__hyperparameterName
param_grid = {
    'tree_clf__n_estimators':,         # Number of trees in the forest
    'tree_clf__max_depth': [5, 10, None],          # Controls tree depth (None allows max depth)
    'tree_clf__min_samples_split':,         # Minimum samples required to split a node
    'tree_clf__max_features': ['sqrt', 'log2'],    # Number of features considered at each split
    'smote__k_neighbors': [3, 5]                  # You can even tune SMOTE parameters simultaneously!
}

#Initialize and run Grid Search
grid_search = GridSearchCV(
    estimator=tree_pipe, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Evaluate
y_pred = grid_search.predict(X_test)
```
##11. Pipeline with SMOTE
```python
# CRITICAL: Import Pipeline from imblearn, NOT sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Define the pipeline with the correct sequential order
safe_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),       # Step 1: Scale features so distance math is correct
    ('smote', SMOTE(random_state=42)), # Step 2: Apply SMOTE on the cleanly scaled data
    ('model', RandomForestClassifier()) # Step 3: Run the classifier
])

# When you run fit, it scales first, then applies SMOTE, then fits the model
safe_pipeline.fit(X_train, y_train)

# When you run predict, it automatically bypasses SMOTE (keeping test data pure)
y_pred = safe_pipeline.predict(X_test)
```

# Scikit-Learn: `fit` vs `fit_transform` — Interview Cheat Sheet

> Quick reference for knowing which estimators reshape `X` and which only learn a mapping `X → y`.

---

## 1. The Mental Model (answer in 10 seconds)

Ask one question: **does calling this produce a new feature matrix?**

| Question | Answer | Methods you get |
|---|---|---|
| Does it output a modified `X`? | Yes | `fit`, `transform`, `fit_transform` |
| Does it output labels/values for `y`? | Yes | `fit`, `predict` (`predict_proba`, `score`) |
| Both? | Rare but real | all of the above (KMeans, LDA, Birch, PLS) |
| Neither — it only tags the rows it was trained on? | Yes | `fit`, `fit_predict` (DBSCAN, LOF) |

Underneath, this is just mixin inheritance:

- `TransformerMixin` → gives you `fit_transform`
- `ClassifierMixin` / `RegressorMixin` → gives you `score`, pairs with `predict`
- `ClusterMixin` → gives you `fit_predict`

An estimator can inherit more than one. **That is the source of every gotcha below.**

```
[X] ──► fit_transform() ──► [X']          # Transformer
[X, y] ──► fit() ──► [weights] ──► predict(X) ──► [ŷ]   # Predictor
```

---

## 2. Category A — Has `fit_transform` (Transformers)

### 2.1 Scalers & Normalizers

| Class | What it does | Stateful? |
|---|---|---|
| `StandardScaler` | Mean → 0, variance → 1 | Yes (learns mean/std) |
| `MinMaxScaler` | Squashes to a fixed range, default `[0, 1]` | Yes (learns min/max) |
| `RobustScaler` | Scales by IQR/median — outlier resistant | Yes (learns quartiles) |
| `MaxAbsScaler` | Divides by max absolute value, preserves sparsity | Yes |
| `Normalizer` | Scales each **row** to unit norm | **No — stateless** |
| `Binarizer` | Thresholds values to 0/1 | **No — stateless** |
| `FunctionTransformer` | Wraps any callable (e.g. `np.log1p`) | **No by default** |

> **Row vs column is the trick here.** Every scaler above works **column-wise** except `Normalizer`, which works **row-wise**. That is why `Normalizer` is stateless — it needs nothing from the training set, so it has no leakage risk.

### 2.2 Encoders & Feature Generators

| Class | Operates on | Notes |
|---|---|---|
| `OneHotEncoder` | `X` (2D) | Sparse binary columns. Always set `handle_unknown='ignore'` in production. |
| `OrdinalEncoder` | `X` (2D) | Categories → integers, for **features**. |
| `LabelEncoder` | `y` (1D only) | Signature is `fit_transform(y)` — it takes **one** argument, not `(X, y)`. |
| `LabelBinarizer` | `y` (1D) | One-hot for the target. |
| `PolynomialFeatures` | `X` | Generates `x₁²`, `x₁x₂`, … Column count explodes fast. |
| `KBinsDiscretizer` | `X` | Continuous → binned ordinal/one-hot. |

### 2.3 Dimensionality Reduction & Text

| Class | Notes |
|---|---|
| `PCA` | Unsupervised linear projection maximizing variance. Also has `inverse_transform`. |
| `TruncatedSVD` | Same idea but works on **sparse** matrices → the standard choice after TF-IDF ("LSA"). |
| `LinearDiscriminantAnalysis` | **Supervised.** Has `transform` *and* `predict` — see gotchas. |
| `CountVectorizer` | Raw text → token count matrix. |
| `TfidfVectorizer` | Raw text → TF-IDF weighted matrix. |
| `LatentDirichletAllocation` | Topic modeling. Transformer only — **a completely different "LDA."** |

### 2.4 Imputers

`SimpleImputer` (mean/median/most_frequent/constant), `KNNImputer` (neighbor-based), `IterativeImputer` (models each feature from the others; still experimental — needs `enable_iterative_imputer`).

### 2.5 Feature Selectors ← commonly forgotten category

`SelectKBest`, `SelectPercentile`, `VarianceThreshold`, `RFE`, `RFECV`, `SelectFromModel`.

They **drop columns**, so they are transformers. `SelectFromModel(RandomForestClassifier())` is the classic "wrap a predictor to make a transformer" answer.

### 2.6 Meta / Composition

`Pipeline`, `ColumnTransformer`, `FeatureUnion`.

> `Pipeline` exposes `transform`/`fit_transform` **only if its final step is a transformer**, and `predict` **only if its final step is a predictor**. It borrows the API of its last step.

---

## 3. Category B — `fit` + `predict` only (Predictors)

**Regression:** `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SVR`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `XGBRegressor`.

**Classification:** `LogisticRegression`, `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, `GaussianNB`, `DecisionTreeClassifier`, `MLPClassifier`.

Small API notes worth having ready:

- `SVC` has **no** `predict_proba` unless you pass `probability=True` (which triggers internal cross-validation and is slow).
- Tree ensembles expose `feature_importances_`, but that is an **attribute**, not a `transform`. To actually drop columns you still need `SelectFromModel`.
- `fit` returns `self`, which is what makes `model.fit(X, y).predict(X_test)` chain.

---

## 4. ⚠️ The Gotchas (this is what actually gets asked)

### 4.1 KMeans **does** have `fit_transform` — correcting a common myth

This is the single most-missed one, and it's the opposite of what most cheat sheets say.

```python
km = KMeans(n_clusters=3)
X_dist = km.fit_transform(X)   # shape: (n_samples, n_clusters)
```

`KMeans.transform` projects each sample into **cluster-distance space** — column *j* is the Euclidean distance from that sample to centroid *j*. So KMeans has the full set: `fit`, `transform`, `fit_transform`, `predict`, `fit_predict`.

This is a real, useful technique: cluster distances as engineered features feeding a downstream classifier.

`MiniBatchKMeans` and `Birch` behave the same way.

### 4.2 The clustering split — who can handle unseen data?

| Model | `transform` | `predict` | `fit_predict` | Why |
|---|---|---|---|---|
| `KMeans` | ✅ | ✅ | ✅ | Stores centroids → new points can be assigned |
| `Birch` | ✅ | ✅ | ✅ | Stores a CF-tree |
| `GaussianMixture` | ❌ | ✅ | ✅ | Stores distributions, but doesn't reshape `X` |
| `DBSCAN` | ❌ | ❌ | ✅ | Density is local to the training set — no model to reuse |
| `AgglomerativeClustering` | ❌ | ❌ | ✅ | Hierarchy is built over the given points only |

**One-liner:** *"Centroid- and distribution-based clusterers can score new data; density- and linkage-based ones cannot, which is why they only expose `fit_predict`."*

### 4.3 Anomaly detection

- `IsolationForest` — `fit`, `predict`, `fit_predict`, `decision_function`. Returns `1` / `-1`, not `0` / `1`.
- `LocalOutlierFactor` — `fit_predict` only by default. It gains `predict` **only** if constructed with `novelty=True`.

### 4.4 Estimators that are both transformer and predictor

| Model | Why |
|---|---|
| `LinearDiscriminantAnalysis` | Reduces to ≤ `n_classes - 1` dims **and** classifies |
| `KMeans` / `Birch` | Cluster-distance space **and** cluster assignment |
| `PLSRegression` | Projects to latent components **and** regresses |

### 4.5 "It has `fit` but neither of the others"

`NearestNeighbors` — `fit` then `kneighbors()`. No `predict`, no `transform`. It's a lookup index, not a model.

---

## 5. The `.transform()` Discipline

```python
# ✅ Correct
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ❌ Leakage — test statistics bleed into the scaler
X_test_s = scaler.fit_transform(X_test)
```

Note this only matters for **stateful** transformers. `Normalizer` and `FunctionTransformer(np.log1p)` learn nothing from the training set, so calling `fit_transform` on the test set is harmless there — but use `transform` uniformly anyway so the habit never breaks.

Also: transformers accept `fit(X, y=None)`. The `y` is ignored, and exists purely so `Pipeline` can pass the same signature to every step.

---

## 6. Interview Defense Scripts

**On leakage:**
> "I call `fit_transform` only on training data. Validation and test data get `.transform()` so no distribution statistics from unseen data leak into the fitted parameters. With cross-validation I put the transformer inside a `Pipeline` so it's refit on each fold's training split rather than once on the full set."

**On pipelines:**
> "In production I wrap transformers and the estimator in a `Pipeline`, with `ColumnTransformer` handling numeric and categorical branches. During `pipeline.fit()` intermediate steps get `fit_transform`; during `predict()` they get `transform`. That makes the leakage rule structural instead of something I have to remember."

**On `fit_transform` efficiency** — *state this precisely, the sloppy version is wrong:*
> "By default `TransformerMixin.fit_transform` is literally `fit(X).transform(X)`, so there's no automatic speedup. But some estimators override it for a real one — `PCA` reuses the SVD it already computed, and `CountVectorizer`/`TfidfVectorizer` avoid a second pass over the corpus. So I prefer `fit_transform` on training data: never slower, sometimes meaningfully faster, and clearer about intent."

**If asked "does KMeans have `transform`?"**
> "Yes — it maps samples into cluster-distance space, one column per centroid. People assume it doesn't because clustering feels like a prediction task, but KMeans inherits from both `ClusterMixin` and `TransformerMixin`."

---

## 7. Reference Python

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

num_cols = ["age", "income"]
cat_cols = ["city", "plan"]

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", numeric,     num_cols),
    ("cat", categorical, cat_cols),
])

clf = Pipeline([
    ("pre",   pre),
    ("model", RandomForestClassifier(random_state=0)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf.fit(X_train, y_train)          # transformers: fit_transform
clf.predict(X_test)                # transformers: transform  ← leakage-safe

# Refit inside every fold — the reason to use a Pipeline at all
cross_val_score(clf, X_train, y_train, cv=5)
```

```python
# KMeans as a feature generator, not a predictor
km = KMeans(n_clusters=8, n_init="auto", random_state=0)
dist_train = km.fit_transform(X_train)   # (n_samples, 8)
dist_test  = km.transform(X_test)        # .transform, not .fit_transform
```

---

## 8. One-Page Recall Table

| Estimator | `transform` | `predict` | `fit_predict` |
|---|:--:|:--:|:--:|
| `StandardScaler` / `PCA` / `SimpleImputer` | ✅ | ❌ | ❌ |
| `TfidfVectorizer` / `SelectKBest` | ✅ | ❌ | ❌ |
| `LogisticRegression` / `RandomForest` / `SVC` | ❌ | ✅ | ❌ |
| `KMeans` / `Birch` | ✅ | ✅ | ✅ |
| `LinearDiscriminantAnalysis` | ✅ | ✅ | ❌ |
| `GaussianMixture` | ❌ | ✅ | ✅ |
| `IsolationForest` | ❌ | ✅ | ✅ |
| `DBSCAN` / `AgglomerativeClustering` | ❌ | ❌ | ✅ |
| `LocalOutlierFactor` | ❌ | only if `novelty=True` | ✅ |
| `NearestNeighbors` | ❌ | ❌ | ❌ |
