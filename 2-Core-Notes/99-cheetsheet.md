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

```

## 2. Handling Missing Data (NaNs)
```python
df.isna().sum()              # Count missing values per column
df.dropna(axis=0, inplace=True) # Drop rows with any missing values
df.dropna(subset=['col'], inplace=True) # Drop rows if missing in specific column
df['col'].fillna(df['col'].median(), inplace=True) # Impute using median
df.ffill(inplace=True)       # Forward fill (useful for time-series)
```

## 3. Data Transformation & Cleaning
```python
df.drop_duplicates(inplace=True)               # Remove duplicate rows
df['col'] = df['col'].astype('int64')          # Convert data types
df.rename(columns={'old':'new'}, inplace=True) # Rename columns
df.drop(columns=['col1', 'col2'], inplace=True)# Drop columns
```

## 4. Advanced Column Manipulation (Lists & Strings)
```python
import ast
# 1. Convert string representation of lists "['A', 'B']" into real lists
df['list_col'] = df['string_col'].apply(ast.literal_eval)

# 2. Explode a column of lists into individual rows
df_exploded = df.explode('list_col')

# 3. String cleanup
df['text'] = df['text'].str.strip().str.lower()
```

## 5. Categorical Encoding
```python
# Nominal Encoding (One-Hot / Dummy Variables)
df = pd.get_dummies(df, columns=['cat_col'], drop_first=True)

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
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[~((df['col'] < (Q1 - 1.5 * IQR)) | (df['col'] > (Q3 + 1.5 * IQR)))]

# 2. Feature Scaling (Scikit-Learn)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['col1', 'col2']] = scaler.fit_transform(df[['col1', 'col2']])
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

# 3. Apply SMOTE to training data only (Prevents data leakage into test set)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
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
