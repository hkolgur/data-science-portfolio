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

## 5. Imputing and  Encoding
```python

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
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
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 

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
    ('encoder',OneHotEncoder(drop='first', handle_unknown='ignore',sparse_output=False))])
num_transformer=Pipeline(steps=[
    ('Imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())])
nb_preprocessor=ColumnTransformer(transformers=
                                  [('num',num_transformer,num_cols),
                                  ('cat',cat_transformer,cat_cols)],
                                  remainder='drop')
nb_pipe=Pipeline(steps=[("preprocessor",nb_preprocessor),
                        ("smote",SMOTE(random_state=42)),
                        ("nb_classifier",GaussianNB())])

nb_pipe.fit(X_train,y_train) #does Both fit and transform
nb_pipe_pred=nb_pipe.predict(X_test) #predict passes X_test through transform
print(classification_report(y_test,nb_pipe_pred))

#Hyper parameter tuning using Grid search
# 3. Initialize Grid Search
# Pass the entire pipeline as the estimator
grid_search = GridSearchCV(
    estimator=clf_pipeline, 
    param_grid=param_grid, 
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
