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
df=df.dropna(axis=0) # Drop rows with any missing values
df=df.dropna(subset=['col']) # Drop rows if missing in specific column
df['col'] = df['col'].fillna(df['col'].median()) # Impute using median
df = df.ffill()       # Forward fill (useful for time-series)
```

## 3. Data Transformation & Cleaning
```python
df.drop_duplicates(inplace=True)               # Remove duplicate rows
df['col'] = df['col'].astype('int64')          # Convert data types
df.rename(columns={'old':'new'}, inplace=True) # Rename columns
df.drop(columns=['col1', 'col2'], inplace=True)# Drop columns
```
## EDA
```python
#Basic EDA commands 
plt.figure(figsize=(8,6))

#Histogram
sns.histplot(data=df,x='bmi',bins='fd',kde=True) #'fd'-IQRbase, 'scott' sdev based
plt.axvline(df['bmi'].mean(), color='red', linestyle='--', label='Mean') #to see vertical line on mean
plt.axvline(df['bmi'].median(), color='blue', linestyle=':', label='Median')
plt.legend()

#Countplot (Categorical Frequency & Class Imbalance)
sns.countplot(data=df, x='gender_col', hue='target_col') 

#barplot
sns.barplot(data=df,x='category_col', y='numeric_value_col')

#pie chart
counts=df['col'].value_counts()
plt.pie(counts,labels=counts.index,startangle=90,autopct='%1.1f%%')

#scatter
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
    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
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
    ("smote", SMOTE(random_state=42)),
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
