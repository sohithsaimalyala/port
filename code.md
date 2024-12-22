# Project Code



```python
# Import necessary libraries
import sqlite3
import pandas as pd

# Step 1: Connect to the database
db_path = "housing.db"  # Path to the SQLite database
conn = sqlite3.connect(db_path)

# Step 2: SQL join query
query = """
SELECT 
    pd.rowid AS PropertyID,
    pd.Area,
    pd.Bedrooms,
    pd.Bathrooms,
    pd.Stories,
    pd.Parking,
    pd.Price,
    a.MainRoad,
    a.GuestRoom,
    a.Basement,
    a.HotWaterHeating,
    a.AirConditioning,
    a.PrefArea,
    f.FurnishingStatus
FROM PropertyDetails pd
JOIN Amenities a ON pd.rowid = a.rowid
JOIN Furnishing f ON pd.rowid = f.rowid
"""

# Step 3: Fetch data into Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Step 4: Close the database connection
conn.close()

# Step 5: Display the DataFrame
print(df.head())
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data_path = "Housing.csv"
data = pd.read_csv(data_path)

# Step 2: Explore the target variable
target = 'price'  # Assuming 'price' is the target variable
print(f"Target Column: {target}")
print(data[target].describe())

# Plot the distribution of the target variable
plt.figure(figsize=(10, 6))
plt.hist(data[target], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Target Variable Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Check for categorical variables for potential stratification
categorical_features = ['furnishingstatus']
for feature in categorical_features:
    print(f"\nDistribution of {feature}:")
    print(data[feature].value_counts())
    data[feature].value_counts().plot(kind='bar', title=f"Distribution of {feature}")
    plt.show()

# Step 3: Determine the need for stratification
# If 'furnishingstatus' is imbalanced, stratify by it
stratify_col = 'furnishingstatus' if data['furnishingstatus'].nunique() > 1 else None

# Step 4: Perform train/test split
X = data.drop(columns=[target])  # Features
y = data[target]                 # Target variable

if stratify_col:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=data[stratify_col]
    )
    print(f"Data stratified by {stratify_col}")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data split without stratification")

# Step 5: Output summary
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# Save the split datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
```


```python
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure inline plotting for Jupyter Notebook
%matplotlib inline

# Load the dataset
data_path = "Housing.csv"
data = pd.read_csv(data_path)

# Step 1: Convert `furnishingstatus` to numerical values
if 'furnishingstatus' in data.columns:
    data['furnishingstatus'] = data['furnishingstatus'].map({
        'unfurnished': 0,
        'semi-furnished': 1,
        'furnished': 2
    })
    print("Furnishingstatus column converted to numerical values.")
else:
    print("Column 'furnishingstatus' not found in the dataset.")

# Convert `yes`/`no` values to `1`/`0`
for col in data.select_dtypes(include=['object']).columns:
    if data[col].str.lower().isin(['yes', 'no']).any():
        data[col] = data[col].map({'yes': 1, 'no': 0})
        print(f"Column '{col}' converted to numerical values.")

# Step 2: Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 3: Extract and process top correlations
# Get the correlation matrix as a DataFrame
correlation_df = correlation_matrix.reset_index().melt(id_vars='index', var_name='Feature2', value_name='Correlation')

# Rename columns for clarity
correlation_df.rename(columns={'index': 'Feature1'}, inplace=True)

# Remove self-correlations
correlation_df = correlation_df[correlation_df['Feature1'] != correlation_df['Feature2']]

# Sort by absolute correlation values in descending order
correlation_df['AbsCorrelation'] = correlation_df['Correlation'].abs()
top_correlations = correlation_df.sort_values(by='AbsCorrelation', ascending=False).head(10)

# Display top correlations
print("\nTop correlations (excluding self-correlations):")
print(top_correlations[['Feature1', 'Feature2', 'Correlation']])


```
```python
#Experiment1

import os
import mlflow

# Set up MLFlow tracking URI and authentication
MLFLOW_TRACKING_URI = "https://dagshub.com/sohithsaimalyala/Project.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sohithsaimalyala'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '29f357e14d4829f0c3e67f7e44b6391e7984e0cd'

# Configure MLFlow
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

# Experiment name
experiment_name = "Housing_Prediction"

# Set or create experiment
try:
    mlflow.set_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    print(f"Experiment '{experiment_name}' does not exist. Attempting to create it.")
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id)

# Start an MLflow run
with mlflow.start_run():
    mlflow.log_param("example_param", 42)
    mlflow.log_metric("example_metric", 0.99)
    print("Run logged successfully.")

```
```python
#Experiment2

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import pandas as pd

# Function to convert categorical values like 'yes'/'no' and 'furnished' to integers
def preprocess_categorical_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Check for categorical columns
            # Convert 'yes'/'no' to 1/0
            if df[col].isin(['yes', 'no']).any():
                df[col] = df[col].map({'yes': 1, 'no': 0})
            
            # Convert 'furnished', 'semi-furnished', 'unfurnished' to 0, 1, 2
            if df[col].isin(['unfurnished', 'semi-furnished', 'furnished']).any():
                df[col] = df[col].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})
    return df

# Apply conversion to training and test data
X_train = preprocess_categorical_columns(X_train)
X_test = preprocess_categorical_columns(X_test)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), X_train.select_dtypes(include=['object']).columns)  # One-hot encode categorical data
    ]
)

# Use XGBRegressor for regression tasks
regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

# Create the pipeline with preprocessing and regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# Start the MLFlow run
name = "experiment2_housing"  # Set your run name
with mlflow.start_run(run_name=name):
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = pipeline.predict(X_test)
    
    # Log the model and metrics with MLFlow
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Log evaluation metrics
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mlflow.log_metric("RMSE", rmse)

    # Log hyperparameters
    mlflow.log_param("objective", "reg:squarederror")

    print("Model training and logging completed.")
```
```python
#Experiment3

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Assume you have already loaded the data (X_train, y_train, X_test, y_test)

# Step 1: Feature Engineering

# Example: Create new features by combining existing ones (e.g., ratios, differences, interaction terms)
def feature_engineering(df):
    # Combine existing features (e.g., creating a new ratio feature)
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-5)  # Avoid division by zero
    
    # Example of creating an interaction feature
    if 'feature3' in df.columns and 'feature4' in df.columns:
        df['interaction_feature'] = df['feature3'] * df['feature4']
    
    # Example: Create polynomial features (e.g., squared term)
    if 'feature5' in df.columns:
        df['feature5_squared'] = df['feature5'] ** 2
    
    return df

# Apply feature engineering to training and test data
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

# Step 2: Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Step 3: Initialize the model (XGBRegressor for regression)
model = XGBRegressor(objective='reg:squarederror')

# Step 4: Create the pipeline with preprocessing and regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Step 5: Start MLFlow run to log the model and results
with mlflow.start_run(run_name="feature_engineering_experiment"):
    # Step 6: Train the model
    pipeline.fit(X_train, y_train)
    
    # Step 7: Predictions
    y_pred_test = pipeline.predict(X_test)
    
    # Step 8: Log model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Step 9: Evaluate model performance (e.g., RMSE, MAE)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    # Step 10: Log metrics in MLFlow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MSE", mse)
    
    # Step 11: Log parameters if applicable (e.g., model parameters, feature engineering steps)
    mlflow.log_param("model_type", "XGBRegressor")
    mlflow.log_param("feature_engineering_steps", "feature_ratio, interaction_feature, feature5_squared")

    # Log any other parameters relevant to the experiment
    # Example: Logging hyperparameters (optional)
    mlflow.log_param("learning_rate", model.get_params().get('learning_rate'))
    mlflow.log_param("max_depth", model.get_params().get('max_depth'))
```
```python
#Experiment4


import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression

# Assuming X_train, X_test, y_train, y_test are already loaded

# Step 1: Perform Correlation Threshold feature selection

def correlation_threshold(X, threshold=0.9):
    # Compute the correlation matrix
    corr_matrix = X.corr().abs()
    
    # Select upper triangle of correlation matrix to check for duplicate correlations
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Drop the correlated features
    X_filtered_corr = X.drop(columns=to_drop)
    
    return X_filtered_corr, to_drop

# Apply correlation threshold to X_train and X_test
X_train_corr, dropped_corr = correlation_threshold(X_train, threshold=0.9)
X_test_corr = X_test[X_train_corr.columns]  # Make sure test set has the same columns after dropping

# Step 2: Perform Feature Importance-based selection using XGBRegressor

# Train an XGBRegressor to get feature importances
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train_corr, y_train)

# Get the feature importances
feature_importances = model.feature_importances_

# Define the threshold for feature importance (e.g., keep features with importance > 0.01)
important_features = X_train_corr.columns[feature_importances > 0.01]
X_train_imp = X_train_corr[important_features]
X_test_imp = X_test_corr[important_features]

# Step 3: Perform Variance Threshold feature selection

# Variance threshold to remove features with low variance
variance_threshold = VarianceThreshold(threshold=0.01)  # 0.01 is a typical threshold, adjust as needed
X_train_var = variance_threshold.fit_transform(X_train_imp)
X_test_var = variance_threshold.transform(X_test_imp)

# Convert the result back to DataFrame
X_train_var_df = pd.DataFrame(X_train_var, columns=important_features[variance_threshold.get_support()])
X_test_var_df = pd.DataFrame(X_test_var, columns=important_features[variance_threshold.get_support()])

# Step 4: Create and train the model pipeline after feature selection

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train_var_df.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train_var_df.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Create the pipeline with preprocessing and regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Step 5: Start MLFlow run to log the model and results
with mlflow.start_run(run_name="feature_selection_experiment"):
    # Step 6: Train the model
    pipeline.fit(X_train_var_df, y_train)
    
    # Step 7: Predictions
    y_pred_test = pipeline.predict(X_test_var_df)
    
    # Step 8: Log model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Step 9: Evaluate model performance (e.g., RMSE, MSE)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    # Step 10: Log metrics in MLFlow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MSE", mse)
    
    # Step 11: Log feature selection details
    mlflow.log_param("feature_selection_method", "Correlation Threshold, Feature Importance, Variance Threshold")
    mlflow.log_param("correlation_threshold", 0.9)
    mlflow.log_param("feature_importance_threshold", 0.01)
    mlflow.log_param("variance_threshold", 0.01)

    # Log the dropped features for correlation and variance threshold
    mlflow.log_param("dropped_features_corr", str(dropped_corr))
    mlflow.log_param("remaining_features_after_importance", str(important_features.tolist()))
    mlflow.log_param("remaining_features_after_variance", str(X_train_var_df.columns.tolist()))
```
```python
#Experiment5


import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Assuming you have already loaded the data (X_train, X_test, y_train, y_test)
# Example data: X_train, y_train, X_test, y_test

# Step 1: Preprocess the data (scaling features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Step 2: Apply PCA for dimensionality reduction
pca = PCA()

# Fit PCA on the training data (after preprocessing)
X_train_scaled = preprocessor.fit_transform(X_train)
pca.fit(X_train_scaled)

# Step 3: Plot the scree plot to show the explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title("Scree Plot")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.show()

# Step 4: Determine the number of components to select based on the cumulative variance (e.g., 95% explained variance)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1  # Choose the number of components that explain 95% variance

print(f"Number of components selected: {num_components}")

# Step 5: Create the pipeline with PCA and XGBRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=num_components)),
    ('regressor', XGBRegressor(objective='reg:squarederror'))
])

# Step 6: Start MLFlow run to log the model and results
with mlflow.start_run(run_name="pca_experiment"):
    # Step 7: Train the model
    pipeline.fit(X_train, y_train)
    
    # Step 8: Predictions
    y_pred_test = pipeline.predict(X_test)
    
    # Step 9: Log the model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Step 10: Evaluate model performance (e.g., RMSE, MSE)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    # Step 11: Log metrics in MLFlow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MSE", mse)
    
    # Step 12: Log PCA details
    mlflow.log_param("num_components_selected", num_components)
    mlflow.log_param("explained_variance_threshold", 0.95)

    # Log the cumulative explained variance for analysis
    mlflow.log_metric("cumulative_explained_variance", cumulative_variance[-1])

    # Save the scree plot as an image and log it in MLFlow
    scree_plot_path = "scree_plot.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.title("Scree Plot")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.savefig(scree_plot_path)
    mlflow.log_artifact(scree_plot_path)

    print(f"Model trained with {num_components} components and logged to MLFlow.")
```


```python
#Experiment6

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming you have already loaded the data (X_train, X_test, y_train, y_test)
# Example data: X_train, y_train, X_test, y_test

# Step 1: Feature Engineering - Create new interaction and polynomial features
def feature_engineering(df):
    # Ensure feature3 and feature4 exist before applying PolynomialFeatures
    if 'feature3' in df.columns and 'feature4' in df.columns:
        # Create polynomial features (quadratic features for example)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[['feature3', 'feature4']])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['feature3', 'feature4']))
        df = pd.concat([df, poly_df], axis=1)
    else:
        print("Warning: 'feature3' and/or 'feature4' are missing from the dataset.")

    # Create interaction terms between two features (make sure they exist)
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['feature_interaction'] = df['feature1'] * df['feature2']
    else:
        print("Warning: 'feature1' and/or 'feature2' are missing from the dataset.")
    
    return df

# Apply feature engineering to training and test data
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

# Step 2: Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Step 3: Define models to compare: XGBRegressor, RandomForestRegressor, and LinearRegression
models = {
    'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearRegression': LinearRegression()
}

# Step 4: Define hyperparameter grid for GridSearchCV
param_grid = {
    'XGBRegressor': {
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 6, 9],
        'regressor__n_estimators': [100, 200]
    },
    'RandomForestRegressor': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    },
    'LinearRegression': {
        # Linear Regression doesn't have hyperparameters to tune, but can still be included for comparison
        'regressor__fit_intercept': [True, False]
    }
}

# Step 5: Train models using GridSearchCV to optimize hyperparameters
best_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Perform GridSearchCV with n_jobs=1 to avoid parallelism and serialization issues
    grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, n_jobs=1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Store the best model for later comparison
    best_models[model_name] = grid_search.best_estimator_
    
    # Log the model and hyperparameters in MLFlow
    with mlflow.start_run(run_name=f'{model_name}_experiment'):
        mlflow.sklearn.log_model(grid_search.best_estimator_, f'{model_name}_model')
        mlflow.log_params(grid_search.best_params_)
        
        # Step 6: Predictions and Evaluation
        y_pred_test = grid_search.best_estimator_.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        
        print(f"{model_name} RMSE: {rmse}")
        print(f"{model_name} MSE: {mse}")

# Step 7: Compare the models' performance (display results)
for model_name, best_model in best_models.items():
    print(f"Best {model_name} model: {best_model}")
```


```python
#Experiment7



import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming X_train, X_test, y_train, y_test are already loaded
# Example data: X_train, y_train, X_test, y_test

# Step 1: Feature Engineering - Optional (You can add feature engineering here)
def feature_engineering(df):
    # Example feature engineering: you can modify or create new features here if needed
    return df

# Apply feature engineering if necessary
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

# Step 2: Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Step 3: Define models to compare: XGBRegressor, RandomForestRegressor, and LinearRegression
models = {
    'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearRegression': LinearRegression()
}

# Step 4: Define the number of features to select in RFE (e.g., select 10 features)
n_features_to_select = 10

# Step 5: Train models using RFE for feature selection and evaluate performance
best_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Create the pipeline with RFE for feature selection
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Step 6: Predictions and Evaluation
    y_pred_test = pipeline.predict(X_test)
    
    # Calculate RMSE, MAE, and R²
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    # Log the model and hyperparameters in MLFlow
    with mlflow.start_run(run_name=f'{model_name}_experiment_with_RFE'):
        mlflow.sklearn.log_model(pipeline, f'{model_name}_model')
        mlflow.log_param("n_features_to_select", n_features_to_select)
        mlflow.log_params({'model_type': model_name})
        
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        print(f"{model_name} RMSE: {rmse}")
        print(f"{model_name} MAE: {mae}")
        print(f"{model_name} R²: {r2}")

    # Store the best model for later comparison
    best_models[model_name] = pipeline

# Step 7: Compare the models' performance (display results)
for model_name, best_model in best_models.items():
    print(f"Best {model_name} model: {best_model}")
```


```python
#F1ScorePlots


import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer

# Assuming you have already loaded the data (X_train, X_test, y_train, y_test)
# Example data: X_train, y_train, X_test, y_test

# Step 1: Feature Engineering - Optional (You can add feature engineering here if needed)
def feature_engineering(df):
    # Example feature engineering: you can modify or create new features here if needed
    return df

# Apply feature engineering if necessary
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)

# Step 2: Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(include=['float64', 'int64']).columns),  # Scale numerical features
        ('cat', OneHotEncoder(), X_train.select_dtypes(include=['object']).columns)  # Encode categorical features
    ]
)

# Step 3: Define models to compare: XGBRegressor, RandomForestRegressor, and LinearRegression
models = {
    'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearRegression': LinearRegression()
}

# Step 4: Define the number of features to select in RFE (e.g., select 10 features)
n_features_to_select = 10

# Step 5: Train models using RFE for feature selection and evaluate F1-score
best_models = {}
f1_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Create the pipeline with RFE for feature selection
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Step 6: Predictions and Evaluation (Convert y_test into categories for classification task)
    y_pred_test = pipeline.predict(X_test)

    # Convert continuous values into discrete classes (for demonstration purposes)
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    y_test_binned = discretizer.fit_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_test_binned = discretizer.transform(y_pred_test.reshape(-1, 1)).flatten()

    # Compute F1-score
    f1 = f1_score(y_test_binned, y_pred_test_binned, average='weighted')
    
    # Log the model and hyperparameters in MLFlow
    with mlflow.start_run(run_name=f'{model_name}_experiment_with_RFE'):
        mlflow.sklearn.log_model(pipeline, f'{model_name}_model')
        mlflow.log_param("n_features_to_select", n_features_to_select)
        mlflow.log_params({'model_type': model_name})
        
        mlflow.log_metric("F1-Score", f1)
        
        print(f"{model_name} F1-Score: {f1}")
        
    # Store the best model for later comparison
    best_models[model_name] = pipeline
    f1_scores[model_name] = f1

# Step 7: Compare the models' performance (F1-scores plot)
model_names = list(f1_scores.keys())
f1_values = list(f1_scores.values())

plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_values, color='skyblue')
plt.xlabel("Model")
plt.ylabel("F1-Score")
plt.title("F1-Score Comparison of Models")
plt.show()

# Step 8: Log the best model based on F1-Score
best_model_name = max(f1_scores, key=f1_scores.get)
print(f"The best model based on F1-Score is: {best_model_name}")
```


```python

import joblib

# Save the best model using joblib
best_model_name = max(f1_scores, key=f1_scores.get)  # Get the model with the best F1-score
best_model = best_models[best_model_name]

# Save the model to a file
joblib.dump(best_model, 'best_model.joblib')
print(f"Model saved as 'best_model.joblib'")
```


```python
from fastapi import FastAPI
from pydantic import BaseModel

# Define your input model
class YourInputModel(BaseModel):
    size: float
    rooms: int
    location: str

app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application"}

# Predict endpoint
@app.post("/predict")
def predict(data: YourInputModel):
    # Example prediction logic
    prediction = (data.size * 1000) + (data.rooms * 500)  # Replace with your actual model logic
    return {"prediction": prediction}
```
