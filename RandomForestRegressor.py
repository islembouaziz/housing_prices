import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer

# Load data
house_data_testing = pd.read_csv('data/test.csv', index_col='Id')
house_data_training = pd.read_csv('data/train.csv', index_col='Id')

# --- FEATURE ENGINEERING ---
def create_features(df):
    df = df.copy()
    # 1. Total Square Footage
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    
    # 2. Total Bathrooms (combining half and full baths from both floors)
    df['TotalBath'] = df.get('FullBath', 0) + (0.5 * df.get('HalfBath', 0)) + df.get('BsmtFullBath', 0) + (0.5 * df.get('BsmtHalfBath', 0))
    
    # 3. House Age (How old the house was when it was sold)
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # 4. Total Porch/Deck Space
    df['TotalPorchSF'] = df.get('OpenPorchSF', 0) + df.get('3SsnPorch', 0) + df.get('EnclosedPorch', 0) + df.get('ScreenPorch', 0) + df.get('WoodDeckSF', 0)
    
    return df

house_data_training = create_features(house_data_training)
house_data_testing = create_features(house_data_testing)
# ---------------------------

# Separate target from predictors
y = house_data_training.SalePrice
X = house_data_training.drop(['SalePrice'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Select categorical and numerical columns
categorical_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
numerical_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, preds)
print('Mean Absolute Error (MAE):', mae)

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual Price': y_valid, 'Predicted Price': preds.round(0)})
comparison['Difference'] = abs(comparison['Actual Price'] - comparison['Predicted Price'])
#print("\nComparison of Actual vs Predicted (First 15 houses):")
#print(comparison.head(15))

# --- Optimizing the Model: Pipelines + Number of Trees + Cross-Validation ---
# This loop safely tests different numbers of trees using 5-fold Cross-Validation
print("\n--- Testing different Hyperparameters (Trees & Depth) with Cross-Validation ---")
print("This might take a moment to calculate...\n")

def get_cv_mae(n_estimators, max_depth, X, y, preprocessor):
    # 1. Create the base model
    test_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    
    # 2. Match it with the pipeline (No log transform needed for Random Forest!)
    test_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', test_model)])
    
    # 3. Get the cross-validation average MAE
    scores = -1 * cross_val_score(test_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

# We test 640 trees (your best number) against different max depths

avg_mae = get_cv_mae(510, 30, X, y, preprocessor)
print(f"Trees: {510} | Max Depth: {30} \t\t Average CV MAE: {avg_mae:.2f}")

#Optional: Generate test predictions for submission
preds_test = my_pipeline.predict(house_data_testing)
output = pd.DataFrame({'Id': house_data_testing.index, 'SalePrice': preds_test})
output.to_csv('submissionRandomForestRegressor.csv', index=False)