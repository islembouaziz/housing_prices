import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def create_features(df):
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['TotalBath'] = df.get('FullBath', 0) + (0.5 * df.get('HalfBath', 0)) + df.get('BsmtFullBath', 0) + (0.5 * df.get('BsmtHalfBath', 0))
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalPorchSF'] = df.get('OpenPorchSF', 0) + df.get('3SsnPorch', 0) + df.get('EnclosedPorch', 0) + df.get('ScreenPorch', 0) + df.get('WoodDeckSF', 0)
    return df

print("Loading data...")
house_data_training = pd.read_csv('data/train.csv', index_col='Id')
house_data_training = create_features(house_data_training)

y = house_data_training.SalePrice
X = house_data_training.drop(['SalePrice'], axis=1)

categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
numerical_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = RandomForestRegressor(n_estimators=510, max_depth=30, random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

print("Training model...")
pipeline.fit(X, y)

# Save the pipeline and column names
with open('api/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save the column names so the API knows what features to expect
with open('api/columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# Save training data stats for "Find House Profile" feature
stats = house_data_training.copy()
with open('api/training_data.pkl', 'wb') as f:
    pickle.dump(stats, f)

print("[OK] Model trained and saved to api/model.pkl")
print(f"   Features: {len(X.columns)} columns")
