import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Housing Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and data at startup
with open("api/model.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("api/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("api/training_data.pkl", "rb") as f:
    training_data = pickle.load(f)


class HouseFeatures(BaseModel):
    OverallQual: int = 5          # Overall Quality (1-10)
    GrLivArea: int = 1500         # Above ground living area (sq ft)
    TotalBsmtSF: int = 800        # Total basement area (sq ft)
    YearBuilt: int = 2000         # Year built
    YrSold: int = 2010            # Year sold
    GarageCars: int = 2           # Garage capacity (cars)
    FullBath: int = 2             # Full bathrooms
    HalfBath: int = 0             # Half bathrooms
    BsmtFullBath: int = 0         # Basement full bathrooms
    BsmtHalfBath: int = 0         # Basement half bathrooms
    TotRmsAbvGrd: int = 7         # Total rooms above grade
    Fireplaces: int = 1           # Number of fireplaces
    LotArea: int = 8500           # Lot area (sq ft)
    Neighborhood: str = "NAmes"   # Neighborhood
    MSZoning: str = "RL"          # Zoning classification
    HouseStyle: str = "1Story"    # House style
    ExterQual: str = "TA"         # Exterior quality
    KitchenQual: str = "TA"       # Kitchen quality
    BsmtQual: str = "TA"          # Basement quality
    GarageArea: int = 480         # Garage area (sq ft)
    OpenPorchSF: int = 40         # Open porch area
    WoodDeckSF: int = 0           # Wood deck area
    EnclosedPorch: int = 0
    ScreenPorch: int = 0
    SSSnPorch: int = 0
    Foundation: str = "PConc"
    CentralAir: str = "Y"
    SaleType: str = "WD"
    SaleCondition: str = "Normal"


def build_input_df(features: HouseFeatures) -> pd.DataFrame:
    """Build a full-featured DataFrame row from user inputs."""
    data = {col: [np.nan] for col in model_columns}
    df = pd.DataFrame(data)

    # Fill in user-provided values
    mapping = {
        'OverallQual': features.OverallQual,
        'GrLivArea': features.GrLivArea,
        'TotalBsmtSF': features.TotalBsmtSF,
        'YearBuilt': features.YearBuilt,
        'YrSold': features.YrSold,
        'GarageCars': features.GarageCars,
        'FullBath': features.FullBath,
        'HalfBath': features.HalfBath,
        'BsmtFullBath': features.BsmtFullBath,
        'BsmtHalfBath': features.BsmtHalfBath,
        'TotRmsAbvGrd': features.TotRmsAbvGrd,
        'Fireplaces': features.Fireplaces,
        'LotArea': features.LotArea,
        'GarageArea': features.GarageArea,
        'OpenPorchSF': features.OpenPorchSF,
        'WoodDeckSF': features.WoodDeckSF,
        'EnclosedPorch': features.EnclosedPorch,
        'ScreenPorch': features.ScreenPorch,
        '3SsnPorch': features.SSSnPorch,
        '1stFlrSF': features.GrLivArea,
        '2ndFlrSF': 0,
        'Neighborhood': features.Neighborhood,
        'MSZoning': features.MSZoning,
        'HouseStyle': features.HouseStyle,
        'ExterQual': features.ExterQual,
        'KitchenQual': features.KitchenQual,
        'BsmtQual': features.BsmtQual,
        'Foundation': features.Foundation,
        'CentralAir': features.CentralAir,
        'SaleType': features.SaleType,
        'SaleCondition': features.SaleCondition,
    }
    for col, val in mapping.items():
        if col in df.columns:
            df[col] = val

    # Add engineered features
    df['TotalSF'] = features.TotalBsmtSF + features.GrLivArea
    df['TotalBath'] = features.FullBath + 0.5 * features.HalfBath + features.BsmtFullBath + 0.5 * features.BsmtHalfBath
    df['HouseAge'] = features.YrSold - features.YearBuilt
    df['TotalPorchSF'] = features.OpenPorchSF + features.WoodDeckSF + features.EnclosedPorch + features.ScreenPorch + features.SSSnPorch

    return df


@app.get("/")
def root():
    return {"status": "Housing Price Predictor API is running"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    """Predict the price of a house given its features."""
    df = build_input_df(features)
    prediction = pipeline.predict(df)[0]
    house_age = features.YrSold - features.YearBuilt
    total_sf = features.TotalBsmtSF + features.GrLivArea

    return {
        "predicted_price": round(float(prediction), 2),
        "summary": {
            "house_age": house_age,
            "total_sf": total_sf,
            "price_per_sqft": round(float(prediction) / total_sf, 2) if total_sf > 0 else 0
        }
    }


@app.get("/neighborhoods")
def get_neighborhoods():
    """Return list of available neighborhoods."""
    return {"neighborhoods": sorted(training_data['Neighborhood'].dropna().unique().tolist())}


@app.get("/find-by-budget/{budget}")
def find_by_budget(budget: float, tolerance: float = 0.15):
    """Find houses in the training data closest to a given budget."""
    low = budget * (1 - tolerance)
    high = budget * (1 + tolerance)
    matches = training_data[
        (training_data['SalePrice'] >= low) &
        (training_data['SalePrice'] <= high)
    ]

    if matches.empty:
        return {"houses": [], "count": 0, "message": "No houses found in this price range. Try a wider budget."}

    # Return aggregated stats of matching houses
    stats = {
        "count": len(matches),
        "price_range": {"min": int(matches['SalePrice'].min()), "max": int(matches['SalePrice'].max())},
        "avg_overall_quality": round(matches['OverallQual'].mean(), 1),
        "avg_living_area": int(matches['GrLivArea'].mean()),
        "avg_year_built": int(matches['YearBuilt'].mean()),
        "avg_bedrooms": round(matches['BedroomAbvGr'].mean(), 1) if 'BedroomAbvGr' in matches else "N/A",
        "avg_bathrooms": round(matches['FullBath'].mean(), 1) if 'FullBath' in matches else "N/A",
        "avg_garage_cars": round(matches['GarageCars'].mean(), 1) if 'GarageCars' in matches else "N/A",
        "top_neighborhoods": matches['Neighborhood'].value_counts().head(5).to_dict(),
        "common_house_styles": matches['HouseStyle'].value_counts().head(3).to_dict(),
        "central_air_pct": round(100 * (matches['CentralAir'] == 'Y').mean()),
    }
    return {"stats": stats}


@app.get("/stats")
def get_dataset_stats():
    """Return general stats about the training dataset."""
    return {
        "total_houses": len(training_data),
        "price_min": int(training_data['SalePrice'].min()),
        "price_max": int(training_data['SalePrice'].max()),
        "price_avg": int(training_data['SalePrice'].mean()),
        "price_median": int(training_data['SalePrice'].median()),
    }
