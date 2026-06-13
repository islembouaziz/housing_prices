import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

house_data_testing = pd.read_csv('data/test.csv')
house_data_training = pd.read_csv('data/train.csv')
features =['MSSubClass', 'LotFrontage', 'LotShape', 'Neighborhood', 'HouseStyle','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','Foundation','BsmtQual','BsmtUnfSF','GrLivArea','BsmtFullBath','FullBath','HalfBath','TotRmsAbvGrd','MoSold','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageCars','GarageArea','KitchenQual','GarageFinish','Fireplaces']
X = house_data_training[features]
y = house_data_training.SalePrice
testing_X = house_data_testing[features]
X = pd.get_dummies(X)
testing_X = pd.get_dummies(testing_X)
X, testing_X = X.align(
    testing_X,
    join='left',
    axis=1,
    fill_value=0
)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_X, train_y)
preds_val = model.predict(val_X)
preds_test = model.predict(testing_X)
print(mean_absolute_error(val_y, preds_val))
#mae: 17674.39112328767
output = pd.DataFrame({'Id': house_data_testing.Id, 'SalePrice': preds_test})
output.to_csv('submissionRandomForestRegressor.csv', index=False)
