import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
def get_mae_for_decision_tree(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
house_data_testing = pd.read_csv('data/test.csv')
house_data_training = pd.read_csv('data/train.csv')
#features =['MSSubClass', 'LotFrontage', 'LotShape', 'Neighborhood', 'HouseStyle','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd','Foundation','BsmtQual','BsmtUnfSF','GrLivArea','BsmtFullBath','FullBath','HalfBath','TotRmsAbvGrd','MoSold']
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
candidate_max_leaf_nodes = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50] 
scores = {leaf_size: get_mae_for_decision_tree(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
print(scores)

best_tree_size = min(scores, key=scores.get)
print("Best tree size: %d" % best_tree_size)
#42: 23075.4963

final_model = DecisionTreeRegressor(max_leaf_nodes=34, random_state=0)
final_model.fit(X, y)
preds_test = final_model.predict(testing_X)
output = pd.DataFrame({'Id': house_data_testing.Id, 'SalePrice': preds_test})
output.to_csv('submissionDecisionTreeRegressor.csv', index=False)



