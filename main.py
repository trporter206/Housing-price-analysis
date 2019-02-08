import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

data = pd.read_csv('train.csv')
y = data.SalePrice

path = r'C:\Udacity\Data\project'
all_files = glob.glob(path+"/*.csv")
train_data = pd.concat((pd.read_csv(f) for f in all_files))

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

X = train_data.drop(['SalePrice','Id','LotFrontage','YearRemodAdd','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtHalfBath','FullBath','HalfBath','GarageYrBlt','GarageArea','OpenPorchSF','EnclosedPorch','MiscVal','MoSold','YrSold'], axis=1).select_dtypes(exclude=['object'])
train_x,test_x,train_y,test_y = train_test_split(X.as_matrix(), y.as_matrix(),test_size=.25)

#train--------------------------------------------------------------------------
imputer = Imputer()
train_x = imputer.fit_transform(train_x)
test_x = imputer.transform(test_x)
model = XGBRegressor()
model.fit(train_x, train_y, verbose=False)
pipeline=make_pipeline(Imputer(), model)
scores = cross_val_score(pipeline, train_x, train_y, scoring='neg_mean_absolute_error')
print('MAE %2f' %(-1*scores.mean()))

# partial_dependence------------------------------------------------------------
# model = GradientBoostingRegressor()
# model.fit(train_x, train_y)
# plot = plot_partial_dependence(model,features=[0,1],X=test_x,feature_names=['LotArea','OverallQual'],grid_resolution=10)

# ------------------------------------------------------------------------------
# print(mean_absolute_error(predictions, test_y))
plt.show()
