import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

train_all = pd.read_csv('/home/meirkhan/Desktop/train.csv')

print "train_data: ", train_all.shape

X_test = pd.read_csv("/home/meirkhan/Desktop/test.csv")
print "test_data: ", X_test.shape

method = RandomForestRegressor(n_jobs=-1,max_features= 'sqrt' ,n_estimators=700, oob_score = True)


# Target 1
train_data1 = train_all.drop(['id', 'bandgap_energy_ev'], 1)
X_train1 = train_data1.ix[:, train_data1.columns != 'formation_energy_ev_natom']
y_train1 = train_data1.ix[:, train_data1.columns == 'formation_energy_ev_natom']
method.fit(X_train1, y_train1.values.ravel())
prediction1 = method.predict(X_test.drop(['id'], 1))

#
# # Build a classification task using 3 informative features
# X, y = make_regression(n_samples=1000,
#                            n_features=10,
#                            n_informative=3,
#                            random_state=0,
#                            shuffle=False)
#
#
# rfc = RandomForestRegressor(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
#
# param_grid = {
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
#
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train1, y_train1)
# print CV_rfc.best_params_


# Target 2
train_data2 = train_all.drop(['id', 'formation_energy_ev_natom'], 1)
X_train2 = train_data2.ix[:, train_data2.columns != 'bandgap_energy_ev']
y_train2 = train_data2.ix[:, train_data2.columns == 'bandgap_energy_ev']
method.fit(X_train2, y_train2.values.ravel())
prediction2 = method.predict(X_test.drop(['id'], 1))

print "Trained"

target1 = (pd.DataFrame({'formation_energy_ev_natom': prediction1}))
result1 = pd.concat([X_test['id'], target1], axis = 1)

target2 = (pd.DataFrame({'bandgap_energy_ev': prediction2}))
# result2 = pd.concat([X_test['id'], target2], axis = 1)

final_result = pd.concat([result1,target2], axis = 1)

final_result.to_csv("/home/meirkhan/Desktop/final_result.csv",index=False)

print("finished")
