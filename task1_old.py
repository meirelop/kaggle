import codecs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import copy

train_data = pd.read_csv('/home/meirkhan/Desktop/task1_train.csv')
train_data.fillna(0, inplace=True)
print "train_data: ", train_data.shape

X_test = pd.read_csv("/home/meirkhan/Desktop/task1_test.csv")
print "X_test: ", X_test.shape

X_train = train_data.ix[:, train_data.columns != 'ESTIM_COST']
y_train = train_data.ix[:, train_data.columns == 'ESTIM_COST']

dataset = pd.concat(objs=[X_train, X_test], axis=0)
method = RandomForestRegressor()

print "X_train: ", X_train.shape
print "y_train: ", y_train.shape

def fetch_feature(df, feature_name):
    key_words = feature_name.split('_')
    prefix = '_'.join([x[0:3] for x in key_words])
    temp_features = pd.get_dummies(df[feature_name], prefix,)
    df = pd.concat([df, temp_features], axis=1)
    df = df.drop(feature_name, 1)
    return df

categorical_features = ['VIN_1', 'VIN_2', 'VIN_3', 'VIN_15', 'VIN_16', 'VIN_17', 'FUEL_TYPE', 'TYPE_OF_DRIVE',
                        'INTERIOR_TYPE', 'TRANSM_TYPE', 'AUTO_CONDITION', 'BODY_TYPE']

for i in categorical_features:
    # X_train = fetch_feature(X_train, i)
    # X_test = fetch_feature(X_test, i)
    dataset = fetch_feature(dataset, i)
print 'dataset:', dataset.shape

X_train = copy.copy(dataset[:6000])
X_test = copy.copy(dataset[6000:])


print 'converted into numeric values'
print X_train.shape
print X_test.shape

print "Training..."

method = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
method.fit(X_train, y_train.values.ravel())
prediction = method.predict(X_test)

print "Trained"

train_target = (pd.DataFrame({'ESTIM_COST': prediction}))
train = pd.concat([X_test['ID'], train_target], axis=1)


writer = pd.ExcelWriter("/home/meirkhan/Desktop/task1_result_exp.xlsx")
train.to_excel(writer, encoding='utf-8', index=False)
writer.save()