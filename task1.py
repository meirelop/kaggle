import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



def read_file():
    train_path = '/home/meirkhan/Desktop/task1_train.csv'
    train_data = pd.read_csv(train_path)
    train_data.fillna(0, inplace=True)
    train_data = train_data.drop(['VIN_1', 'VIN_2'], 1)
    df_train_data = pd.DataFrame(data=train_data)
    train_data.columns = train_data.columns.str.strip()

    test_path = '/home/meirkhan/Desktop/task1_test.csv'
    X_test = pd.read_csv(test_path)
    X_test.fillna(0, inplace=True)
    X_test = X_test.drop(['VIN_1', 'VIN_2'], 1)
    X_test = pd.DataFrame(data=X_test)
    X_test.columns = X_test.columns.str.strip()
    # isnull = df.isnull().any().any()
    return df_train_data, X_test


def split_data(df):
    X_train = df.ix[:, df.columns != 'ESTIM_COST']
    y_train = df.ix[:, df.columns == 'ESTIM_COST']


    print 'X_train size:', X_train.shape
    print 'y_train size:', y_train.shape

    return X_train, y_train


def get_discrete(df, feature_name):
    key_words = feature_name.split('_')
    prefix = '_'.join([x[0:3] for x in key_words])
    temp_features = pd.get_dummies(df[feature_name], prefix,)
    df = pd.concat([df, temp_features], axis=1)
    df = df.drop(feature_name, 1)
    return df


def get_prediction():
    method = RandomForestRegressor()
    method.fit(X_train, y_train)
    prediction = method.predict(X_test)

    predicted = (pd.DataFrame({'ESTIM_COST': prediction}))
    predicted_num = pd.concat([X_test['ID'], predicted], axis=1)

    result_file = '/home/meirkhan/Desktop/task1_result.xlsx'
    writer = pd.ExcelWriter(result_file)
    predicted_num.to_excel(writer, index=False, encoding='utf-8')
    writer.save()


if __name__ == '__main__':
    df_train_data, df_x_test = read_file()
    categorical_feature = ['VIN_1', 'VIN_2', 'VIN_3', 'FUEL_TYPE', 'BODY_TYPE', 'TYPE_OF_DRIVE',
                           'INTERIOR_TYPE', 'TRANSM_TYPE', 'AUTO_CONDITION']
    for i in categorical_feature:
        df_train_data = get_discrete(df_train_data, i)
    for j in categorical_feature:
        X_test = get_discrete(df_x_test, j)

    X_train, y_train = split_data(df_train_data)
    get_prediction()



