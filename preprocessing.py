import numpy as np
import pandas as pd
from fancyimpute import BiScaler, SoftImpute
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.impute import *
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Ridge, Lasso
from sklearn.model_selection import cross_val_score


def preprocess_null_fill(df, columns=None, fill_value=-10):
    if columns is None:
        columns = df.columns.tolist()
    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
    df[columns] = imputer.fit_transform(df[columns])


def preprocess_null_fill_mean(df, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    df[columns] = imputer.fit_transform(df[columns])


def preprocess_null_mc(df, columns, bc_iter=100, mc_iter=100):
    sub_df = df[columns]
    X = sub_df.values.astype(float)
    missing_mask = np.isnan(X)
    print(X.shape)

    X_incomplete_normalized = BiScaler(max_iters=bc_iter).fit_transform(X)
    print("X_incomplete_normalization done")
    X_filled_soft_impute = SoftImpute(max_iters=mc_iter, min_value=0).fit_transform(X_incomplete_normalized)
    print(X_filled_soft_impute)

    # replace filled negative values to zeroes
    fill_values = X_filled_soft_impute[missing_mask]
    print(fill_values)
    # neg_indices = (fill_values < 0)
    # fill_values[neg_indices] = 0
    X[missing_mask] = fill_values
    print(X)

    sub_df.loc[:, :] = X
    df[columns] = sub_df


def fill_by_regressor(df, regressor, feature_columns, target_column):
    if regressor == 'lr':
        model = LinearRegression()
    elif regressor == 'ridge':
        model = Ridge()
    elif regressor == 'lasso':
        model = Lasso()
    elif regressor == 'sgd':
        model = SGDRegressor()
    else:
        raise ValueError('Invalid regressor type')

    # 拆分数据集，包括缺失值和非缺失值
    df_missing = df[df[target_column].isnull()]
    df_non_missing = df.dropna(subset=[target_column])

    # 准备特征矩阵和目标向量
    X_train = df_non_missing[feature_columns]
    y_train = df_non_missing[target_column]
    X_test = df_missing[feature_columns]

    # 训练线性回归模型
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    print('Score:', scores)
    print('Mean score', np.mean(scores))

    # 使用训练好的模型填充缺失值
    model.fit(X_train, y_train)
    df.loc[df[target_column].isnull(), target_column] = model.predict(X_test)

    return df


def fill_by_classifier(df, feature_columns, target_column):
    # 拆分数据集，包括缺失值和非缺失值
    df_missing = df[df[target_column].isnull()]
    df_non_missing = df.dropna(subset=[target_column])

    # 准备特征矩阵和目标向量
    X_train = df_non_missing[feature_columns]
    y_train = df_non_missing[target_column]
    X_test = df_missing[feature_columns]

    # 训练多类别逻辑回归模型
    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print('Score:', scores)
    print('Mean score', np.mean(scores))

    model.fit(X_train, y_train)
    # 使用训练好的模型填充缺失值
    df.loc[df[target_column].isnull(), target_column] = model.predict(X_test)

    return df


def merge_comp(df):
    comp_rates = ['comp' + str(i) + '_rate' for i in range(1, 9)]
    comp_invs = ['comp' + str(i) + '_inv' for i in range(1, 9)]
    comp_rate_percent_diffs = ['comp' + str(i) + '_rate_percent_diff' for i in range(1, 9)]

    df['comp_avg_rate'] = df[comp_rates].mean(axis=1)
    df['comp_avg_inv'] = df[comp_invs].mean(axis=1)
    df['comp_avg_rate_percent_diff'] = df[comp_rate_percent_diffs].mean(axis=1)

    comps = comp_rates + comp_invs + comp_rate_percent_diffs
    df.drop(comps, axis=1, inplace=True)


def parse_date(df):
    df.date_time = pd.to_datetime(df.date_time)
    df['year'] = df.date_time.dt.year
    df['month'] = df.date_time.dt.month
