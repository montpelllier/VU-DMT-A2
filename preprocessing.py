import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import *
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models.MLP import MLP


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
    df_non_missing = df_non_missing[feature_columns + [target_column]].drop_duplicates()
    # df_non_missing = df_non_missing[feature_columns+[target_column]].dropna(how='any')

    # 准备特征矩阵和目标向量
    X_train = df_non_missing[feature_columns]
    y_train = df_non_missing[target_column]
    X_test = df_missing[feature_columns]

    # 训练线性回归模型
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print('Score:', scores)
    print('Mean MAE score', np.mean(scores))

    # 使用训练好的模型填充缺失值
    model.fit(X_train, y_train)
    df.loc[df[target_column].isnull(), target_column] = model.predict(X_test)

    return df


def fill_by_classifier(df, feature_columns, target_column):
    # 拆分数据集，包括缺失值和非缺失值
    df_missing = df[df[target_column].isnull()]
    df_non_missing = df.dropna(subset=[target_column])
    df_non_missing = df_non_missing[feature_columns + [target_column]].drop_duplicates()

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


def fill_by_nn(df, feature_columns, target_column, epoch_num=100):
    df_missing = df[df[target_column].isnull()]
    df_non_missing = df.dropna(subset=[target_column])
    df_non_missing = df_non_missing[feature_columns + [target_column]].drop_duplicates()

    X_train = df_non_missing[feature_columns]
    y_train = df_non_missing[target_column]
    X_test = df_missing[feature_columns]

    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(len(dataloader))

    learning_rate = 0.01
    model = MLP(len(feature_columns), hidden_size=64, output_size=1)
    # criterion = nn.MSELoss()  # 均方误差损失函数
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []  # 保存损失值
    for epoch in range(epoch_num):
        epoch_loss = 0.0  # 保存每个epoch的损失
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)  # 更新当前epoch的损失

        epoch_loss /= len(dataset)  # 计算每个样本的平均损失
        losses.append(epoch_loss)  # 将当前epoch的损失保存到列表中
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {epoch_loss:.4f}')

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    # 在测试集上测试模型
    with torch.no_grad():
        predicted = model(X_test).detach().numpy()
        df.loc[df[target_column].isnull(), target_column] = predicted


def predict_score(df, feature_columns, target_column, model_name):
    X_train = df[feature_columns]
    y_train = df[target_column].copy()
    y_train *= 3
    y_train **= 2
    # if model_name == 'fm':
    #     model = pyfms.Regressor(len(feature_columns), k=2)
    #     reg = pyfms.regularizers.L2(0, 0, .01)
    #     model.fit(X_train, y_train, nb_epoch=50000, verbosity=5000, regularizer=reg)
    # elif model_name == 'lr':
    if model_name == 'lr':
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:
        raise ValueError('Invalid model type')

    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    print(f'{model} MSE: {mse}, MAE: {mae}.')
    return model


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
    start_date = df.date_time.min()
    df['days'] = (df.date_time - start_date).dt.days
