import logging
import numpy as np
from sklearn.impute import *


def preprocess_null(df):
    # imputer = MissingIndicator()
    imputer = SimpleImputer(strategy='constant', fill_value=-10)
    df = imputer.fit_transform(df)


def merge_comp(df):
    comp_rates = ['comp'+str(i)+'_rate' for i in range(1, 9)]
    comp_invs = ['comp'+str(i)+'_inv' for i in range(1, 9)]
    comp_rate_percent_diffs = ['comp'+str(i)+'_rate_percent_diff' for i in range(1, 9)]

    df['comp_avg_rate'] = df[comp_rates].mean(axis=1)
    df['comp_avg_inv'] = df[comp_invs].mean(axis=1)
    df['comp_avg_rate_percent_diff'] = df[comp_rate_percent_diffs].mean(axis=1)

    comps = comp_rates + comp_invs + comp_rate_percent_diffs
    df.drop(comps, axis=1, inplace=True)
