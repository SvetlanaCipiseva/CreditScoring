import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class Dataset:
    df = pd.DataFrame()

    @staticmethod
    def get_data():
        if not Dataset.df.empty:
            return Dataset.df
        else:
            script_dir = os.path.dirname(__file__)
            rel_path = '../data/loans.csv'
            abs_file_path = os.path.join(script_dir, rel_path)
            Dataset.df = pd.read_csv(abs_file_path)
            return Dataset.df


def clean_age(df):
    df = df[(df.Age >= 18) & (df.Age <= 80)]
    return df


def clean_customer_type(df):
    df = df[df.CustomerType.isin([1, 2])]
    return df


def binning(feature):
    q = np.linspace(0, 1, 21)
    breaks = np.quantile(feature, q).astype('int')
    breaks[-1] = breaks[-1] + 1
    breaks = list(dict.fromkeys(breaks))
    new_feature = np.digitize(feature, breaks)
    categories = [f'[{breaks[i]} - {breaks[i + 1]})' for i in range(len(breaks) - 1)]
    mapping = {i + 1: categories[i] for i in range(len(categories))}
    series = pd.Categorical(np.vectorize(mapping.get)(new_feature), categories=categories, ordered=True)
    return series


def clean_dataset(df):
    df = df.copy()
    df = df.dropna()
    df.DisbursementDate = pd.to_datetime(df.DisbursementDate)
    df = clean_age(df)
    df = clean_customer_type(df)
    df['AgeBin'] = binning(df.Age)
    df['CRBScoreBin'] = binning(df.CRBScore)
    df['MonthsSinceOpenBin'] = df.MonthsSinceOpen // 3 * 3
    df['FinancialMeasure1Bin'] = binning(df.FinancialMeasure1)
    df['FinancialMeasure2Bin'] = binning(df.FinancialMeasure2)
    df['FinancialMeasure4Bin'] = binning(df.FinancialMeasure4)
    df = df.drop(columns='MonthsSinceActive')
    return df


def bining(feature):
    q = np.linspace(0, 1, 21)
    np.quantile(feature, q)


def splitDataset(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
