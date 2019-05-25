import pandas as pd
import numpy as np
import os


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
    df=df[df.CustomerType.isin([1, 2])]
    return df

def binning(feature):
    q = np.linspace(0, 1, 21)
    breaks = np.quantile(feature, q).astype('int')
    breaks[-1] = breaks[-1]+1
    new_feature = np.digitize(feature,breaks)
    mapping = {i+1 : f'[{breaks[i]} - {breaks[i+1]})' for i in range(len(breaks)-1)}
    return np.vectorize(mapping.get)(new_feature)

def clean_dataset(df):
    df = df.copy()
    df = df.dropna()
    df.DisbursementDate = pd.to_datetime(df.DisbursementDate)
    df=clean_age(df)
    df=clean_customer_type(df)
    df['AgeBin'] = binning(df.Age)
    df['CRBScoreBin'] = binning(df.CRBScore)
    df = df.drop(columns='MonthsSinceActive')
    return df

def bining(feature):
    q = np.linspace(0, 1, 21)
    np.quantile(feature, q)


