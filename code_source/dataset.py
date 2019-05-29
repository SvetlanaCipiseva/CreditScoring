import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from code_source.features.numberloans import number_of_previous_loans
from sklearn.preprocessing import MinMaxScaler


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

    @staticmethod
    def clean_age(df):
        df = df[(df.Age >= 18) & (df.Age <= 80)]
        return df

    @staticmethod
    def clean_customer_type(df):
        df = df[df.CustomerType.isin([1, 2])]
        return df

    @staticmethod
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

    @staticmethod
    def clean_dataset(df):
        df = df.copy()
        df = df.dropna()
        df.DisbursementDate = pd.to_datetime(df.DisbursementDate)
        df = Dataset.clean_age(df)
        df = Dataset.clean_customer_type(df)
        df['AgeBin'] = Dataset.binning(df.Age)
        df['CRBScoreBin'] = Dataset.binning(df.CRBScore)
        df['MonthsSinceOpenBin'] = df.MonthsSinceOpen // 3 * 3
        df['FinancialMeasure1Bin'] = Dataset.binning(df.FinancialMeasure1)
        df['FinancialMeasure2Bin'] = Dataset.binning(df.FinancialMeasure2)
        df['FinancialMeasure4Bin'] = Dataset.binning(df.FinancialMeasure4)
        df = df.drop(columns='MonthsSinceActive')
        df['DisbursementDay'] = df.DisbursementDate.dt.day
        df = number_of_previous_loans(df)
        df = df[~df.FinancialMeasure2Bin.isna()]
        return df

    @staticmethod
    def bining(feature):
        q = np.linspace(0, 1, 21)
        np.quantile(feature, q)


Dataset.df = Dataset.get_data()
Dataset.clean_df = Dataset.clean_dataset(Dataset.df)

Dataset.X = Dataset.clean_df[['Age', 'CustomerType', 'SOR',
                              'MonthsSinceOpen',
                              'FinancialMeasure1',
                              'FinancialMeasure2',
                              'FinancialMeasure4', 'CRBScore',
                              'PreviousLoans',
                              'DisbursementDay', 'Amount']]
Dataset.y = Dataset.clean_df['Default']

X_train, X_test, y_train, y_test = train_test_split(Dataset.X,
                                                    Dataset.y, test_size=0.3,
                                                    random_state=0, stratify=Dataset.clean_df['Default'])
X_train = X_train.drop(columns='Amount')
X_test_amount = X_test.Amount
X_test = X_test.drop(columns='Amount')

Dataset.X_train = X_train
Dataset.X_test = X_test
Dataset.y_train = y_train
Dataset.y_test = y_test
Dataset.X_test_amount = X_test_amount

scaler = MinMaxScaler()
scaler.fit(Dataset.X_train)

Dataset.X_train_scaled = scaler.transform(Dataset.X_train)
Dataset.X_test_scaled = scaler.transform(Dataset.X_test)


def absolute_path(rel_path):
    rel_path = '../data/' + rel_path
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)
