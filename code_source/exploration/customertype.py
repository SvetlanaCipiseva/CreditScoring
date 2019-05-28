from code_source.dataset import Dataset
import pandas as pd


class CustomerTypeContainer:
    df = Dataset.clean_df

    amount_min_customer_1 = df[df.CustomerType == 1.0].Amount.min()
    amount_max_customer_1 = df[df.CustomerType == 1.0].Amount.max()
    amount_median_customer_1 = df[df.CustomerType == 1.0].Amount.median()
    amount_min_customer_2 = df[df.CustomerType == 2.0].Amount.min()
    amount_max_customer_2 = df[df.CustomerType == 2.0].Amount.max()
    amount_median_customer_2 = df[df.CustomerType == 2.0].Amount.median()
