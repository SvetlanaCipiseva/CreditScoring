import pandas as pd
import os

class Dataset:
    df=pd.DataFrame()

    @staticmethod
    def get_data():
        if not Dataset.df.empty:
            return Dataset.df
        else:
            script_dir = os.path.dirname(__file__)
            rel_path = '../data/loans.csv'
            abs_file_path = os.path.join(script_dir, rel_path)
            Dataset.df = pd.read_csv(abs_file_path)
            Dataset.df.DisbursementDate = pd.to_datetime(Dataset.df.DisbursementDate)
            return Dataset.df