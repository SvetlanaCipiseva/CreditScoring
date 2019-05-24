class DataCleanig:

    @staticmethod
    def remove_nans(df):
        df=df.dropna()
        return df

