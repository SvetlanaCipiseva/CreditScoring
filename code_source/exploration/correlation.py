from code_source.dataset import Dataset
import pandas as pd


class CorrelationContainer:
    df = Dataset.df

    corr_FM1_FM2 = round(df[['FinancialMeasure3', 'FinancialMeasure4']].corr().iloc[1, 0], 3)
    corr_CRBScore_Default = round(df[['CRBScore', 'Default']].corr().iloc[1, 0], 3)
    corr_MonthsSinceOpen_Age = round(df[['MonthsSinceOpen', 'Age']].corr().iloc[1, 0], 3)
    corr_SOR_Amount = round(df[['SOR', 'Amount']].corr().iloc[1, 0], 3)
