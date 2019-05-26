import pandas as pd

def number_of_previous_loans(df):
    tmp = df.groupby(['Customer_WID', 'DisbursementDate']).Customer_WID.count().reset_index(name='LoanCount')
    tmp = tmp.sort_values(['Customer_WID', 'DisbursementDate'])
    tmp['PreviousLoans'] = tmp.groupby(['Customer_WID']).LoanCount.cumsum(axis = 0) - 1
    del tmp['LoanCount']
    if 'PreviousLoans' in df.columns:
        del df['PreviousLoans']
    df = pd.merge(df, tmp, how='left', on=['Customer_WID', 'DisbursementDate'])
    return df