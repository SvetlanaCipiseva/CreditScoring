import scorecardpy as sc
from code_source.dataset import Dataset


class WOE:
    df_sample = Dataset.X_train[['Age', 'MonthsSinceOpen', 'FinancialMeasure1', 'FinancialMeasure2',
                                 'FinancialMeasure4', 'CRBScore', 'PreviousLoans', 'DisbursementDay']].copy()
    df_sample['Default'] = Dataset.y_train

    optimal_bin = sc.woebin(df_sample, 'Default')

    x_train_woe = sc.woebin_ply(Dataset.X_train, optimal_bin)
    x_test_woe = sc.woebin_ply(Dataset.X_test, optimal_bin)
