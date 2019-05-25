from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np

interest_rate = 0.13
lace = 0.05

def calculate_revenue_curve(test_loans, prediction):
    fpr, tpr, thresholds = roc_curve(test_loans.Default, prediction)

    def row_return(row):
        if row.Default == 0:
            return row.Amount * (interest_rate / 12 + lace)
        else:
            return -row.Amount

    def calculate_revenue(threshold):
        selected_loans = test_loans[prediction <= threshold]
        loan_count = selected_loans.shape[0]
        amount = selected_loans.Amount.sum()
        #revenue = sum(selected_loans.apply(row_return, axis = 1))
        default_rate = selected_loans.Default.mean()
        revenue = amount * (1-default_rate)*(interest_rate / 12 + lace) - amount * default_rate

        return pd.DataFrame({
            "Threshold": threshold,
            "LoanCount": loan_count,
            "Amount": amount,
            "Revenue": revenue,
            "DefaultRate": default_rate
        },index=[0])

    if len(thresholds) > 200:
        sampled_indexes = np.linspace(0, len(thresholds)-1, num=200, dtype=np.int)
        thresholds = thresholds[sampled_indexes]

    df_list = [calculate_revenue(threshold) for threshold in thresholds]
    return pd.concat(df_list)
