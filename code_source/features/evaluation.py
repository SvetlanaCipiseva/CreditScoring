import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

interest_rate = 0.13
lace = 0.05


def calculate_revenue(cutoff, test_dataset, y_predict, y_test, amounts):
    predict_paid_df = test_dataset[y_predict <= cutoff]  # predicted paid loans
    loan_count = predict_paid_df.shape[0]
    amount_sum = amounts[y_predict <= cutoff].sum()
    default_rate = y_test[y_predict <= cutoff].mean()
    revenue = amount_sum * (1 - default_rate) * (interest_rate / 12 + lace) - amount_sum * default_rate

    return pd.DataFrame({'Threshold': cutoff,
                         'LoanCount': loan_count,
                         'Amount': amount_sum,
                         'Revenue': revenue,
                         'DefaultRate': default_rate}, 
                        index=[0])


def calculate_revenue_curve(test_dataset, y_predict, y_test, amounts):
    fpr, tpr, cutoffs = roc_curve(y_test, y_predict)

    if len(cutoffs) > 200:
        sampled_indexes = np.linspace(0, len(cutoffs) - 1, num=200, dtype=np.int)
        cutoffs = cutoffs[sampled_indexes]

    df_list = [calculate_revenue(cutoff, test_dataset, y_predict, y_test, amounts) for cutoff in cutoffs]
    return pd.concat(df_list).reset_index()
