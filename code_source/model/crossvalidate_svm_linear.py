import os
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate
from code_source.dataset import Dataset

script_dir = os.path.dirname(__file__)

rel_path_load = '../../data/svm_linear.model'
abs_file_path_load = os.path.join(script_dir, rel_path_load)

rel_path_dump = '../../data/cross_validate_svm_linear.dataframe'
abs_file_path_dump = os.path.join(script_dir, rel_path_dump)

rel_path_dump_auc = '../../data/auc_svm_linear.dataframe'
abs_file_path_dump_auc = os.path.join(script_dir, rel_path_dump_auc)

svm_linear = pickle.load(open(abs_file_path_load, 'rb'))
print("Before cross_validate")
scores = cross_validate(svm_linear, Dataset.X_train_scaled, Dataset.y_train, scoring=['roc_auc'], cv=20, verbose=2, n_jobs=-1, return_train_score=True)
print("Right after cross_validate")
tmp_test = pd.DataFrame({'dataset': 'test', 'score': scores['test_roc_auc']})
auc_test_median_gb = tmp_test.score.median()
auc_test_std_gb = tmp_test.score.std()
tmp_train = pd.DataFrame({'dataset': 'train', 'score': scores['train_roc_auc']})
auc_train_median_gb = tmp_train.score.median()
auc_train_std_gb = tmp_train.score.std()
tmp = pd.concat([tmp_test, tmp_train])

auc_df_gb = pd.DataFrame({'Variables': ['auc_test_median_gb',
                                        'auc_test_std_gb',
                                        'auc_train_median_gb',
                                        'auc_train_std_gb'],
                          'Values': [auc_test_median_gb,
                                     auc_test_std_gb,
                                     auc_train_median_gb,
                                     auc_train_std_gb]
                          })

pickle.dump(tmp, open(abs_file_path_dump, 'wb'))
pickle.dump(auc_df_gb, open(abs_file_path_dump_auc, 'wb'))

