import os
import pickle
from sklearn.preprocessing import StandardScaler
from code_source.dataset import Dataset, absolute_path

script_dir = os.path.dirname(__file__)

rel_path_y_gb = '../../data/y_predict_proba_gb.array'
abs_file_path_y_gb = os.path.join(script_dir, rel_path_y_gb)

rel_path_y_rf = '../../data/y_predict_proba_rf.array'
abs_file_path_y_rf = os.path.join(script_dir, rel_path_y_rf)

rel_path_y_svm_linear = '../../data/y_predict_proba_svm_linear.array'
abs_file_path_y_svm_linear = os.path.join(script_dir, rel_path_y_svm_linear)

rel_path_y_svm_rbf = '../../data/y_predict_proba_svm_rbf.array'
abs_file_path_y_svm_rbf = os.path.join(script_dir, rel_path_y_svm_rbf)

rel_path_y_svm_poly = '../../data/y_predict_proba_svm_poly.array'
abs_file_path_y_svm_poly = os.path.join(script_dir, rel_path_y_svm_poly)

# Gradient Boosting
gradient_boosting = pickle.load(open(absolute_path('gradient_boosting.model'), 'rb'))
y_predict_gb = gradient_boosting.predict_proba(Dataset.X_test)[:, 1]
pickle.dump(y_predict_gb, open(abs_file_path_y_gb, 'wb'))

# Random forest
random_forest = pickle.load(open(absolute_path('random_forest.model'), 'rb'))
y_predict_rf = random_forest.predict_proba(Dataset.X_test)[:, 1]
pickle.dump(y_predict_rf, open(abs_file_path_y_rf, 'wb'))

# SVM
X_train = Dataset.X_train.astype('float64')
X_test = Dataset.X_test.astype('float64')
scaler = StandardScaler()
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
# SVM Linear
svm_linear = pickle.load(open(absolute_path('svm_linear.model'), 'rb'))
y_predict_smv_linear = svm_linear.predict_proba(X_test_scaled)[:, 1]
pickle.dump(y_predict_smv_linear, open(abs_file_path_y_svm_linear, 'wb'))
# SVM RBF
svm_rbf = pickle.load(open(absolute_path('svm_rbf.model'), 'rb'))
y_predict_smv_rbf = svm_rbf.predict_proba(X_test_scaled)[:, 1]
pickle.dump(y_predict_smv_rbf, open(abs_file_path_y_svm_rbf, 'wb'))
# SVM polynomial
svm_poly = pickle.load(open(absolute_path('svm_poly.model'), 'rb'))
y_predict_smv_poly = svm_poly.predict_proba(X_test_scaled)[:, 1]
pickle.dump(y_predict_smv_poly, open(abs_file_path_y_svm_poly, 'wb'))

from code_source.dataset import Dataset, absolute_path
a = pickle.load(open(absolute_path('gradient_boosting.model'), 'rb'))
