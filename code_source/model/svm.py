from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from code_source.dataset import Dataset
import os

script_dir = os.path.dirname(__file__)

rel_path_linear = '../../data/svm_linear.model'
abs_file_path_linear = os.path.join(script_dir, rel_path_linear)

rel_path_rbf = '../../data/svm_rbf.model'
abs_file_path_rbf = os.path.join(script_dir, rel_path_rbf)

rel_path_poly = '../../data/svm_poly.model'
abs_file_path_poly = os.path.join(script_dir, rel_path_poly)

X_train = Dataset.X_train.astype('float64')
X_test = Dataset.X_test.astype('float64')
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC with linear kernel
svm_linear = SVC(kernel='linear', random_state=0, probability=True)
svm_linear.fit(X_train_scaled, Dataset.y_train)
pickle.dump(svm_linear, open(abs_file_path_linear, 'wb'))

# SVC with Radial basis function (RBF) kernel
svm_rbf = SVC(kernel='rbf', gamma=1, random_state=0, probability=True)
svm_rbf.fit(X_train_scaled, Dataset.y_train)
pickle.dump(svm_rbf, open(abs_file_path_rbf, 'wb'))

# SVC with polynomial (degree 2) kernel
svm_poly=SVC(kernel='poly', degree=2, random_state=0, probability=True)
svm_poly.fit(X_train_scaled, Dataset.y_train)
pickle.dump(svm_poly, open(abs_file_path_poly, 'wb'))