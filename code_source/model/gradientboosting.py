from sklearn.ensemble import GradientBoostingClassifier
import pickle
from code_source.dataset import Dataset
import os

script_dir = os.path.dirname(__file__)

rel_path_linear = '../../data/gradient_boosting.model'
abs_file_path = os.path.join(script_dir, rel_path_linear)

gradient_boosting = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200,
                                               subsample=0.5, max_depth=5)
gradient_boosting.fit(Dataset.X_train, Dataset.y_train)
pickle.dump(gradient_boosting, open(abs_file_path, 'wb'))
