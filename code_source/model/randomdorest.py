import os
import pickle

from sklearn.ensemble import RandomForestClassifier

from code_source.dataset import Dataset

script_dir = os.path.dirname(__file__)

rel_path = '../../data/random_forest.model'
abs_file_path = os.path.join(script_dir, rel_path)

random_forest = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=500, n_jobs=-1,
                                       random_state=0)
random_forest.fit(Dataset.X_train, Dataset.y_train)
pickle.dump(random_forest, open(abs_file_path, 'wb'))
