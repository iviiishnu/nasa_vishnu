import joblib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Dict, Any


def load_models() -> Dict[str, Any]:
	"""Lazy-load model artifacts and return them in a dict.

	This avoids importing heavy compiled extensions at module import time
	(which previously triggered binary incompatibility errors).
	"""
	try:
		clf = joblib.load('models/exoplanet_model.pkl')
		imputer = joblib.load('models/imputer.pkl')
		features = joblib.load('models/features_list.pkl')
		full_dataset_features = np.load('models/full_dataset_features.npy')
		full_df = pd.read_pickle('models/full_dataset.pkl')
		kdtree = KDTree(full_dataset_features)
	except Exception as e:
		# Re-raise with a clearer message about possible binary incompatibility
		raise RuntimeError(
			"Failed to load model artifacts. This is often caused by a binary "
			"incompatibility between NumPy and compiled Python extensions (e.g., scikit-learn). "
			"Try reinstalling compatible versions, for example: `pip install --force-reinstall "
			"numpy==1.25.2 scikit-learn==1.2.2` or recreate the virtualenv. Original error: "
			f"{e}"
		) from e

	return {
		'clf': clf,
		'imputer': imputer,
		'features': features,
		'full_dataset_features': full_dataset_features,
		'full_df': full_df,
		'kdtree': kdtree,
	}
