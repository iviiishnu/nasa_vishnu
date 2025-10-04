import numpy as np
import pandas as pd
from app.model_utils import load_models

def predict_and_find_similar(user_input, k=3):
    # Load models/artifacts lazily to avoid import-time binary issues
    models = load_models()
    clf = models['clf']
    imputer = models['imputer']
    features = models['features']
    full_df = models['full_df']
    kdtree = models['kdtree']

    input_df = pd.DataFrame([user_input], columns=features)
    input_imputed = imputer.transform(input_df)
    prediction_numeric = clf.predict(input_imputed)[0]
    prediction_prob = clf.predict_proba(input_imputed)[0]
    target_map_rev = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
    prediction_label = target_map_rev.get(prediction_numeric, 'Unknown')
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    key_factors = {features[i]: round(importances[i], 4) for i in top_indices}
    dist, ind = kdtree.query(input_imputed, k=k)
    similar_exoplanets = full_df.iloc[ind[0]][['kepoi_name', 'kepler_name', 'koi_disposition']]
    similar_exoplanets = similar_exoplanets.reset_index(drop=True)
    return {
        'prediction': prediction_label,
        'confidence': round(prediction_prob[prediction_numeric], 4),
        'key_factors': key_factors,
        'similar_exoplanets': similar_exoplanets.to_dict(orient='records')
    }
