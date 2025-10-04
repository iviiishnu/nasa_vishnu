from app.model_utils import load_models

try:
    models = load_models()
    features = models.get('features')
    print('Loaded models OK')
    print('Number of features:', len(features))
    print('First 10 features:', features[:10])
except Exception as e:
    import traceback
    print('Error loading models:')
    traceback.print_exc()
