from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import RootModel
import pandas as pd
import io
import csv
from io import StringIO
from .predict import predict_and_find_similar
from .model_utils import load_models

app = FastAPI()


class ExoplanetInput(RootModel[dict]):
    pass

@app.post('/predict')
async def predict(data: ExoplanetInput):
    user_input = data.root
    # ...rest of your code...

    # Lazy-load model artifacts to get feature list (and surface clear errors)
    try:
        models = load_models()
        features = models['features']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate input keys match features
    missing = set(features) - set(user_input.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    result = predict_and_find_similar(user_input)
    return result

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Decode bytes to text for more robust CSV handling
    if file.filename.endswith('.csv'):
        # try utf-8 then latin1
        text = None
        for enc in ('utf-8', 'latin1'):
            try:
                text = contents.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded CSV file")

        # Try to sniff delimiter on a sample then parse; fallback to python engine
        try:
            sample = text[:8192]
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except Exception:
                sep = ','
            df = pd.read_csv(StringIO(text), sep=sep)
        except Exception as e:
            # fallback: try python engine with automatic separator detection
            try:
                df = pd.read_csv(StringIO(text), sep=None, engine='python')
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}; fallback error: {e2}")
    elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Lazy-load features for validation
    try:
        models = load_models()
        features = models['features']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate columns
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    
    results = []
    for _, row in df.iterrows():
        user_input = row[features].to_dict()
        pred = predict_and_find_similar(user_input)
        results.append(pred)
    return {"results": results}
