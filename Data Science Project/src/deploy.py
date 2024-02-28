import uvicorn
import joblib
import pickle
from fastapi import FastAPI
from wine import Wine

app = FastAPI()
model = joblib.load(open('../models/gradient_boosting/model.joblib', 'rb'))
with open('../models/gradient_boosting/features_scaler.pkl', 'rb') as f:
    features_scaler = pickle.load(f)
with open('../models/gradient_boosting/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

@app.get("/")
def index():
    return {"message": "Wine quality predictor API"}

@app.post("/wine/predict")
def predict_wine_quality(data: Wine):

    data = data.dict()
    fixed_acidity = data.get('fixed_acidity')
    volatile_acidity = data.get('volatile_acidity')
    citric_acid = data.get('citric_acid')
    residual_sugar = data.get('residual_sugar')
    chlorides = data.get('chlorides')
    free_sulfur_dioxide = data.get('free_sulfur_dioxide')
    total_sulfur_dioxide = data.get('total_sulfur_dioxide')
    density = data.get('density')
    pH = data.get('pH')
    suplhates = data.get('sulphates')
    alcohol = data.get('alcohol')
    type = data.get('type')
    
    ### scale inputs
    x_scaler = features_scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                           free_sulfur_dioxide, total_sulfur_dioxide, density, pH, suplhates, alcohol, type]])
    
    prediction = target_scaler.inverse_transform([model.predict(x_scaler)])
    print(prediction)
    
    return {
        'prediction' : prediction[0][0]
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)