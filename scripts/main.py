from fastapi import FastAPI, HTTPException
import joblib
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/")
def predict(tv: float, radio: float, newspaper: float):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/sales_model.pkl'))
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    model = joblib.load(model_path)
    prediction = model.predict([[tv, radio, newspaper]])
    return {"prediction": prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
