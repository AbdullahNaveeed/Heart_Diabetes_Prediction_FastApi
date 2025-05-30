from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models and scalers
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("diabetes_scaler.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_heart")
async def predict_heart(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = heart_scaler.transform(input_data)
    prediction = heart_model.predict(input_data_scaled)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    return templates.TemplateResponse("index.html", {"request": request, "heart_result": result})

@app.post("/predict_diabetes")
async def predict_diabetes(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_data_scaled = diabetes_scaler.transform(input_data)
    prediction = diabetes_model.predict(input_data_scaled)
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
    return templates.TemplateResponse("index.html", {"request": request, "diabetes_result": result})
