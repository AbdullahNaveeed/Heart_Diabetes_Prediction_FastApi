from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
heart_model = joblib.load("heart_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")

# Load scalers if used
try:
    heart_scaler = joblib.load("heart_scaler.pkl")
except:
    heart_scaler = None

try:
    diabetes_scaler = joblib.load("diabetes_scaler.pkl")
except:
    diabetes_scaler = None

# Load accuracies
def load_accuracy(filename: str) -> str:
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except:
        return "N/A"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    heart_accuracy = load_accuracy("heart_accuracy.txt")
    diabetes_accuracy = load_accuracy("diabetes_accuracy.txt")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "heart_accuracy": heart_accuracy,
        "diabetes_accuracy": diabetes_accuracy
    })

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
       
    if heart_scaler:
        input_data = heart_scaler.transform(input_data)

    prediction = heart_model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    heart_accuracy = load_accuracy("heart_accuracy.txt")
    diabetes_accuracy = load_accuracy("diabetes_accuracy.txt")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "heart_result": result,
        "heart_accuracy": heart_accuracy,
        "diabetes_accuracy": diabetes_accuracy
    })

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

    if diabetes_scaler:
        input_data = diabetes_scaler.transform(input_data)

    prediction = diabetes_model.predict(input_data)
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

    heart_accuracy = load_accuracy("heart_accuracy.txt")
    diabetes_accuracy = load_accuracy("diabetes_accuracy.txt")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "diabetes_result": result,
        "heart_accuracy": heart_accuracy,
        "diabetes_accuracy": diabetes_accuracy
    })
