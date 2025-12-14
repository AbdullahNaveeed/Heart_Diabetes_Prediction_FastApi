Heart Disease & Diabetes Prediction Web App
Overview

This project is a FastAPI-based machine learning web application that predicts heart disease and diabetes using pre-trained models. It combines a Python backend for inference, basic HTML/CSS for user interaction, and Jupyter notebooks for model training and experimentation.

This is not just an API — it includes:

Trained ML models (.pkl)
Data preprocessing using scalers
A browser-based UI
Reproducible training notebooks

If you break something here, it’s because you didn’t respect the data preprocessing or environment setup. Read properly.

Features

Heart Disease Prediction
Diabetes Prediction
Web Interface
REST API
Model Training

Jupyter notebooks included
Raw datasets provided (CSV / XLSX)



How to Run the FastAPI Project
1. Clone the Repository
git clone https://github.com/your-username/heart-diabetes-fastapi.git
cd heart-diabetes-fastapi

2. Create a Virtual Environment (Don’t Skip This)
python -m venv venv
Activate it:
Windows:
venv\Scripts\activate
Linux / macOS:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt
If this fails, your Python version is probably outdated. Use Python 3.9+.

4. Run the FastAPI Server
uvicorn app.main:app --reload
If your main.py is not inside app/, adjust the path accordingly. Don’t guess — look at your folder.

5. Open the Application
Web UI:
http://127.0.0.1:8000
Interactive API Docs (Swagger):
http://127.0.0.1:8000/docs
If /docs doesn’t work, your FastAPI setup is broken.

