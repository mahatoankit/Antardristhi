from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Optional
import io

app = FastAPI()

# CORS configuration (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic health check endpoint
@app.get("/")
async def root():
    return {"message": "Business Insights API is running"}

# User registration endpoint
@app.post("/register")
async def register_user(user_type: str, email: str, password: str):
    # In a real app, you'd hash the password and store in database
    return {"message": f"User registered as {user_type}", "email": email}

# Data upload endpoint
@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        # Check if file is CSV or Excel
        if file.filename.endswith('.csv'):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Basic data analysis
        analysis = {
            "columns": list(df.columns),
            "sample_data": df.head().to_dict(orient='records'),
            "stats": df.describe().to_dict()
        }
        
        return {"filename": file.filename, "analysis": analysis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))