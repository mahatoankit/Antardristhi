from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

# Configuration (move to config.py in production)
SECRET_KEY = "your-secret-key-here"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Sample database (replace with real DB in production)
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "user_type": "business",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    user_type: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class ChatMessage(BaseModel):
    content: str
    is_user: bool
    timestamp: datetime = datetime.now()

class DataAnalysisResult(BaseModel):
    insight: str
    visualization: Optional[str] = None
    data: Optional[dict] = None

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Business Insights API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Basic data analysis functions
def analyze_dataframe(df: pd.DataFrame, prompt: str) -> dict:
    """Basic analysis function that responds to common business questions"""
    analysis = {
        "columns": list(df.columns),
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head().to_dict(orient='records')
    }
    
    prompt = prompt.lower()
    
    if "sales" in prompt and "trend" in prompt:
        if "date" in df.columns or "month" in df.columns:
            date_col = "date" if "date" in df.columns else "month"
            analysis["insight"] = f"Sales trends analysis by {date_col}"
            analysis["visualization"] = "line_chart"
        else:
            analysis["insight"] = "No date column found for trend analysis"
    
    elif "summary" in prompt or "overview" in prompt:
        analysis["insight"] = "Dataset summary"
        analysis["stats"] = df.describe().to_dict()
    
    elif "customer" in prompt and "segment" in prompt:
        if "customer" in df.columns:
            analysis["insight"] = "Customer segmentation analysis"
            analysis["visualization"] = "pie_chart"
        else:
            analysis["insight"] = "No customer data found for segmentation"
    
    else:
        analysis["insight"] = "Here are some basic insights about your data"
    
    return analysis

# Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/upload/")
async def upload_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    try:
        # Check file type
        if file.filename.endswith('.csv'):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel.")
        
        # Basic analysis
        analysis = analyze_dataframe(df, "overview")
        
        return {
            "filename": file.filename,
            "user": current_user.username,
            "analysis": analysis
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/", response_model=DataAnalysisResult)
async def analyze_data(
    prompt: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    try:
        # Read file
        if file.filename.endswith('.csv'):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Perform analysis based on prompt
        analysis = analyze_dataframe(df, prompt)
        
        return DataAnalysisResult(
            insight=analysis.get("insight", "Analysis completed"),
            visualization=analysis.get("visualization"),
            data=analysis
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Business Insights API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }