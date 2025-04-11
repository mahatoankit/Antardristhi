from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status,
    Form,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import json
import logging
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AnalysisEngine
from .services.analysis_service import AnalysisEngine

# Import database models and functions
from .core.database import get_db, engine, SessionLocal
from .models.models import Base, User as UserModel
from .crud import crud as user_crud
from .schemas import schemas

# Configuration (move to config.py in production)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Create tables if they don't exist yet
Base.metadata.create_all(bind=engine)

# Sample database (will be replaced with real DB below)
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


# Initialize Analysis Engine
analysis_engine = AnalysisEngine()


# New models for ML-powered analysis
class AnalysisRequest(BaseModel):
    prompt: str
    data_id: str


class TimeSeriesRequest(BaseModel):
    data_id: str
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    periods: Optional[int] = 30


class ClusteringRequest(BaseModel):
    data_id: str
    features: Optional[List[str]] = None
    n_clusters: Optional[int] = None


class AnomalyDetectionRequest(BaseModel):
    data_id: str
    features: Optional[List[str]] = None
    contamination: Optional[float] = 0.05


class VisualizationRequest(BaseModel):
    analysis_id: str
    viz_type: str


# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/login"
)  # Updated to match the new endpoint

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
        "sample_data": df.head().to_dict(orient="records"),
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
    file: UploadFile = File(...), current_user: User = Depends(get_current_active_user)
):
    try:
        # Check file type
        if file.filename.endswith(".csv"):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV or Excel.",
            )

        # Preprocess data with ML engine
        preprocessing_result = analysis_engine.preprocess_data(df)

        if "error" in preprocessing_result:
            raise HTTPException(status_code=500, detail=preprocessing_result["error"])

        # Basic analysis (keeping original functionality)
        analysis = analyze_dataframe(df, "overview")

        # Combine results
        result = {
            "filename": file.filename,
            "user": current_user.username,
            "basic_analysis": analysis,
            "ml_preprocessing": preprocessing_result,
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/", response_model=DataAnalysisResult)
async def analyze_data(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    try:
        # Read file
        if file.filename.endswith(".csv"):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Preprocess with ML engine
        preprocessing_result = analysis_engine.preprocess_data(df)

        if "error" in preprocessing_result:
            # Fall back to basic analysis if ML preprocessing fails
            analysis = analyze_dataframe(df, prompt)
            return DataAnalysisResult(
                insight=analysis.get("insight", "Analysis completed"),
                visualization=analysis.get("visualization"),
                data=analysis,
            )

        # Use ML-powered analysis with the prompt
        ml_result = analysis_engine.analyze_data_with_prompt(
            preprocessing_result["data_id"], prompt
        )

        if "error" in ml_result:
            # Fall back to basic analysis if ML analysis fails
            analysis = analyze_dataframe(df, prompt)
            return DataAnalysisResult(
                insight=analysis.get("insight", "Analysis completed"),
                visualization=analysis.get("visualization"),
                data=analysis,
            )

        # Extract explanation and visualization
        explanation = ml_result.get("result", {}).get(
            "explanation", "Analysis completed"
        )

        # Prepare result using ML analysis
        return DataAnalysisResult(
            insight=explanation,
            visualization=ml_result.get("result", {})
            .get("plots", {})
            .get("clusters", ""),
            data=ml_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Business Insights API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# New routes for ML analysis
@app.post("/ml/analyze/", tags=["ML Analysis"])
async def analyze_with_ml(analysis_request: AnalysisRequest):
    """
    Analyze data using natural language prompt and ML
    """
    result = analysis_engine.analyze_data_with_prompt(
        analysis_request.data_id, analysis_request.prompt
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/ml/time-series/", tags=["ML Analysis"])
async def analyze_time_series(request: TimeSeriesRequest):
    """
    Perform time series analysis and forecasting
    """
    result = analysis_engine.analyze_with_time_series(
        request.data_id, request.date_col, request.value_col, request.periods
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/ml/clustering/", tags=["ML Analysis"])
async def analyze_clustering(request: ClusteringRequest):
    """
    Perform clustering/segmentation analysis
    """
    result = analysis_engine.analyze_with_clustering(
        request.data_id, request.features, request.n_clusters
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/ml/anomaly-detection/", tags=["ML Analysis"])
async def analyze_anomalies(request: AnomalyDetectionRequest):
    """
    Perform anomaly/outlier detection
    """
    result = analysis_engine.analyze_with_anomaly_detection(
        request.data_id, request.features, request.contamination
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/ml/visualization/", tags=["ML Analysis"])
async def get_visualization(request: VisualizationRequest):
    """
    Get visualization data for a specific analysis
    """
    result = analysis_engine.get_visualization_data(
        request.analysis_id, request.viz_type
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/ml/analyses/", tags=["ML Analysis"])
async def list_analyses():
    """
    Get a list of all analyses
    """
    return analysis_engine.get_analysis_list()


@app.get("/ml/datasets/", tags=["ML Analysis"])
async def list_datasets():
    """
    Get a list of all datasets
    """
    return analysis_engine.get_data_list()


@app.post("/ml/clear-cache/", tags=["ML Analysis"])
async def clear_cache(older_than_days: int = 1):
    """
    Clear cached data and analyses
    """
    return analysis_engine.clear_cache(older_than_days)


# Authentication routes
@app.post("/auth/register", response_model=schemas.User, tags=["Authentication"])
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user in the database
    """
    print(f"Attempting to register user: {user.username}, email: {user.email}")

    # Check if username already exists
    db_user = user_crud.get_user_by_username(db, username=user.username)
    if db_user:
        print(f"Username {user.username} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email already exists
    db_user = user_crud.get_user_by_email(db, email=user.email)
    if db_user:
        print(f"Email {user.email} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create the user
    try:
        new_user = user_crud.create_user(db=db, user=user)
        print(f"Successfully created user: {new_user.username}")
        return new_user
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@app.post("/auth/login", response_model=schemas.Token, tags=["Authentication"])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    Authenticate and login a user
    """
    # Try to get user from database
    user = user_crud.get_user_by_username(db, username=form_data.username)
    if not user or not user_crud.verify_password(
        form_data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer", "user": user}


@app.get("/auth/user", response_model=schemas.User, tags=["Authentication"])
def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
):
    """
    Get the current authenticated user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Get user from database
    user = user_crud.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception

    return user


# Initialize default profiles
@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        user_crud.create_default_profiles(db)
        logger.info("Default profiles created successfully")
    except Exception as e:
        logger.error(f"Error creating default profiles: {e}")
    finally:
        db.close()
