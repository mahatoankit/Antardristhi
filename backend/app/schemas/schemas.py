from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# User schemas
class ProfileBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProfileCreate(ProfileBase):
    pass


class Profile(ProfileBase):
    id: int

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str
    profiles: Optional[List[str]] = []


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    profiles: Optional[List[str]] = None
    is_active: Optional[bool] = None


class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    profiles: List[Profile] = []

    class Config:
        from_attributes = True


# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    user: User


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None


# Data Upload schemas
class DataUploadBase(BaseModel):
    filename: str
    file_type: str


class DataUploadCreate(DataUploadBase):
    pass


class DataUpload(DataUploadBase):
    id: int
    original_filename: str
    file_path: str
    file_size: int
    upload_date: datetime
    data_hash: str
    user_id: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# Analysis schemas
class AnalysisBase(BaseModel):
    data_upload_id: int
    analysis_type: str
    prompt: str
    parameters: Optional[Dict[str, Any]] = None


class AnalysisCreate(AnalysisBase):
    pass


class Analysis(AnalysisBase):
    id: int
    results: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Visualization schemas
class VisualizationBase(BaseModel):
    analysis_id: int
    viz_type: str
    data: Dict[str, Any]


class VisualizationCreate(VisualizationBase):
    pass


class Visualization(VisualizationBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Chat schemas
class ChatBase(BaseModel):
    title: str


class ChatCreate(ChatBase):
    pass


class Chat(ChatBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Chat Message schemas
class ChatMessageBase(BaseModel):
    content: str
    is_user: bool = True
    data_reference_id: Optional[int] = None
    analysis_reference_id: Optional[int] = None


class ChatMessageCreate(ChatMessageBase):
    chat_id: int


class ChatMessage(ChatMessageBase):
    id: int
    chat_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Analysis Request schemas
class AnalysisRequest(BaseModel):
    prompt: str
    data_id: int


class TimeSeriesRequest(BaseModel):
    data_id: int
    date_col: Optional[str] = None
    value_col: Optional[str] = None
    periods: Optional[int] = 30


class ClusteringRequest(BaseModel):
    data_id: int
    features: Optional[List[str]] = None
    n_clusters: Optional[int] = None


class AnomalyDetectionRequest(BaseModel):
    data_id: int
    features: Optional[List[str]] = None
    contamination: Optional[float] = 0.05


class VisualizationRequest(BaseModel):
    analysis_id: int
    viz_type: str
