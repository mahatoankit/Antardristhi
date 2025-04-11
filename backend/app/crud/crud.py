from sqlalchemy.orm import Session
from ..models.models import (
    User,
    Profile,
    DataUpload,
    Analysis,
    Visualization,
    Chat,
    ChatMessage,
)
from ..schemas import schemas
from passlib.context import CryptContext
from typing import List, Optional, Dict, Any
import hashlib
import os
import json

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# User operations
def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    return db.query(User).offset(skip).limit(limit).all()


# Create default profiles if they don't exist
def create_default_profiles(db: Session):
    default_profiles = [
        {"name": "student", "description": "Student user profile"},
        {"name": "business_owner", "description": "Business owner profile"},
        {"name": "developer", "description": "Developer profile"},
        {"name": "analyst", "description": "Data analyst profile"},
        {"name": "general", "description": "General user profile"},
    ]

    for profile_data in default_profiles:
        existing_profile = get_profile_by_name(db, profile_data["name"])
        if not existing_profile:
            profile = schemas.ProfileCreate(**profile_data)
            create_profile(db, profile)
            print(f"Created default profile: {profile_data['name']}")


def create_user(db: Session, user: schemas.UserCreate) -> User:
    print(f"Creating user with username: {user.username}, email: {user.email}")

    # Create user
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=get_password_hash(user.password),
        is_active=True,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Add profiles if specified
    if user.profiles:
        print(f"Processing profiles: {user.profiles}")
        for profile_name in user.profiles:
            profile = get_profile_by_name(db, profile_name)
            if not profile:
                # Create profile if it doesn't exist
                print(f"Creating missing profile: {profile_name}")
                profile = create_profile(
                    db,
                    schemas.ProfileCreate(
                        name=profile_name, description=f"{profile_name} profile"
                    ),
                )

            db_user.profiles.append(profile)
            print(f"Added profile {profile_name} to user {user.username}")

        db.commit()
        db.refresh(db_user)
    else:
        # Add default 'general' profile if no profiles specified
        print("No profiles specified, adding 'general' profile")
        default_profile = get_profile_by_name(db, "general")
        if not default_profile:
            default_profile = create_profile(
                db,
                schemas.ProfileCreate(
                    name="general", description="General user profile"
                ),
            )
        db_user.profiles.append(default_profile)
        db.commit()
        db.refresh(db_user)

    return db_user


def update_user(
    db: Session, user_id: int, user_update: schemas.UserUpdate
) -> Optional[User]:
    db_user = get_user(db, user_id)
    if not db_user:
        return None

    update_data = user_update.dict(exclude_unset=True)

    # Handle password separately
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    # Handle profiles separately
    if "profiles" in update_data:
        profile_names = update_data.pop("profiles")
        db_user.profiles = []
        for profile_name in profile_names:
            profile = get_profile_by_name(db, profile_name)
            if profile:
                db_user.profiles.append(profile)

    # Update other fields
    for key, value in update_data.items():
        setattr(db_user, key, value)

    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int) -> bool:
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False


# Profile operations
def get_profile(db: Session, profile_id: int) -> Optional[Profile]:
    return db.query(Profile).filter(Profile.id == profile_id).first()


def get_profile_by_name(db: Session, name: str) -> Optional[Profile]:
    return db.query(Profile).filter(Profile.name == name).first()


def get_profiles(db: Session, skip: int = 0, limit: int = 100) -> List[Profile]:
    return db.query(Profile).offset(skip).limit(limit).all()


def create_profile(db: Session, profile: schemas.ProfileCreate) -> Profile:
    db_profile = Profile(**profile.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile


# DataUpload operations
def get_data_upload(db: Session, upload_id: int) -> Optional[DataUpload]:
    return db.query(DataUpload).filter(DataUpload.id == upload_id).first()


def get_data_uploads_by_user(
    db: Session, user_id: int, skip: int = 0, limit: int = 100
) -> List[DataUpload]:
    return (
        db.query(DataUpload)
        .filter(DataUpload.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_data_upload(
    db: Session,
    user_id: int,
    file_path: str,
    original_filename: str,
    file_size: int,
    file_type: str,
    data_df=None,
) -> DataUpload:
    # Generate a unique filename
    filename = f"{hashlib.md5(original_filename.encode()).hexdigest()}_{os.path.basename(original_filename)}"

    # Calculate data hash if dataframe is provided
    data_hash = None
    row_count = None
    column_count = None
    columns_metadata = None

    if data_df is not None:
        # Create a hash of the data content
        data_hash = hashlib.md5(data_df.to_json().encode()).hexdigest()
        row_count = len(data_df)
        column_count = len(data_df.columns)

        # Create metadata for columns
        columns_metadata = {}
        for col in data_df.columns:
            col_type = str(data_df[col].dtype)
            missing_count = data_df[col].isna().sum()
            unique_count = data_df[col].nunique()

            # For numerical columns, add statistics
            if data_df[col].dtype.kind in "ifc":
                columns_metadata[col] = {
                    "type": col_type,
                    "missing_count": int(missing_count),
                    "unique_count": int(unique_count),
                    "min": (
                        float(data_df[col].min()) if not data_df[col].empty else None
                    ),
                    "max": (
                        float(data_df[col].max()) if not data_df[col].empty else None
                    ),
                    "mean": (
                        float(data_df[col].mean()) if not data_df[col].empty else None
                    ),
                    "std": (
                        float(data_df[col].std()) if not data_df[col].empty else None
                    ),
                }
            else:
                columns_metadata[col] = {
                    "type": col_type,
                    "missing_count": int(missing_count),
                    "unique_count": int(unique_count),
                }

    db_upload = DataUpload(
        filename=filename,
        original_filename=original_filename,
        file_path=file_path,
        file_size=file_size,
        file_type=file_type,
        data_hash=data_hash,
        user_id=user_id,
        row_count=row_count,
        column_count=column_count,
        columns_metadata=columns_metadata,
    )

    db.add(db_upload)
    db.commit()
    db.refresh(db_upload)
    return db_upload


# Analysis operations
def get_analysis(db: Session, analysis_id: int) -> Optional[Analysis]:
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


def get_analyses_by_upload(
    db: Session, data_upload_id: int, skip: int = 0, limit: int = 100
) -> List[Analysis]:
    return (
        db.query(Analysis)
        .filter(Analysis.data_upload_id == data_upload_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_analysis(
    db: Session, analysis: schemas.AnalysisCreate, results: Dict[str, Any] = None
) -> Analysis:
    db_analysis = Analysis(
        data_upload_id=analysis.data_upload_id,
        analysis_type=analysis.analysis_type,
        prompt=analysis.prompt,
        parameters=analysis.parameters,
        results=results,
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis


# Visualization operations
def get_visualization(db: Session, viz_id: int) -> Optional[Visualization]:
    return db.query(Visualization).filter(Visualization.id == viz_id).first()


def get_visualizations_by_analysis(
    db: Session, analysis_id: int
) -> List[Visualization]:
    return (
        db.query(Visualization).filter(Visualization.analysis_id == analysis_id).all()
    )


def create_visualization(
    db: Session, visualization: schemas.VisualizationCreate
) -> Visualization:
    db_viz = Visualization(**visualization.dict())
    db.add(db_viz)
    db.commit()
    db.refresh(db_viz)
    return db_viz


# Chat operations
def get_chat(db: Session, chat_id: int) -> Optional[Chat]:
    return db.query(Chat).filter(Chat.id == chat_id).first()


def get_chats_by_user(db: Session, user_id: int) -> List[Chat]:
    return db.query(Chat).filter(Chat.user_id == user_id).all()


def create_chat(db: Session, user_id: int, title: str) -> Chat:
    db_chat = Chat(user_id=user_id, title=title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat


def update_chat(db: Session, chat_id: int, title: str) -> Optional[Chat]:
    db_chat = get_chat(db, chat_id)
    if not db_chat:
        return None

    db_chat.title = title
    db.commit()
    db.refresh(db_chat)
    return db_chat


def delete_chat(db: Session, chat_id: int) -> bool:
    db_chat = get_chat(db, chat_id)
    if db_chat:
        db.delete(db_chat)
        db.commit()
        return True
    return False


# ChatMessage operations
def get_chat_message(db: Session, message_id: int) -> Optional[ChatMessage]:
    return db.query(ChatMessage).filter(ChatMessage.id == message_id).first()


def get_chat_messages(db: Session, chat_id: int) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at)
        .all()
    )


def create_chat_message(db: Session, message: schemas.ChatMessageCreate) -> ChatMessage:
    db_message = ChatMessage(**message.dict())
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message
