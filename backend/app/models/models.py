from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Float,
    JSON,
    Table,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from ..core.database import Base

# Association table for many-to-many relationships
user_profiles = Table(
    "user_profiles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("profile_id", Integer, ForeignKey("profiles.id")),
)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    full_name = Column(String(100))
    hashed_password = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    profiles = relationship("Profile", secondary=user_profiles, back_populates="users")
    uploads = relationship("DataUpload", back_populates="user")
    chats = relationship("Chat", back_populates="user")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True)
    description = Column(String(255))

    # Relationships
    users = relationship("User", secondary=user_profiles, back_populates="profiles")


class DataUpload(Base):
    __tablename__ = "data_uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    original_filename = Column(String(255))
    file_path = Column(String(255))
    file_size = Column(Integer)
    file_type = Column(String(50))
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    data_hash = Column(String(64), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    # Metadata about the dataset
    row_count = Column(Integer)
    column_count = Column(Integer)
    columns_metadata = Column(JSON)  # Store column names, types, etc.

    # Relationships
    user = relationship("User", back_populates="uploads")
    analyses = relationship("Analysis", back_populates="data_upload")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    data_upload_id = Column(Integer, ForeignKey("data_uploads.id"))
    analysis_type = Column(
        String(50)
    )  # e.g., "clustering", "time_series", "anomaly_detection"
    prompt = Column(Text)
    parameters = Column(JSON)  # Store analysis parameters
    results = Column(JSON)  # Store analysis results
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    data_upload = relationship("DataUpload", back_populates="analyses")
    visualizations = relationship("Visualization", back_populates="analysis")


class Visualization(Base):
    __tablename__ = "visualizations"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    viz_type = Column(String(50))  # e.g., "line_chart", "scatter_plot", "heatmap"
    data = Column(JSON)  # Store visualization data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    analysis = relationship("Analysis", back_populates="visualizations")


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship(
        "ChatMessage", back_populates="chat", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    content = Column(Text)  # Text representation of the message
    content_json = Column(
        JSON, nullable=True
    )  # Structured content for rich messages (images, tables)
    image_data = Column(
        Text, nullable=True
    )  # Base64 encoded image data for visualizations
    message_type = Column(
        String(50), default="text"
    )  # Type of message: text, chart, image, table, error
    is_user = Column(
        Boolean, default=True
    )  # True for user messages, False for assistant responses
    data_reference_id = Column(
        String(255), nullable=True
    )  # Reference to data being analyzed
    analysis_reference_id = Column(
        String(255), nullable=True
    )  # Reference to analysis performed
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    chat = relationship("Chat", back_populates="messages")
