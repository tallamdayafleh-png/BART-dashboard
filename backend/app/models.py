import uuid
from enum import Enum
from datetime import UTC, datetime

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(default_factory=get_datetime_utc)
    items: list[Item] = Relationship(back_populates="owner", cascade_delete=True)


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(SQLModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(default_factory=get_datetime_utc)
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)

class DetectionStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"


# ---------- Table ----------

class Detection(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Image: store only the path/key; the file lives on disk or in object storage
    image_path: str = Field(max_length=512)

    # Bounding box, normalized 0-1 relative to image width/height
    bbox_x: float = Field(ge=0, le=1)
    bbox_y: float = Field(ge=0, le=1)
    bbox_w: float = Field(ge=0, le=1)
    bbox_h: float = Field(ge=0, le=1)

    # What the model found
    defect_type: str = Field(max_length=64, index=True)   # e.g. "crack", "squat", "loose_fastener"
    confidence: float = Field(ge=0, le=1, index=True)
    model_name: str = Field(max_length=128)
    model_version: str = Field(max_length=64)

    # Where it was found (GPS and/or BART track reference)
    latitude: float | None = None
    longitude: float | None = None
    track_segment: str | None = Field(default=None, max_length=64)   # e.g. "A-line, Track 1"
    milepost: float | None = None

    captured_at: datetime = Field(index=True)        # when the camera took the image
    created_at: datetime = Field(default_factory=get_datetime_utc)

    # Engineer review
    status: DetectionStatus = Field(default=DetectionStatus.pending, index=True)
    reviewed_by_id: uuid.UUID | None = Field(default=None, foreign_key="user.id")
    reviewed_at: datetime | None = None
    review_note: str | None = Field(default=None, max_length=1000)


# ---------- API schemas ----------

class DetectionPublic(SQLModel):
    id: uuid.UUID
    image_url: str            # filled in by the route, points at /detections/{id}/image
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    defect_type: str
    confidence: float
    model_name: str
    model_version: str
    latitude: float | None
    longitude: float | None
    track_segment: str | None
    milepost: float | None
    captured_at: datetime
    status: DetectionStatus
    reviewed_by_id: uuid.UUID | None
    reviewed_at: datetime | None
    review_note: str | None


class DetectionsPublic(SQLModel):
    data: list[DetectionPublic]
    count: int


class ReviewDecision(SQLModel):
    decision: DetectionStatus            # must be "accepted" or "rejected"
    note: str | None = Field(default=None, max_length=1000)
