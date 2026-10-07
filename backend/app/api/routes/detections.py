"""
Detection review endpoints for the BART defect dashboard.

Save as backend/app/api/routes/detections.py, then in backend/app/api/main.py:
    from app.api.routes import detections
    api_router.include_router(detections.router)

Images are read from backend/data/detections by default. Set the IMAGE_ROOT
environment variable to use a different folder (e.g. where the model writes its images).
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Detection,
    DetectionPublic,
    DetectionsPublic,
    DetectionStatus,
    ReviewDecision,
)

router = APIRouter(prefix="/detections", tags=["detections"])

# Default: backend/data/detections (this file is backend/app/api/routes/detections.py)
DEFAULT_IMAGE_ROOT = Path(__file__).resolve().parents[3] / "data" / "detections"
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", DEFAULT_IMAGE_ROOT)).resolve()


def to_public(d: Detection) -> DetectionPublic:
    return DetectionPublic(
        **d.model_dump(exclude={"image_path", "created_at"}),
        image_url=f"/api/v1/detections/{d.id}/image",
    )


@router.get("/", response_model=DetectionsPublic)
def list_detections(
    session: SessionDep,
    current_user: CurrentUser,
    status: DetectionStatus | None = None,
    defect_type: str | None = None,
    track_segment: str | None = None,
    min_confidence: float = Query(0, ge=0, le=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> DetectionsPublic:
    """List detections with filters. Used by the table and map views."""
    filters = [col(Detection.confidence) >= min_confidence]
    if status:
        filters.append(Detection.status == status)
    if defect_type:
        filters.append(Detection.defect_type == defect_type)
    if track_segment:
        filters.append(Detection.track_segment == track_segment)

    count = session.exec(select(func.count()).select_from(Detection).where(*filters)).one()
    rows = session.exec(
        select(Detection)
        .where(*filters)
        .order_by(col(Detection.captured_at).desc())
        .offset(skip)
        .limit(limit)
    ).all()
    return DetectionsPublic(data=[to_public(d) for d in rows], count=count)


@router.get("/next", response_model=DetectionPublic | None)
def next_pending(session: SessionDep, current_user: CurrentUser) -> DetectionPublic | None:
    """Next detection for the review queue: least-confident pending detections first,
    since those are the ones most likely to need a human call."""
    d = session.exec(
        select(Detection)
        .where(Detection.status == DetectionStatus.pending)
        .order_by(col(Detection.confidence).asc(), col(Detection.captured_at).asc())
        .limit(1)
    ).first()
    return to_public(d) if d else None


@router.get("/{detection_id}", response_model=DetectionPublic)
def get_detection(
    detection_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> DetectionPublic:
    d = session.get(Detection, detection_id)
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    return to_public(d)


@router.get("/{detection_id}/image")
def get_detection_image(
    detection_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> FileResponse:
    """Serve the image file. The bounding box is drawn by the frontend on top of it."""
    d = session.get(Detection, detection_id)
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    path = (IMAGE_ROOT / d.image_path).resolve()
    if not path.is_relative_to(IMAGE_ROOT) or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.post("/{detection_id}/review", response_model=DetectionPublic)
def review_detection(
    detection_id: uuid.UUID,
    body: ReviewDecision,
    session: SessionDep,
    current_user: CurrentUser,
) -> DetectionPublic:
    """Engineer accepts or rejects a detection. Records who decided and when."""
    if body.decision == DetectionStatus.pending:
        raise HTTPException(status_code=422, detail="Decision must be 'accepted' or 'rejected'")

    # Lock the row so two engineers can't review the same detection at once
    d = session.exec(
        select(Detection).where(Detection.id == detection_id).with_for_update()
    ).first()
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    if d.status != DetectionStatus.pending:
        raise HTTPException(
            status_code=409,
            detail=f"Already {d.status.value} by another reviewer",
        )

    d.status = body.decision
    d.review_note = body.note
    d.reviewed_by_id = current_user.id
    d.reviewed_at = datetime.now(timezone.utc)
    session.add(d)
    session.commit()
    session.refresh(d)
    return to_public(d)


@router.post("/{detection_id}/reopen", response_model=DetectionPublic)
def reopen_detection(
    detection_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> DetectionPublic:
    """Undo a review (superusers only), e.g. if an engineer clicked the wrong button."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Only admins can reopen detections")
    d = session.get(Detection, detection_id)
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    d.status = DetectionStatus.pending
    d.reviewed_by_id = None
    d.reviewed_at = None
    d.review_note = None
    session.add(d)
    session.commit()
    session.refresh(d)
    return to_public(d)