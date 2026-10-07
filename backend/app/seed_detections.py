"""
Fill the database with FAKE defect detections for testing the dashboard.

Everything here is made up: generated track images, approximate locations near
BART stations, random defect types and confidence scores. No real BART data.

Save as backend/app/seed_detections.py, then from the backend folder run:

    uv run python app/seed_detections.py            # add 25 fake detections
    uv run python app/seed_detections.py --count 50 # add 50
    uv run python app/seed_detections.py --reset    # delete all detections + images first
"""

import argparse
import random
import struct
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session, delete, select

from app.api.routes.detections import IMAGE_ROOT
from app.core.config import settings
from app.core.db import engine
from app.models import Detection, DetectionStatus, User

# Approximate locations near BART stations (for map pins only, not survey data)
LOCATIONS = [
    ("West Oakland - 12th St, Track 1", 37.8048, -122.2950),
    ("Lake Merritt - Fruitvale, Track 2", 37.7970, -122.2653),
    ("Fruitvale - Coliseum, Track 1", 37.7749, -122.2241),
    ("MacArthur - Rockridge, Track 2", 37.8290, -122.2671),
    ("Balboa Park - Glen Park, Track 1", 37.7217, -122.4475),
    ("Daly City - Balboa Park, Track 2", 37.7061, -122.4690),
    ("Fremont - Union City, Track 1", 37.5574, -121.9764),
    ("Pleasant Hill - Walnut Creek, Track 2", 37.9284, -122.0560),
    ("Richmond - El Cerrito del Norte, Track 1", 37.9369, -122.3533),
    ("Dublin/Pleasanton - West Dublin, Track 2", 37.7017, -121.8992),
]

DEFECT_TYPES = ["crack", "squat", "corrugation", "loose_fastener", "missing_clip", "spalled_tie"]

W, H = 640, 480
RAIL_X = (180, 460)   # left edge of each rail in the generated image
RAIL_W = 22


# ---------- tiny PNG writer (no extra packages needed) ----------

def write_png(path: Path, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + bytes(pixels[y * W * 3:(y + 1) * W * 3]) for y in range(H))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def make_track_image(rng: random.Random, defect_type: str) -> tuple[bytearray, tuple[float, float, float, float]]:
    """Draw a top-down fake track (ballast, ties, two rails) with a dark defect mark.
    Returns the pixels and the defect's bounding box as fractions (x, y, w, h)."""
    px = bytearray(W * H * 3)

    def put(x: int, y: int, c: tuple[int, int, int]) -> None:
        if 0 <= x < W and 0 <= y < H:
            i = (y * W + x) * 3
            px[i:i + 3] = bytes(c)

    # Ballast: grey gravel noise
    for y in range(H):
        for x in range(W):
            g = 95 + rng.randint(-25, 25)
            put(x, y, (g, g - 4, g - 10))

    # Ties: brown horizontal bands
    for top in range(10, H, 70):
        for y in range(top, top + 34):
            for x in range(110, 530):
                b = rng.randint(-8, 8)
                put(x, y, (105 + b, 78 + b, 55 + b))

    # Rails: light steel vertical bands with a shiny centre line
    for rx in RAIL_X:
        for y in range(H):
            for x in range(rx, rx + RAIL_W):
                shine = 30 if abs(x - (rx + RAIL_W // 2)) < 3 else 0
                put(x, y, (160 + shine, 165 + shine, 170 + shine))

    # Defect: dark irregular mark, on a rail or on a tie depending on type
    if defect_type in ("spalled_tie", "missing_clip", "loose_fastener"):
        cx = rng.choice(RAIL_X) + rng.choice([-20, RAIL_W + 20])
    else:
        cx = rng.choice(RAIL_X) + RAIL_W // 2
    cy = rng.randint(60, H - 60)
    half_w, half_h = rng.randint(12, 22), rng.randint(18, 40)

    x, y = cx, cy - half_h
    while y < cy + half_h:
        for dx in range(-2, 3):
            put(x + dx, y, (30, 25, 25))
        x = max(cx - half_w, min(cx + half_w, x + rng.randint(-2, 2)))
        y += 1

    pad = 10
    bx, by = (cx - half_w - pad) / W, (cy - half_h - pad) / H
    bw, bh = (2 * (half_w + pad)) / W, (2 * (half_h + pad)) / H
    clamp = lambda v: round(min(max(v, 0.0), 1.0), 4)  # noqa: E731
    return px, (clamp(bx), clamp(by), clamp(bw), clamp(bh))


# ---------- seeding ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Add fake detections for testing")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--reset", action="store_true", help="delete existing detections and images first")
    parser.add_argument("--seed", type=int, default=None, help="random seed for repeatable data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        if args.reset:
            session.exec(delete(Detection))
            session.commit()
            for f in IMAGE_ROOT.glob("fake_*.png"):
                f.unlink()
            print("Deleted existing detections and fake images.")

        admin = session.exec(select(User).where(User.email == settings.FIRST_SUPERUSER)).first()
        now = datetime.now(UTC)

        for n in range(args.count):
            defect_type = rng.choice(DEFECT_TYPES)
            segment, lat, lon = rng.choice(LOCATIONS)
            pixels, (bx, by, bw, bh) = make_track_image(rng, defect_type)

            captured_at = now - timedelta(hours=rng.randint(1, 24 * 14), minutes=rng.randint(0, 59))
            file_name = f"fake_{captured_at:%Y%m%d_%H%M%S}_{n:03d}.png"
            write_png(IMAGE_ROOT / file_name, pixels)

            d = Detection(
                image_path=file_name,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                defect_type=defect_type,
                confidence=round(rng.uniform(0.35, 0.99), 3),
                model_name="fake-yolo-track",
                model_version="0.0-test",
                latitude=round(lat + rng.uniform(-0.004, 0.004), 6),
                longitude=round(lon + rng.uniform(-0.004, 0.004), 6),
                track_segment=segment,
                milepost=round(rng.uniform(0.1, 30.0), 2),
                captured_at=captured_at,
            )

            # About 1 in 5 already reviewed, so the list view has some variety
            if admin and rng.random() < 0.2:
                d.status = rng.choice([DetectionStatus.accepted, DetectionStatus.rejected])
                d.reviewed_by_id = admin.id
                d.reviewed_at = captured_at + timedelta(hours=rng.randint(1, 12))
                d.review_note = "Seeded test review"

            session.add(d)
            print(f"  {n + 1:>3}/{args.count}  {defect_type:<15} {segment}")

        session.commit()

    print(f"\nAdded {args.count} fake detections. Images are in {IMAGE_ROOT}")


if __name__ == "__main__":
    main()