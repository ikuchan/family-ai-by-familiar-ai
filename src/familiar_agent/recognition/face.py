"""Face-based person recognition (optional — requires deepface).

Register a person's face:
    from familiar_agent.recognition.face import register_face
    register_face("alice", "/path/to/photo.jpg")

Face images are stored in ~/.familiar_ai/faces/<name>/
"""
from __future__ import annotations
import asyncio, logging, shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager, RecognitionHint

logger = logging.getLogger(__name__)
FACE_DB = Path.home() / ".familiar_ai" / "faces"


async def recognize_face_async(
    image_path: str,
    manager: "PersonMemoryManager",
) -> "RecognitionHint | None":
    """Identify a person from an image. Returns None if deepface not installed."""
    try:
        import deepface  # noqa: F401
    except ImportError:
        logger.debug("deepface not installed; skipping face recognition")
        return None
    if not FACE_DB.exists():
        return None
    return await asyncio.to_thread(_recognize_sync, image_path, manager)


def _recognize_sync(image_path: str, manager) -> "RecognitionHint | None":
    from deepface import DeepFace
    from ..person_memory_manager import RecognitionHint
    try:
        results = DeepFace.find(
            img_path=image_path, db_path=str(FACE_DB),
            enforce_detection=False, silent=True,
        )
        if not results or results[0].empty:
            return None
        top      = results[0].iloc[0]
        name     = Path(str(top["identity"])).parent.name
        distance = float(top.get("distance", 1.0))
        conf     = max(0.0, 1.0 - distance * 2.0)
        persons  = {p["name"]: p for p in manager.list_persons()}
        if name not in persons:
            return None
        return RecognitionHint(
            person_id=persons[name]["id"],
            confidence=conf,
            source="face",
            reason=f"deepface distance={distance:.3f}",
        )
    except Exception as e:
        logger.warning("Face recognition error: %s", e)
        return None


def register_face(name: str, image_path: str) -> Path:
    """Copy image to face DB for person `name`."""
    dst_dir = FACE_DB / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(image_path).name
    shutil.copy2(image_path, dst)
    logger.info("Registered face for %s: %s", name, dst)
    return dst
