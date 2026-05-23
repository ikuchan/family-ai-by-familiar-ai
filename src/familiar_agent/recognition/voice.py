"""Voice-based speaker identification (optional — requires resemblyzer).

Register a person's voice:
    from familiar_agent.recognition.voice import VoiceIdentifier
    vi = VoiceIdentifier(manager)
    vi.register_voice("alice", "/path/to/sample.wav")
"""
from __future__ import annotations
import asyncio, json, logging, pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager, RecognitionHint

logger = logging.getLogger(__name__)
VOICE_DB = Path.home() / ".familiar_ai" / "voices.pkl"


class VoiceIdentifier:
    def __init__(self, manager: "PersonMemoryManager") -> None:
        self._manager = manager
        self._embeddings: dict[str, np.ndarray] = self._load()

    def _load(self) -> dict[str, np.ndarray]:
        if VOICE_DB.exists():
            try:
                with open(VOICE_DB, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        VOICE_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(VOICE_DB, "wb") as f:
            pickle.dump(self._embeddings, f)

    def register_voice(self, person_id: str, audio_path: str) -> bool:
        """Extract and store voice embedding for a person."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            wav = preprocess_wav(audio_path)
            vec = VoiceEncoder().embed_utterance(wav)
            self._embeddings[person_id] = vec
            self._save()
            logger.info("Registered voice for %s", person_id)
            return True
        except ImportError:
            logger.debug("resemblyzer not installed")
            return False
        except Exception as e:
            logger.warning("Voice registration error: %s", e)
            return False

    async def identify_async(self, audio_path: str) -> "RecognitionHint | None":
        return await asyncio.to_thread(self._identify_sync, audio_path)

    def _identify_sync(self, audio_path: str) -> "RecognitionHint | None":
        if not self._embeddings:
            return None
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            from ..person_memory_manager import RecognitionHint
            wav = preprocess_wav(audio_path)
            q   = VoiceEncoder().embed_utterance(wav)
            best_pid, best_score = None, 0.0
            for pid, ref in self._embeddings.items():
                score = float(np.dot(q, ref) / (np.linalg.norm(q) * np.linalg.norm(ref) + 1e-10))
                if score > best_score:
                    best_score, best_pid = score, pid
            if best_pid and best_score > 0.75:
                return RecognitionHint(
                    person_id=best_pid, confidence=best_score,
                    source="voice", reason=f"cosine={best_score:.3f}",
                )
            return None
        except ImportError:
            return None
        except Exception as e:
            logger.warning("Voice identification error: %s", e)
            return None
