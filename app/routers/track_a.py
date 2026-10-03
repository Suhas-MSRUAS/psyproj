"""Track A endpoints: AnnoMI conversations, fictional-audio ASR, RAG chatbot.

This router never imports from track_b and never reads WISDM/UCI HAR data —
see tests/test_separation.py for the audit that checks this holds.
"""
from __future__ import annotations

import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from common.backends import (
    build_asr_backend,
    build_embedding_backend,
    build_llm_backend,
    build_whisper_backend,
    resolve_device,
    retrieval_min_similarity,
)
from common.config import get_config, resolve_path
from common.schemas import AudioCaseOutput, ChatResponse, ConversationOutput, UploadedAudioOutput
from track_a.asr import evaluate_case, evaluate_upload, list_cases, validate_audio
from track_a.extraction import extract_from_conversation
from track_a.ingest_annomi import build_conversations, load_annomi
from track_a.rag.chatbot import answer_question
from track_a.rag.corpus import load_corpus
from track_a.rag.index import RAGIndex

router = APIRouter(prefix="/track-a", tags=["track-a"])


@lru_cache(maxsize=1)
def _annomi_conversations():
    cfg = get_config()
    df, ingestion_report = load_annomi(resolve_path(cfg["paths"]["annomi_simple_csv"]))
    return build_conversations(df), ingestion_report


def _load_once(builder):
    """Like lru_cache(maxsize=1), but thread-safe: the startup warm-up thread
    and a request arriving meanwhile must not both load a multi-GB model
    onto the GPU at the same time."""
    lock = threading.Lock()
    box: list = []

    def get():
        if not box:
            with lock:
                if not box:
                    box.append(builder())
        return box[0]

    get.loaded = lambda: bool(box)
    return get


@_load_once
def _rag_index() -> RAGIndex:
    cfg = get_config()
    corpus = load_corpus(resolve_path(cfg["paths"]["rag_corpus_dir"]))
    return RAGIndex(corpus, build_embedding_backend())


_llm = _load_once(build_llm_backend)
_asr = _load_once(build_asr_backend)
_whisper = _load_once(build_whisper_backend)


def _audio_dir() -> Path:
    cfg = get_config()
    real_dir = resolve_path(cfg["paths"]["fictional_audio_dir"])
    if list(real_dir.glob("*.wav")):
        return real_dir
    return resolve_path(cfg["paths"]["fixtures_dir"]) / "audio"


@router.get("/annomi")
def list_annomi_conversations():
    conversations, _ = _annomi_conversations()
    return [
        {"conversation_id": f"annomi-{tid}", "mi_quality": c.mi_quality, "topic": c.topic, "n_turns": len(c.turns)}
        for tid, c in conversations.items()
    ]


@router.get("/annomi/{transcript_id}", response_model=ConversationOutput)
def get_annomi_conversation(transcript_id: int):
    conversations, _ = _annomi_conversations()
    conv = conversations.get(transcript_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"No AnnoMI transcript_id={transcript_id}")
    statement = extract_from_conversation(conv)
    return ConversationOutput(
        conversation_id=f"annomi-{conv.transcript_id}",
        mi_quality=conv.mi_quality,
        topic=conv.topic,
        turns=[{"utterance_id": t.utterance_id, "interlocutor": t.interlocutor, "text": t.text} for t in conv.turns],
        statements=[statement],
    )


def warm_up() -> None:
    """Load every Track A model once, so the first chat/upload isn't slow."""
    _rag_index()
    _llm()
    try:
        _whisper()
    except ImportError:
        pass  # upload endpoint reports this itself


@router.get("/status")
def model_status():
    cfg = get_config()["track_a"]
    loaded = {"chatbot": _llm.loaded(), "search": _rag_index.loaded(), "whisper": _whisper.loaded()}
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = {"name": torch.cuda.get_device_name(0), "vram_used_gb": round((total - free) / 1e9, 1),
                   "vram_total_gb": round(total / 1e9, 1)}
    except ImportError:
        pass
    return {
        "chatbot": cfg["llm"]["real_checkpoint"] if cfg["llm"]["backend"] == "real" else "stub (extractive)",
        "search": cfg["embedding"]["real_checkpoint"] if cfg["embedding"]["backend"] == "real" else "stub (bag-of-words)",
        "whisper": cfg["asr"]["real_checkpoint"],
        "device": resolve_device(),
        "gpu": gpu,
        "loaded": loaded,
    }


@router.get("/passages")
def list_rag_passages():
    """Citation metadata for the approved RAG corpus, so a UI can show what a cited chunk id points to."""
    return [{"chunk_id": p.chunk_id, "title": p.title, "url": p.url} for p in _rag_index().passages]


@router.get("/audio")
def list_audio_cases():
    return [p.stem for p in list_cases(_audio_dir())]


@router.get("/audio/{case_id}", response_model=AudioCaseOutput)
def get_audio_case(case_id: str):
    wav_path = _audio_dir() / f"{case_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"No audio case '{case_id}'")
    validation = validate_audio(wav_path)
    if validation.flags:
        raise HTTPException(status_code=422, detail={"case_id": case_id, "quality_flags": validation.flags})
    return evaluate_case(wav_path, _asr())


_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@router.post("/audio/upload", response_model=UploadedAudioOutput)
def upload_audio(file: UploadFile = File(...), reference_transcript: Optional[str] = Form(None)):
    """Transcribe an uploaded clip with Whisper (wav/mp3/m4a/ogg/flac...).

    The clip is written to a temp file only for the duration of the request
    and deleted afterwards. Nothing is stored or linked to AnnoMI.
    """
    suffix = Path(file.filename or "clip").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File larger than 100 MB.")
            tmp.write(chunk)
        tmp.close()
        try:
            asr = _whisper()
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Whisper is not installed. Run: pip install torch openai-whisper (see requirements-real.txt).",
            )
        try:
            return evaluate_upload(Path(tmp.name), file.filename or "clip", asr, reference_transcript)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)


class ChatRequest(BaseModel):
    conversation_or_case_id: str
    question: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cfg = get_config()["track_a"]["retrieval"]
    return answer_question(
        req.conversation_or_case_id,
        req.question,
        _rag_index(),
        _llm(),
        top_k=cfg["top_k"],
        min_similarity=retrieval_min_similarity(),
    )
