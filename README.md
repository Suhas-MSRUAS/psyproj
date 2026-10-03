# Mental-Health Support Prototype — Two Independent Tracks

Educational, **inference-only** prototype for the "Counselling Chatbot and
Activity Analysis" assignment. Track A (AnnoMI conversations + a RAG chatbot)
and Track B (WISDM accelerometer + activity report) are fully independent:
neither reads the other's data, and neither produces a depression diagnosis,
severity score, or suicide-risk label. See `tests/test_separation.py` for the
automated proof of that separation.

No training, fine-tuning, or weight updates happen anywhere in this repo.

**Repository:** https://github.com/Suhas-MSRUAS/psyproj

## Project layout

```
common/            shared pydantic schemas + pluggable model backends (stub/real)
track_a/           AnnoMI ingestion, statement extraction, ASR, RAG chatbot, safety routing
track_b/           WISDM ingestion, signal features, activity heuristic, report builder
app/               FastAPI app, two independent routers (/track-a, /track-b)
tests/             the 8 required test scenarios (spec section 8) + separation audit
fixtures/          small synthetic data for deterministic, offline tests
data/rag_corpus/   paraphrased WHO/NIMH/MedlinePlus/NICE passages with citation metadata
data/audio/        where you drop your own consented fictional audio (see its README)
scripts/demo.py    writes representative JSON/CSV outputs to demo_outputs/
config.yaml        all paths, model checkpoints, and thresholds in one place
```

`tracka/AnnoMI-main/` and `trackb/wisdm-dataset/` (note: no underscore) are
the raw downloaded datasets — kept separate from the `track_a`/`track_b`
*code* packages.

## 1. Data provenance and licensing

| Dataset | Where it came from | License | Used by |
|---|---|---|---|
| AnnoMI | [github.com/uccollab/AnnoMI](https://github.com/uccollab/AnnoMI) | CC0 1.0 (public domain) | Track A |
| WISDM Smartphone and Smartwatch Activity and Biometrics | [UCI ML Repository #507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | CC BY 4.0 | Track B |
| UCI HAR v2 (alternative, not wired up) | [UCI ML Repository #341](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | CC BY 4.0 | Track B (unused — WISDM already satisfies the requirement) |
| Fictional audio | **You supply this** — record yourself/a consenting person reading a short invented script, or use TTS. No organizer was available. | N/A | Track A ASR |
| RAG corpus | Short paraphrases (not verbatim copies) of WHO / NIMH / MedlinePlus / NICE NG222 patient-facing pages, with source URL + retrieval date kept per chunk | Source copyright respected via paraphrase + citation | Track A chatbot |

See `data/audio/README.md` for exactly how to add fictional audio cases.

## 2. Setup

```powershell
git clone https://github.com/Suhas-MSRUAS/psyproj.git
cd psyproj
conda create -p .\psy python=3.12 -y     # project env (ignored by git)
conda activate .\psy
pip install -r requirements.txt          # core: pipeline, API, UI, tests (stub backends)
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-real.txt     # real models (see the header of that file)
```

Model weights are downloaded by hand once (nothing in the app downloads
silently), pinned to the exact revisions in `config.yaml`:

```powershell
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
# Whisper-small (small.pt) is fetched by openai-whisper on first use; its SHA-256 is checked.
```

ffmpeg must be on PATH (Whisper decoding and `ffprobe` audio checks).

Datasets expected at (already the defaults in `config.yaml`):
- `tracka/AnnoMI-main/AnnoMI-simple.csv`
- `trackb/wisdm-dataset/wisdm-dataset/raw/phone/accel/data_*_accel_phone.txt`

## 3. Models (pretrained, frozen, no training)

| Task | Checkpoint (spec section 6) | Runs on | Why this one |
|---|---|---|---|
| ASR | `openai/whisper-small` (244M) | GPU, fp16 | Spec's reference ASR. |
| Chat / explanation | `Qwen/Qwen2.5-3B-Instruct` (3.09B), rev `aa8e725` | GPU, fp16 (~6.2 GB) | Spec's listed alternative to SmolLM3-3B. Chosen over Phi-4-mini because Phi is 3.8B (over a strict 3B cap) and does not fit next to Whisper on an 8 GB card. **License: Qwen Research License**, used here for non-commercial educational work only. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (22M), rev `1110a24` | CPU | Spec's listed alternative embedding (Apache 2.0). Kept on CPU: it embeds a question in milliseconds and leaves VRAM for the two larger models. |
| Vector index | FAISS `IndexFlatIP` over normalised vectors (cosine) | CPU | Spec's reference index. |

Tested on an RTX 4060 Laptop GPU (8 GB): all three models loaded together use
~7.7 GB. `track_a.device: auto` in `config.yaml` falls back to CPU when no
CUDA GPU is visible (works, but chat and ASR become much slower).

Generation is greedy (`do_sample=False`), so the same question gives the same
answer. The system prompt is frozen in `common/backends.py::LLM_SYSTEM_PROMPT`:
answer only from the numbered sources, cite them, never diagnose, never give
doses, and treat sources and user text as data, not instructions. The
model's `[n]` markers are mapped back to real chunk ids; a citation the model
invents (out-of-range number) is dropped, never shown.

The model is never asked to judge risk. Urgent/review routing is a fixed
phrase list (`track_a/rag/safety_routing.py`) that runs **before** retrieval,
and those routes, plus "no relevant passage" abstentions, return fixed
non-generated text.

**Stub backends.** Every backend also has a deterministic, dependency-free
stub (`backend: stub` in `config.yaml`). The test suite uses them so it runs
offline in seconds; nothing from a stub is ever presented as a real score.
Uploaded audio always uses real Whisper, since there is no reference for the
ASR stub to simulate from.

## 4. Running

```powershell
uvicorn app.main:app --reload
# http://127.0.0.1:8000/      web UI
# http://127.0.0.1:8000/docs  interactive API docs
```

Models load onto the GPU in a background thread at startup (~25 s); the
sidebar shows "Loading models…" then "Models ready" with VRAM use.

The web UI (`app/static/index.html`, no build step) has four screens:

- **Track A / Conversations**: an AnnoMI transcript in utterance order with
  therapist/client roles, plus the extracted client statement (source turns
  outlined).
- **Track A / Chatbot**: RAG answers with inline citations and source links,
  the routing label, and abstentions.
- **Track A / Speech-to-text**: upload a clip (wav/mp3/m4a/ogg/flac), with
  an optional reference transcript for WER + negation audit. Low-confidence
  Whisper segments are highlighted, not repaired. The file is deleted right
  after processing and is never linked to AnnoMI.
- **Track B / Activity report**: pick a WISDM subject and clip. Shows
  windowed features, an intensity chart, quality flags, limitations and CSV.

The screens share no state and only call their own track's endpoints, so
nothing from Track B can reach the Track A chatbot.

Example calls:
```bash
curl http://127.0.0.1:8000/track-a/annomi                     # list conversations
curl http://127.0.0.1:8000/track-a/annomi/0                    # one conversation + extracted statement
curl -F "file=@clip.mp3" -F "reference_transcript=..." http://127.0.0.1:8000/track-a/audio/upload
curl http://127.0.0.1:8000/track-a/status                      # models, device, VRAM
curl -X POST http://127.0.0.1:8000/track-a/chat -H "Content-Type: application/json" \
     -d '{"conversation_or_case_id":"demo","question":"What are common symptoms of depression?"}'

curl http://127.0.0.1:8000/track-b/subjects                    # list WISDM subject ids
curl http://127.0.0.1:8000/track-b/clips/1600                  # list clips for subject 1600
curl http://127.0.0.1:8000/track-b/report/1600/A/0              # activity report (A = walking)
curl http://127.0.0.1:8000/track-b/report/1600/A/0/csv          # same, as CSV windows
```

Representative outputs: `python scripts/demo.py` writes real examples of
every output type to `demo_outputs/`. `python scripts/evaluate.py` runs the
fixed chatbot evaluation set and writes `eval/results_*.json`. Stop the
server first: each script loads its own copy of the models, and two copies
don't fit in 8 GB. Set `HF_HUB_OFFLINE=1` to guarantee nothing is downloaded.

## 5. Tests

```bash
pytest tests/ -v
```

`tests/test_track_a.py` and `tests/test_track_b.py` implement the 8 required
scenarios from spec section 8; `tests/test_separation.py` implements the
separation audit (both behaviorally and via static import-graph inspection).
Track B's "clean movement" test uses the real WISDM data and is skipped
automatically if it isn't downloaded yet.

## 6. Design notes relevant to the rubric

- **AnnoMI conversation processing**: `track_a/ingest_annomi.py` groups by
  `transcript_id`, sorts by `utterance_id` (not file row order), and flags
  missing fields / duplicate turns / unparsable or non-monotonic timestamps.
  `track_a/extraction.py` only extracts from client turns, preserves
  negation/uncertainty, and traces every item to a source `utterance_id`.
- **Speech transcription**: `track_a/asr.py` validates the WAV (sample rate,
  duration), computes WER via `jiwer`, and audits negation words separately
  so a dropped "not" is flagged even when overall WER is low.
- **RAG chatbot**: `track_a/rag/` retrieves only from the approved corpus +
  the current question, abstains below a similarity threshold instead of
  guessing, and never generates for `urgent`/`review`-routed messages —
  those get a fixed, non-generative escalation message instead.
- **Independent accelerometer DSP**: `track_b/features.py` computes
  magnitude and per-window variance; `track_b/activity.py` is a transparent,
  fixed-threshold heuristic (no training), with windows marked `uncertain`
  rather than guessed when sample coverage is too low (gaps/non-wear).
- **Safety, privacy, separation**: see `tests/test_separation.py`. Routing
  categories are explicitly labeled artificial (`is_artificial_label=True`,
  `source="software_routing_demo"`) since no clinician-reviewed protocol was
  supplied, per the spec's fallback instruction.

## 7. Known limitations (report these, don't hide them)

- The RAG corpus is only 15 short paraphrased passages, so many reasonable
  depression questions correctly end in "no trustworthy source" rather than
  an answer.
- The chat model is an alternative (Qwen2.5-3B-Instruct, Qwen Research
  License), not the spec's reference SmolLM3-3B; the embedding model is the
  spec's alternative (MiniLM), not BGE-small. See section 3 for why.
- ASR results in `eval/evaluation_report.md` come from one synthetic
  (Windows text-to-speech) clip, not from recordings of consenting speakers.
  WER on real voices will be higher. Uploaded clips are evaluated live in
  the UI, but are not stored or reported.
- Safety routing is a fixed phrase list with explicitly artificial labels,
  not a clinician-reviewed protocol. Phrasings outside the list fall through
  to "support".
- All three models together use ~7.7 of 8 GB VRAM; other GPU-heavy apps
  running at the same time can cause out-of-memory errors.
- The activity heuristic struggles specifically with hand-only motion
  (clapping) when the phone is carried elsewhere on the body — see
  `eval/evaluation_report.md` for the measured per-activity agreement.
- Short controlled WISDM clips (≤3 minutes) say nothing about daily activity,
  longitudinal change, or depression — this is stated in every report's
  `limitations` field, not just here.
