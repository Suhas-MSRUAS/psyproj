# Fictional audio drop folder

Track A's ASR step (spec section A2) needs 6-10 short English WAV recordings of
**original, fictional** scripts plus their reference transcripts. There is no
real "organizer" for this assignment, so you supply these yourself:

- Record yourself (or a consenting friend) reading a short invented script, **or**
- Generate the audio with any TTS tool.

Either way you already know the exact text, so you can write an exact reference
transcript — that's what makes the WER evaluation meaningful.

## Expected layout

Drop one `.wav` and one same-named `.json` per case:

```
data/audio/
  case_001.wav
  case_001.json
  case_002.wav
  case_002.json
  ...
```

`case_001.json`:

```json
{
  "case_id": "case_001",
  "reference_transcript": "I have not been sleeping well for the past two weeks.",
  "speaker_consent": true,
  "notes": "fictional script, not a real patient"
}
```

`track_a/asr.py` reads every pair in this folder. Until you add real files, the
pipeline runs against the synthetic fixtures in `fixtures/audio/` instead, so
code and tests work offline before you record anything.
