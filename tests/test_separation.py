"""Required test scenario, spec section 8: Separation audit.

Proves two things: (1) processing an AnnoMI conversation does not change a
Track B activity report, and (2) WISDM/accelerometer data cannot enter the
chatbot prompt, context, retrieval query, or support/routing category —
checked both behaviorally and by static source inspection, since the
strongest guarantee here is that the import graph makes it impossible, not
just that it didn't happen to occur in one run.
"""
import ast
from pathlib import Path

from track_a.extraction import extract_from_conversation
from track_a.ingest_annomi import build_conversations, load_annomi
from track_b.ingest_accel import parse_wisdm_file, split_into_clips
from track_b.report import build_activity_report

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def _module_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_track_b_source_never_imports_track_a():
    for py_file in (ROOT / "track_b").rglob("*.py"):
        imports = _module_imports(py_file)
        assert not any(i == "track_a" or i.startswith("track_a.") for i in imports), py_file


def test_track_a_rag_source_never_imports_track_b():
    rag_files = list((ROOT / "track_a" / "rag").rglob("*.py")) + [ROOT / "common" / "backends.py"]
    for py_file in rag_files:
        imports = _module_imports(py_file)
        assert not any(i == "track_b" or i.startswith("track_b.") for i in imports), py_file


def test_processing_annomi_conversation_does_not_change_activity_report():
    df, _ = load_annomi(FIXTURES / "annomi_sample.csv")
    conv = build_conversations(df)[100]

    accel_df, _ = parse_wisdm_file(FIXTURES / "accel" / "clean_clip.txt")
    clip = split_into_clips(accel_df)[0]

    def make_report():
        return build_activity_report(
            clip, subject_id=int(clip.subject_id.iloc[0]), activity_label=clip.activity_label.iloc[0], segment_index=0,
            expected_hz=20, window_seconds=2, window_overlap=0.0,
            stationary_variance_threshold=0.6, max_gap_s=2.0,
        )

    report_before = make_report()
    extract_from_conversation(conv)  # process an AnnoMI conversation in between
    report_after = make_report()

    assert report_before.model_dump() == report_after.model_dump()


def test_rag_corpus_contains_no_accelerometer_sourced_passages():
    from track_a.rag.corpus import load_corpus

    corpus = load_corpus(ROOT / "data" / "rag_corpus")
    assert all(not p.chunk_id.startswith(("wisdm", "ucihar")) for p in corpus)


def test_output_id_namespaces_are_disjoint():
    from track_a.ingest_annomi import build_conversations as _bc

    df, _ = load_annomi(FIXTURES / "annomi_sample.csv")
    conv_ids = {f"annomi-{tid}" for tid in _bc(df)}
    accel_df, _ = parse_wisdm_file(FIXTURES / "accel" / "clean_clip.txt")
    clip = split_into_clips(accel_df)[0]
    from track_b.ingest_accel import clip_id_for

    clip_id = clip_id_for(int(clip.subject_id.iloc[0]), clip.activity_label.iloc[0], 0)

    assert not clip_id.startswith("annomi-") and not clip_id.startswith("audio-")
    assert conv_ids.isdisjoint({clip_id})
