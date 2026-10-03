"""RAG-grounded chatbot (spec section A3).

Retrieves from the approved corpus plus the current question only — never
from Track B accelerometer data, which this module does not import or
reference in any form. Abstains when no retrieved passage clears the
similarity threshold instead of inventing medical claims or citations.
"""
from __future__ import annotations

import re

from common.backends import LLMBackend
from common.schemas import ChatResponse
from track_a.rag.index import RAGIndex
from track_a.rag.safety_routing import classify

# Fixed, non-generative escalation text. Deliberately NOT produced by the LLM:
# immediate-risk and review cases must use a distinct, predictable supportive
# route rather than a free-form model judgement call.
_URGENT_MESSAGE = (
    "This sounds urgent. I'm not able to provide crisis support myself, but please "
    "reach out right now to a crisis line or emergency services in your area, or "
    "to someone you trust who can stay with you. You deserve immediate human help, "
    "not an automated response."
)
_REVIEW_MESSAGE = (
    "Thank you for sharing that — it sounds like a lot to carry. This has been "
    "flagged for a human (not this chatbot) to follow up with you. In the "
    "meantime, please consider reaching out to a counsellor, doctor, or someone "
    "you trust."
)


_NO_SOURCE_MESSAGE = (
    "I don't have a passage from the approved sources that answers this "
    "directly, so I won't guess. Please consult a clinician or a source "
    "like the WHO or NIMH depression pages."
)

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Requests to be assessed personally. Answered with fixed text, never by the
# model: the spec forbids classifying anyone as depressed.
_DIAGNOSIS_REQUEST_RE = re.compile(
    r"\bdiagnos(e|is|ing)\b.*\b(me|my)\b|\b(me|my)\b.*\bdiagnos"
    r"|\bdo i have (clinical |major )?depression\b|\bam i (clinically )?depressed\b"
    r"|\bdo i suffer from depression\b|\bis (this|it|that) depression\b"
    r"|\bdo you think i('m| am| have)\b.*\bdepress",
    re.IGNORECASE,
)
# Generated text that drifts into assessing the user despite the prompt.
_DIAGNOSTIC_OUTPUT_RE = re.compile(
    r"\byou (have|are suffering from|suffer from) (clinical |major )?depression\b"
    r"|\byou are (clinically )?depressed\b"
    r"|\byou (may|might|could|probably|likely) (be experiencing|have|be suffering from|be showing)"
    r"( (signs|symptoms) of)? (clinical |major )?depression\b",
    re.IGNORECASE,
)
_DIAGNOSIS_CHUNK = "medlineplus-depression-3"
_DIAGNOSIS_MESSAGE = (
    "I can't tell whether you or anyone else has depression. Only a qualified "
    "healthcare provider can assess that. MedlinePlus suggests speaking with a "
    "healthcare provider if symptoms last more than two weeks or get in the way of "
    f"daily life [{_DIAGNOSIS_CHUNK}]. I'm happy to explain general information "
    "about depression if that would help."
)


def _resolve_citations(answer: str, chunk_ids: list[str], extractive: bool) -> tuple[str, list[str]]:
    """Map the model's [n] markers onto real chunk ids, so every citation is
    verifiable against a retrieved chunk. Out-of-range numbers are dropped
    from the text rather than shown as citations. A generative model that
    cites nothing gets no citations; only an extractive backend (whose
    output is the passages themselves) has all its passages listed."""
    cited: list[str] = []

    def repl(m: re.Match) -> str:
        ids = []
        for n in (int(x) for x in m.group(1).split(",")):
            if 1 <= n <= len(chunk_ids):
                ids.append(chunk_ids[n - 1])
                if chunk_ids[n - 1] not in cited:
                    cited.append(chunk_ids[n - 1])
        return f"[{', '.join(ids)}]" if ids else ""

    answer = _CITE_RE.sub(repl, answer)
    if not cited and extractive:
        cited = list(chunk_ids)
    return answer, cited


def answer_question(
    conversation_or_case_id: str,
    question: str,
    index: RAGIndex,
    llm: LLMBackend,
    top_k: int,
    min_similarity: float,
) -> ChatResponse:
    routing = classify(question)

    if routing.category == "urgent":
        return ChatResponse(
            conversation_or_case_id=conversation_or_case_id,
            question=question,
            answer=_URGENT_MESSAGE,
            abstained=True,
            cited_passage_ids=[],
            routing=routing,
        )
    if routing.category == "review":
        return ChatResponse(
            conversation_or_case_id=conversation_or_case_id,
            question=question,
            answer=_REVIEW_MESSAGE,
            abstained=True,
            cited_passage_ids=[],
            routing=routing,
        )

    if _DIAGNOSIS_REQUEST_RE.search(question):
        return ChatResponse(
            conversation_or_case_id=conversation_or_case_id,
            question=question,
            answer=_DIAGNOSIS_MESSAGE,
            abstained=True,
            cited_passage_ids=[_DIAGNOSIS_CHUNK],
            routing=routing,
        )

    retrieved = index.search(question, top_k)
    relevant = [r for r in retrieved if r.similarity >= min_similarity]

    if not relevant:
        # Fixed text, never generated: with no supporting passage there is
        # nothing the model is allowed to say about medical facts.
        answer = _NO_SOURCE_MESSAGE
        cited_ids: list[str] = []
        abstained = True
    else:
        passages = [index.get_passage(r.chunk_id).text for r in relevant]
        answer, cited_ids = _resolve_citations(
            llm.generate(question, passages), [r.chunk_id for r in relevant], getattr(llm, "extractive", False)
        )
        abstained = False
        if _DIAGNOSTIC_OUTPUT_RE.search(answer):
            answer, cited_ids, abstained = _DIAGNOSIS_MESSAGE, [_DIAGNOSIS_CHUNK], True

    return ChatResponse(
        conversation_or_case_id=conversation_or_case_id,
        question=question,
        answer=answer,
        abstained=abstained,
        cited_passage_ids=cited_ids,
        routing=routing,
    )
