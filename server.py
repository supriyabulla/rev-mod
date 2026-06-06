"""
server.py  (v3)
----------------
Changes from v2:
  • POST /api/flag        — save a flagged question
  • GET  /api/flags       — return all flags (for UI badge)
  • /api/process          — loads flags before generation, passes to mcq_generator
  • /api/answer           — stores question_hash in answer record
  • /api/history          — sort_order param: "newest" (default) | "oldest"
"""

import sys, os, time, threading
from pathlib import Path
from typing import Optional
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile, json

from llm_client import check_ollama_running, list_installed_models
from pdf_processor import extract_text_from_pdf
from concept_extractor import extract_concepts_from_sections
from mcq_generator import generate_questions_for_session, shuffle_options
from difficulty_engine import DifficultyState
from feedback_system import generate_feedback
from session_manager import (
    SessionState, generate_session_id,
    save_session_summary, save_study_sheet,
    build_session_summary, list_saved_sessions,
    load_session_detail, load_study_sheet,
)
from study_sheet_generator import generate_study_sheet
from question_flags import (
    save_flag, load_flags, get_flagged_hashes,
    get_flagged_topics, get_flag_summary,
)

app = FastAPI(title="Study Assistant", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_sessions: dict = {}
_pdf_tmp_files: dict = {}


class SessionData:
    def __init__(self):
        self.pdf_data = None
        self.knowledge = None
        self.questions = []
        self.difficulty_state: Optional[DifficultyState] = None
        self.session_state: Optional[SessionState] = None
        self.current_q_idx = 0
        self.q_start_time = None
        self.total_target = 10
        self.status = "idle"
        self.progress_message = ""
        self.progress_pct = 0
        self.error = None
        self._current_q = None


class StartSessionRequest(BaseModel):
    session_id: str
    difficulty: str = "medium"
    mode: str = "adaptive"
    num_questions: int = 10


class AnswerRequest(BaseModel):
    session_id: str
    answer_index: int


class FlagRequest(BaseModel):
    session_id: str
    question_text: str
    options: list
    topic: str
    section_title: str
    difficulty: str
    reason: str = "wrong_question"   # wrong_question | unclear | example_based | other


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


@app.get("/api/status")
async def get_status():
    ok, model = check_ollama_running()
    return {"ollama_running": ok, "model": model, "models": list_installed_models() if ok else []}


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    session_id = generate_session_id()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    content = await file.read()
    tmp.write(content); tmp.close()
    _pdf_tmp_files[session_id] = tmp.name
    sess = SessionData(); sess.status = "pdf_uploaded"
    _sessions[session_id] = sess
    return {"session_id": session_id, "filename": file.filename, "size_kb": len(content) // 1024}


@app.post("/api/process")
async def process_pdf(req: StartSessionRequest):
    if req.session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[req.session_id]
    pdf_path = _pdf_tmp_files.get(req.session_id)
    if not pdf_path:
        raise HTTPException(400, "No PDF uploaded")
    sess.total_target = req.num_questions
    sess.status = "processing"

    # Load flags ONCE before the thread starts (thread-safe read)
    flagged_hashes = get_flagged_hashes()
    flagged_topics = get_flagged_topics()

    def _process():
        try:
            sess.progress_message = "Extracting text from PDF..."
            sess.progress_pct = 10
            sess.pdf_data = extract_text_from_pdf(pdf_path)

            sess.progress_message = f"Analyzing {len(sess.pdf_data.sections)} sections with AI..."
            sess.progress_pct = 35
            sess.knowledge = extract_concepts_from_sections(sess.pdf_data.sections)

            sess.progress_message = "Generating questions..."
            sess.progress_pct = 70
            questions = generate_questions_for_session(
                knowledge=sess.knowledge,
                sections=sess.pdf_data.sections,
                difficulty=req.difficulty,
                count=req.num_questions + 8,
                flagged_hashes=flagged_hashes,     # v3: skip flagged
                flagged_topics=flagged_topics,      # v3: deprioritise flagged topics
            )
            if not questions:
                sess.error = "Failed to generate questions. Check Ollama is running."
                sess.status = "error"; return

            sess.questions = questions
            sess.session_state = SessionState(
                session_id=req.session_id, pdf_path=pdf_path,
                pdf_title=sess.pdf_data.title, start_time=time.time(),
                difficulty=req.difficulty, mode=req.mode,
            )
            sess.difficulty_state = DifficultyState(
                current_difficulty=req.difficulty, mode=req.mode,
            )
            sess.current_q_idx = 0
            sess.progress_pct = 100
            sess.progress_message = f"Ready! Generated {len(questions)} questions."
            sess.status = "ready"
        except Exception as e:
            sess.error = str(e); sess.status = "error"

    threading.Thread(target=_process, daemon=True).start()
    return {"ok": True}


@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[session_id]
    return {
        "status": sess.status, "message": sess.progress_message,
        "pct": sess.progress_pct, "error": sess.error,
        "pdf_title": sess.pdf_data.title if sess.pdf_data else None,
        "sections": len(sess.pdf_data.sections) if sess.pdf_data else 0,
        "questions_generated": len(sess.questions),
    }


@app.get("/api/question/{session_id}")
async def get_question(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[session_id]
    if sess.status not in ("ready", "quiz"):
        raise HTTPException(400, f"Session not ready (status: {sess.status})")
    if sess.current_q_idx >= len(sess.questions) or sess.current_q_idx >= sess.total_target:
        return {"done": True}

    sess.status = "quiz"
    q = shuffle_options(sess.questions[sess.current_q_idx])
    sess._current_q = q
    sess.q_start_time = time.time()

    clean_options = []
    for opt in q.options:
        text = opt
        for prefix in ["A) ", "B) ", "C) ", "D) "]:
            if text.startswith(prefix):
                text = text[len(prefix):]; break
        clean_options.append(text)

    ds = sess.difficulty_state
    return {
        "done": False,
        "question_num": sess.current_q_idx + 1,
        "total": sess.total_target,
        "question": q.question,
        "options": clean_options,
        "difficulty": q.difficulty,
        "question_type": q.question_type,
        "topic": q.topic,
        "question_hash": q.question_hash,   # v3: sent to frontend for flag button
        "section_title": q.section_title,
        "accuracy": round(ds.get_overall_accuracy() * 100, 1),
        "current_difficulty": ds.current_difficulty,
        "weak_topics": ds.get_weak_topics()[:4],
        "strong_topics": ds.get_strong_topics()[:4],
    }


@app.post("/api/answer")
async def submit_answer(req: AnswerRequest):
    if req.session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[req.session_id]
    q = sess._current_q
    elapsed = round(time.time() - (sess.q_start_time or time.time()), 1)
    feedback = generate_feedback(q, req.answer_index, elapsed, use_llm=True)
    change_msg = sess.difficulty_state.record_answer(
        question_num=sess.current_q_idx + 1, topic=q.topic,
        was_correct=feedback.is_correct, response_time=elapsed,
        section_title=q.section_title,
    )

    clean_opts = []
    for opt in q.options:
        text = opt
        for prefix in ["A) ", "B) ", "C) ", "D) "]:
            if text.startswith(prefix):
                text = text[len(prefix):]; break
        clean_opts.append(text)

    sess.session_state.answers.append({
        "question_num": sess.current_q_idx + 1,
        "question_text": q.question,
        "question_hash": q.question_hash,
        "options": clean_opts,
        "correct_index": q.correct_index,
        "user_index": req.answer_index,
        "is_correct": feedback.is_correct,
        "topic": q.topic,
        "difficulty": q.difficulty,
        "question_type": q.question_type,
        "response_time": elapsed,
        "explanation": feedback.explanation,
        "reinforcement": feedback.reinforcement,
        "is_followup": False,
        "is_flagged": False,   # may be set later via /api/flag
    })

    sess.current_q_idx += 1
    is_last = (sess.current_q_idx >= sess.total_target or
               sess.current_q_idx >= len(sess.questions))
    if is_last:
        sess.status = "done"

    return {
        "is_correct": feedback.is_correct,
        "correct_index": q.correct_index,
        "message": feedback.message,
        "explanation": feedback.explanation,
        "reinforcement": feedback.reinforcement,
        "elapsed": elapsed,
        "difficulty_change": change_msg,
        "is_last": is_last,
    }


# ── Flag endpoint (v3 NEW) ─────────────────────────────────────────────────────

@app.post("/api/flag")
async def flag_question(req: FlagRequest):
    """
    Flag a question as bad/wrong.
    Saves to ~/.study_assistant/flags.json.
    Future sessions will skip this question and deprioritise its topic.
    """
    record = save_flag(
        question_text=req.question_text,
        options=req.options,
        topic=req.topic,
        section_title=req.section_title,
        difficulty=req.difficulty,
        reason=req.reason,
        session_id=req.session_id,
    )
    # Mark the answer record as flagged if session is still active
    if req.session_id in _sessions:
        sess = _sessions[req.session_id]
        if sess.session_state:
            for ans in sess.session_state.answers:
                if ans.get("question_text") == req.question_text:
                    ans["is_flagged"] = True
    return {"ok": True, "question_hash": record.question_hash, "message": "Question flagged. It won't appear in future sessions."}


@app.get("/api/flags")
async def get_flags():
    """Returns flag summary for the UI."""
    return get_flag_summary()


# ── Summary ────────────────────────────────────────────────────────────────────

@app.get("/api/summary/{session_id}")
async def get_summary(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[session_id]
    if not sess.difficulty_state or not sess.session_state:
        raise HTTPException(400, "No session data")

    summary = build_session_summary(sess.session_state, sess.difficulty_state, time.time())
    save_session_summary(summary)

    def _gen_sheet():
        try:
            sheet = generate_study_sheet(
                session_id=session_id, pdf_title=summary.pdf_title,
                knowledge=sess.knowledge, question_log=summary.question_log,
                sections=sess.pdf_data.sections if sess.pdf_data else [],
            )
            save_study_sheet(sheet)
            summary.has_study_sheet = True
            save_session_summary(summary)
        except Exception as e:
            print(f"Study sheet generation failed: {e}")

    threading.Thread(target=_gen_sheet, daemon=True).start()

    return {
        "pdf_title": summary.pdf_title, "date": summary.date,
        "total_questions": summary.total_questions,
        "correct_answers": summary.correct_answers,
        "incorrect_answers": summary.incorrect_answers,
        "accuracy_percent": summary.accuracy_percent,
        "avg_response_time": summary.avg_response_time,
        "topic_accuracy": summary.topic_accuracy,
        "weak_topics": summary.weak_topics, "strong_topics": summary.strong_topics,
        "difficulty_history": summary.difficulty_history,
        "duration_minutes": summary.duration_minutes, "mode": summary.mode,
        "topic_categories": summary.topic_categories,
        "has_study_sheet": summary.has_study_sheet,
        "session_id": session_id,
    }


# ── History (v3: sort_order param) ────────────────────────────────────────────

@app.get("/api/history")
async def get_history(sort_order: str = Query("newest", enum=["newest", "oldest"])):
    """
    Returns session list.
    sort_order: "newest" (default) | "oldest"
    """
    sessions = list_saved_sessions()
    if sort_order == "oldest":
        sessions = list(reversed(sessions))
    return {"sessions": sessions[:50], "sort_order": sort_order}


@app.get("/api/history/{session_id}")
async def get_history_detail(session_id: str):
    detail = load_session_detail(session_id)
    if not detail:
        raise HTTPException(404, f"Session {session_id} not found")
    return detail


@app.get("/api/study-sheet/{session_id}")
async def get_study_sheet(session_id: str):
    sheet = load_study_sheet(session_id)
    if not sheet:
        raise HTTPException(404, "Study sheet not found")
    return asdict(sheet)
