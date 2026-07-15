import warnings
warnings.filterwarnings("ignore")

import re
import ast
import operator
import logging

import torch
from ddgs import DDGS
from transformers import pipeline
from rapidfuzz import process, fuzz

logger = logging.getLogger("qna.core")

# ============================================================
# Model Initialization
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Loading QA model on %s", device)

qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0 if device == "cuda" else -1,
)

# ============================================================
# Question Type Detection
# ============================================================
def is_math_question(q: str) -> bool:
    return bool(re.fullmatch(
        r"\s*-?\d+(?:\.\d+)?\s*[\+\-\*/]\s*-?\d+(?:\.\d+)?\s*=?\s*", q
    ))

def _num(node):
    """Extract a numeric value, handling unary minus (e.g. the -2 in '-2+3')."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -node.operand.value
    return node.value

def solve_math(expr: str) -> str:
    expr = expr.strip().rstrip("=").strip()
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv
    }
    node = ast.parse(expr, mode="eval").body
    left = _num(node.left)
    right = _num(node.right)
    try:
        return str(ops[type(node.op)](left, right))
    except ZeroDivisionError:
        return "Undefined (division by zero)"

def is_yes_no_claim(q: str) -> bool:
    return q.lower().startswith(("is ", "are ", "was ", "were ", "do ", "does "))

def expects_person(q: str) -> bool:
    return q.lower().startswith("who ")

# ============================================================
# Text Cleaning (fixes glued-word artifacts from scraped snippets)
# ============================================================
def normalize_text(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

_GLUED_LEADING = re.compile(r"^(is|was|are|the|a|an)(?=[A-Z])")
_GLUED_TRAILING = re.compile(r"(?<=[a-z])(is|was|are|has)$")

def clean_answer(answer: str) -> str:
    cleaned = answer.strip()
    cleaned = _GLUED_LEADING.sub("", cleaned).strip()
    match = _GLUED_TRAILING.search(cleaned)
    if match:
        candidate = cleaned[: match.start()].strip()
        if " " in candidate and len(candidate) >= 3:
            cleaned = candidate
    return cleaned.strip()

_LEADING_DATE_JUNK = re.compile(
    r"^\s*[A-Za-z0-9,\s]{0,25}?(ago|20\d{2})\s*-\s*", re.IGNORECASE
)

def clean_summary(text: str) -> str:
    return _LEADING_DATE_JUNK.sub("", text).strip()

def smart_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:-") + "..."

# ============================================================
# Role Detection & Explanation
# ============================================================
def detect_role(question: str):
    q = question.lower()
    if "governor" in q:
        return "Governor"
    if "chief minister" in q or " cm " in f" {q} ":
        return "Chief Minister"
    if "prime minister" in q or " pm " in f" {q} ":
        return "Prime Minister"
    return None

def extract_state(question: str):
    q = question.lower()
    if "andhra pradesh" in q or " ap" in f" {q} ":
        return "Andhra Pradesh"
    if "delhi" in q:
        return "Delhi"
    return ""

def generate_role_explanation(name: str, role: str, state: str):
    if role == "Governor":
        return (
            f"{name} is the Governor of {state}. "
            "The Governor is the constitutional head of the state and is "
            "appointed by the President of India."
        )
    if role == "Chief Minister":
        return (
            f"{name} is the Chief Minister of {state}. "
            "The Chief Minister is the elected head of the state government."
        )
    if role == "Prime Minister":
        return (
            f"{name} is the Prime Minister of India, "
            "serving as the head of the Union Government."
        )
    return f"{name} holds the position of {role}."

# ============================================================
# Answer Validation Heuristics
# ============================================================
GENERIC_ROLES = {"chief minister", "prime minister", "president", "governor", "pm", "cm"}

def is_generic_role_answer(answer: str, person_expected: bool) -> bool:
    return person_expected and answer.lower().strip() in GENERIC_ROLES

def looks_like_person(name: str) -> bool:
    words = name.split()
    return len(words) >= 2 and sum(w[0].isupper() for w in words if w) >= 2

def final_answer_is_valid(answer: str, person_expected: bool) -> bool:
    if not person_expected:
        return True
    if answer.lower().startswith("the "):
        return False
    return looks_like_person(answer)

def confidence_label(votes):
    if votes >= 5:
        return "Very High"
    if votes >= 3:
        return "High"
    if votes == 2:
        return "Medium"
    return "Low"

# ============================================================
# Fuzzy Consensus Engine
# ============================================================
def fuzzy_consensus(predictions):
    if not predictions:
        return None

    clusters = {}
    for p in predictions:
        m = process.extractOne(
            p["answer"], clusters.keys(),
            scorer=fuzz.token_set_ratio, score_cutoff=85
        )
        if m:
            clusters[m[0]].append(p)
        else:
            clusters[p["answer"]] = [p]

    best_key = max(clusters, key=lambda k: len(clusters[k]))
    cluster = clusters[best_key]
    best = max(cluster, key=lambda x: x["score"])

    return {
        "answer": best["answer"],
        "votes": len(cluster),
        "total": len(predictions),
        "summary": cluster[0]["context"],
        "sources": list({p["source"] for p in cluster})
    }

# ============================================================
# Web QA
# ============================================================
def web_qa(query, max_results: int = 10, timeout: int = 10):
    person_expected = expects_person(query)
    predictions = []

    try:
        with DDGS(timeout=timeout) as ddgs:
            results = list(ddgs.text(
                query,
                region="in-en",
                safesearch="Off",
                backend="lite",
                max_results=max_results
            ))
    except Exception:
        logger.exception("Web search failed")
        return None

    for r in results:
        raw_body = r.get("body", "").strip()
        if len(raw_body.split()) < 6:
            continue

        body = normalize_text(raw_body)

        out = qa_model(question=query, context=body)
        answer = clean_answer(out["answer"])

        if not answer:
            continue
        if is_generic_role_answer(answer, person_expected):
            continue

        score = out["score"]
        if person_expected and looks_like_person(answer):
            score *= 2.0

        predictions.append({
            "answer": answer,
            "score": score,
            "context": body,
            "source": r.get("href")
        })

    result = fuzzy_consensus(predictions)

    if not result:
        return None
    if not final_answer_is_valid(result["answer"], person_expected):
        return None

    return result

# ============================================================
# Main Brain
# ============================================================
def answer_question(query: str) -> dict:
    if is_math_question(query):
        return {
            "answer": solve_math(query),
            "confidence": "Certain",
            "status": "Verified"
        }

    if is_yes_no_claim(query):
        result = web_qa(query)
        if result:
            verdict = "No" if "not" in result["summary"].lower() else "Yes"
            return {
                "answer": verdict,
                "confidence": confidence_label(result["votes"]),
                "status": "Verified",
                "summary": smart_truncate(clean_summary(result["summary"]), 250),
                "sources": result["sources"]
            }

    result = web_qa(query)
    if not result:
        return {"answer": "No reliable answer found.", "status": "Unverified"}

    role = detect_role(query)
    state = extract_state(query)

    if role:
        summary = generate_role_explanation(result["answer"], role, state)
    else:
        summary = smart_truncate(clean_summary(result["summary"]), 300)

    return {
        "answer": result["answer"],
        "confidence": confidence_label(result["votes"]),
        "consensus": f"{result['votes']}/{result['total']}",
        "status": "Verified",
        "summary": summary,
        "sources": result["sources"]
    }
