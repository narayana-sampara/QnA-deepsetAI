import warnings
warnings.filterwarnings("ignore")

import re
import ast
import operator
import logging
import time

from ddgs import DDGS
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from rapidfuzz import process, fuzz

from app.config import settings

logger = logging.getLogger("qna.core")

SEARCH_REGION = "in-en"
SEARCH_BACKEND = "lite"
SEARCH_OPTIONS = (
    (SEARCH_REGION, SEARCH_BACKEND),
    ("wt-wt", SEARCH_BACKEND),
    ("us-en", SEARCH_BACKEND),
)
SEARCH_ATTEMPTS = 2
SEARCH_TARGET_RESULTS = 6
SEARCH_RESULTS_PER_QUERY = 5
MIN_RELEVANT_TERM_MATCHES = 2

# ============================================================
# Model Initialization (Hugging Face Inference API — no local weights)
# ============================================================
QA_MODEL_NAME = settings.hf_qa_model
if not settings.hf_token:
    logger.warning("HF_TOKEN is not configured; Hugging Face Inference API calls will fail")
logger.info("Using HF Inference API QA model %s", QA_MODEL_NAME)

_hf_client = InferenceClient(
    provider="hf-inference",
    api_key=settings.hf_token,
)


def qa_model(question: str, context: str) -> dict:
    result = _hf_client.question_answering(
        question=question,
        context=context,
        model=QA_MODEL_NAME,
    )
    return {"answer": result.answer, "score": result.score}

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

def expects_winner(q: str) -> bool:
    return q.lower().strip().startswith("who won ")

def expects_person(q: str) -> bool:
    q = q.lower().strip()
    if not q.startswith("who "):
        return False

    # "Who won/owns/makes..." can validly resolve to a team, country, or
    # organization. Enforce person-shaped answers only for role/identity forms.
    if detect_role(q):
        return True
    return bool(re.match(r"^who\s+(is|was|are|were)\b", q))

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
# Search Query Rewriting & Relevance
# ============================================================
_QUESTION_MARK = re.compile(r"\?+$")
_STOP_TERMS = {
    "a", "an", "and", "are", "as", "at", "by", "current", "definition",
    "did", "do", "does", "for", "from", "how", "in", "is", "of", "on",
    "or", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "won",
}

def _clean_query_text(query: str) -> str:
    return _QUESTION_MARK.sub("", query).strip()

def _strip_leading_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)

def _add_unique(values: list[str], value: str) -> None:
    value = re.sub(r"\s+", " ", value).strip()
    if value and value.lower() not in {v.lower() for v in values}:
        values.append(value)

def build_search_queries(query: str) -> list[str]:
    cleaned = _clean_query_text(query)
    q = cleaned.lower()
    queries: list[str] = []

    capital_match = re.match(r"what\s+is\s+the\s+capital\s+of\s+(.+)$", q)
    if capital_match:
        place = cleaned[capital_match.start(1):].strip()
        _add_unique(queries, f"{place} capital city")
        _add_unique(queries, f"capital city of {place}")

    won_match = re.match(r"who\s+won\s+(?:the\s+)?(.+)$", q)
    if won_match:
        event = cleaned[won_match.start(1):].strip()
        _add_unique(queries, f"{event} winner")

    identity_match = re.match(r"who\s+(?:is|was|are|were)\s+(.+)$", q)
    if identity_match:
        subject = _strip_leading_article(cleaned[identity_match.start(1):])
        if any(role in q for role in ("prime minister", "chief minister", "governor", "president")):
            _add_unique(queries, f"current {subject}")
        _add_unique(queries, subject)

    definition_match = re.match(r"what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?(.+)$", q)
    if definition_match and not capital_match:
        subject = _strip_leading_article(cleaned[definition_match.start(1):])
        _add_unique(queries, f"{subject} definition")

    _add_unique(queries, cleaned)
    return queries

def _keyword_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP_TERMS
    }

def is_relevant_result(query: str, search_query: str, context: str) -> bool:
    terms = _keyword_terms(search_query) or _keyword_terms(query)
    if not terms:
        return True

    haystack = context.lower()
    matches = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", haystack))
    required = min(MIN_RELEVANT_TERM_MATCHES, len(terms))
    return matches >= required

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

def is_matchup_answer(answer: str) -> bool:
    return bool(re.search(r"\b(?:v|vs|versus)\.?\b", answer.lower()))

def is_truncated_answer(answer: str) -> bool:
    return "..." in answer or "…" in answer

def has_winner_evidence(answer: str, context: str) -> bool:
    answer_pattern = re.escape(answer.lower())
    for sentence in re.split(r"(?<=[.!?])\s+", context.lower()):
        if answer.lower() not in sentence:
            continue
        if re.search(
            rf"{answer_pattern}.{{0,100}}\b(won|wins|winner|winners|champion|champions|title|defeated|beat)\b",
            sentence,
        ):
            return True
        if re.search(
            rf"\b(winner|winners|champion|champions)\b.{{0,100}}{answer_pattern}",
            sentence,
        ):
            return True
    return False

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

    def cluster_rank(key):
        cluster = clusters[key]
        return (
            len(cluster),
            sum(p["score"] for p in cluster) / len(cluster),
            max(p["score"] for p in cluster),
        )

    best_key = max(clusters, key=cluster_rank)
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
def search_web(query: str, max_results: int, timeout: int):
    seen = set()
    results = []
    target_results = min(max_results, SEARCH_TARGET_RESULTS)
    request_size = min(max_results, SEARCH_RESULTS_PER_QUERY)

    for search_query in build_search_queries(query):
        for region, backend in SEARCH_OPTIONS:
            for attempt in range(SEARCH_ATTEMPTS):
                try:
                    with DDGS(timeout=timeout) as ddgs:
                        batch = list(ddgs.text(
                            search_query,
                            region=region,
                            safesearch="Off",
                            backend=backend,
                            max_results=request_size
                        ))
                except Exception:
                    logger.exception(
                        "Web search failed for query %r region=%s backend=%s",
                        search_query, region, backend,
                    )
                    batch = []

                for result in batch:
                    raw_title = result.get("title", "").strip()
                    raw_body = result.get("body", "").strip()
                    context = f"{raw_title}. {raw_body}" if raw_title else raw_body
                    if not is_relevant_result(query, search_query, normalize_text(context)):
                        continue

                    key = result.get("href") or result.get("title") or result.get("body")
                    if key in seen:
                        continue
                    seen.add(key)
                    result["_search_query"] = search_query
                    results.append(result)
                    if len(results) >= target_results:
                        return results

                if batch:
                    break

                if attempt + 1 < SEARCH_ATTEMPTS:
                    time.sleep(0.25)

    return results

def web_qa(query, max_results: int = 10, timeout: int = 10):
    person_expected = expects_person(query)
    winner_expected = expects_winner(query)
    predictions = []

    try:
        results = search_web(query, max_results=max_results, timeout=timeout)
    except Exception:
        logger.exception("Web search failed")
        return None

    for r in results:
        raw_title = r.get("title", "").strip()
        raw_body = r.get("body", "").strip()
        if len(raw_body.split()) < 6:
            continue

        title = normalize_text(raw_title)
        body = normalize_text(raw_body)
        context = f"{title}. {body}" if title else body
        search_query = r.get("_search_query", query)

        if not is_relevant_result(query, search_query, context):
            continue

        try:
            out = qa_model(question=query, context=context)
        except HfHubHTTPError:
            logger.exception("HF Inference API call failed for query %r", query)
            continue
        answer = clean_answer(out["answer"])

        if not answer:
            continue
        if is_truncated_answer(answer):
            continue
        if is_generic_role_answer(answer, person_expected):
            continue
        if winner_expected and is_matchup_answer(answer):
            continue
        if winner_expected and not has_winner_evidence(answer, context):
            continue

        score = out["score"]
        if person_expected and looks_like_person(answer):
            score *= 2.0

        predictions.append({
            "answer": answer,
            "score": score,
            "context": context,
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
