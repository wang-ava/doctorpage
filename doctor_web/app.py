from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AuthenticationError, OpenAI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


APP_TITLE = "Doctor Language Bridge"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
TOKEN_ANALYTICS_MODEL = os.getenv("DOCTOR_WEB_TOKEN_ANALYTICS_MODEL", MODEL_NAME)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://127.0.0.1:8000")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", APP_TITLE)
OPENROUTER_PORTAL_URL = os.getenv("OPENROUTER_PORTAL_URL", "https://openrouter.ai/")
MAX_INPUT_CHARS = int(os.getenv("DOCTOR_WEB_MAX_INPUT_CHARS", "12000"))
MAX_IMAGE_COUNT = int(os.getenv("DOCTOR_WEB_MAX_IMAGE_COUNT", "4"))
TOP_LOGPROBS = int(os.getenv("DOCTOR_WEB_TOP_LOGPROBS", "5"))
REQUEST_TIMEOUT = float(os.getenv("DOCTOR_WEB_TIMEOUT", "240"))
PIPELINE_BIN_COUNT = int(os.getenv("DOCTOR_WEB_PIPELINE_BINS", "20"))
LOGIC_BREAK_DROP_THRESHOLD = float(os.getenv("DOCTOR_WEB_LOGIC_BREAK_DROP_THRESHOLD", "1.5"))
LOGIC_BREAK_Z_THRESHOLD = float(os.getenv("DOCTOR_WEB_LOGIC_BREAK_Z_THRESHOLD", "2.0"))
LOGIC_BREAK_ABS_LOGPROB_THRESHOLD = float(
    os.getenv("DOCTOR_WEB_LOGIC_BREAK_ABS_LOGPROB_THRESHOLD", "-2.5")
)
MUTATION_THRESHOLD = float(os.getenv("DOCTOR_WEB_MUTATION_THRESHOLD", "2.0"))
MUTATION_HIGH_THRESHOLD = float(os.getenv("DOCTOR_WEB_MUTATION_HIGH_THRESHOLD", "3.0"))
USER_API_KEY_HEADER = "x-openrouter-api-key"
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,120}$")
MODEL_SUGGESTIONS = [
    item.strip()
    for item in os.getenv("DOCTOR_WEB_MODEL_SUGGESTIONS", "").split(",")
    if item.strip()
]
if MODEL_NAME not in MODEL_SUGGESTIONS:
    MODEL_SUGGESTIONS.insert(0, MODEL_NAME)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STAGE_LABELS = {
    "translate_input": "Input Translation",
    "answer_in_english": "English Answer",
    "translate_answer_back": "Back-translation",
}

LANGUAGE_DETECTION_PROMPT = """You classify the language of clinician-written medical questions.
Decide whether the text is English.
Return JSON only with these keys:
- is_english: boolean
- language_name: string
- language_code: string
- rationale: string
"""

TRANSLATE_TO_ENGLISH_PROMPT = """Translate the following clinician question into precise medical English.
Preserve abbreviations, uncertainty, measurements, and formatting.
Output only the English translation.

Source language: {source_language}

Text:
{text}
"""

ANSWER_SYSTEM_PROMPT = """You are a senior multimodal clinical reasoning assistant for physicians.
You may receive a clinician question plus medical images.
Answer in English.
Structure the answer with:
1. Impression
2. Key Findings
3. Differential / Considerations
4. Recommended Next Steps
5. Risks / Red Flags
Be concise, clinically useful, and explicit about uncertainty.
Do not mention that you are following formatting instructions.
Do not expose chain-of-thought.
"""

BACK_TRANSLATION_PROMPT = """Translate the following English clinical answer into {target_language}.
Preserve medical terminology, structured sections, and uncertainty markers.
Output only the translated answer.

English answer:
{text}
"""


class UploadedImage(BaseModel):
    name: str = "image"
    media_type: str = Field(default="image/png")
    data_url: str


class ConsultationRequest(BaseModel):
    text: str
    images: list[UploadedImage] = Field(default_factory=list)
    model: str | None = None


def resolve_requested_model(model_name: str | None) -> str:
    cleaned = (model_name or "").strip()
    if not cleaned:
        return MODEL_NAME
    if not MODEL_ID_RE.fullmatch(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid OpenRouter model ID, for example openai/gpt-4o.",
        )
    return cleaned


def supports_token_analytics(model_name: str) -> bool:
    return model_name.strip() == TOKEN_ANALYTICS_MODEL


def is_usable_api_key(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    placeholder_markers = {
        "your-api-key-here",
        "your-openrouter-api-key",
        "your key here",
    }
    return cleaned.lower() not in placeholder_markers


def extract_api_key_from_request(request: Request) -> str:
    header_value = (request.headers.get(USER_API_KEY_HEADER) or "").strip()
    if is_usable_api_key(header_value):
        return header_value

    authorization = (request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()
        if is_usable_api_key(bearer_token):
            return bearer_token

    raise HTTPException(
        status_code=400,
        detail=(
            "Please provide your own OpenRouter API key before running an analysis. "
            f"Create an account at {OPENROUTER_PORTAL_URL} and paste the key into the access field."
        ),
    )


def get_client(api_key: str) -> OpenAI:
    if not is_usable_api_key(api_key):
        raise RuntimeError("A usable OpenRouter API key was not provided.")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        },
    )


def json_line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def normalize_stream_delta(content: Any) -> str:
    return normalize_message_content(content)


BYTE_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")


def repair_escaped_bytes(text: str) -> str:
    if not text or "\\x" not in text:
        return text

    def replace_match(match: re.Match[str]) -> str:
        raw = match.group(0)
        hex_bytes = raw.replace("\\x", "")
        try:
            return bytes.fromhex(hex_bytes).decode("utf-8")
        except UnicodeDecodeError:
            try:
                return bytes.fromhex(hex_bytes).decode("latin1")
            except Exception:
                return raw

    return BYTE_ESCAPE_RE.sub(replace_match, text)


def has_escaped_bytes(text: str) -> bool:
    return bool(text and BYTE_ESCAPE_RE.search(text))


def build_multimodal_user_content(question_text: str, images: list[UploadedImage]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Clinician question:\n"
                f"{question_text.strip()}\n\n"
                "If images are attached, integrate them into the answer."
            ),
        }
    ]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": image.data_url}})
    return parts


def safe_probability(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    bounded = max(min(logprob, 0.0), -60.0)
    return math.exp(bounded)


def safe_perplexity(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    bounded = max(min(-logprob, 60.0), -60.0)
    return math.exp(bounded)


def summarize_position_buckets(tokens_info: list[dict[str, Any]], bin_count: int = PIPELINE_BIN_COUNT) -> list[dict[str, Any]]:
    if not tokens_info:
        return []

    bins: list[list[dict[str, Any]]] = [[] for _ in range(bin_count)]
    total = len(tokens_info)
    denom = max(total - 1, 1)
    for index, token in enumerate(tokens_info):
        bin_index = min(int((index / denom) * bin_count), bin_count - 1)
        bins[bin_index].append(token)

    summary: list[dict[str, Any]] = []
    for bin_index, entries in enumerate(bins):
        valid_logprobs = [entry["logprob"] for entry in entries if entry.get("logprob") is not None]
        valid_probs = [entry["prob"] for entry in entries if entry.get("prob") is not None]
        valid_ppls = [entry["perplexity"] for entry in entries if entry.get("perplexity") is not None]
        summary.append(
            {
                "bin": bin_index,
                "token_count": len(entries),
                "start_ratio": bin_index / bin_count,
                "end_ratio": (bin_index + 1) / bin_count,
                "mean_logprob": sum(valid_logprobs) / len(valid_logprobs) if valid_logprobs else None,
                "mean_prob": sum(valid_probs) / len(valid_probs) if valid_probs else None,
                "mean_perplexity": sum(valid_ppls) / len(valid_ppls) if valid_ppls else None,
            }
        )
    return summary


def detect_logic_breaks(tokens_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tokens_info) < 3:
        return []

    valid_logprobs = [token["logprob"] for token in tokens_info if token.get("logprob") is not None]
    if not valid_logprobs:
        return []

    mu = sum(valid_logprobs) / len(valid_logprobs)
    variance = sum((value - mu) ** 2 for value in valid_logprobs) / max(len(valid_logprobs), 1)
    sigma = math.sqrt(variance) if variance > 1e-12 else 1.0

    events: list[dict[str, Any]] = []
    for index in range(1, len(tokens_info)):
        current = tokens_info[index]
        previous = tokens_info[index - 1]
        current_lp = current.get("logprob")
        previous_lp = previous.get("logprob")
        if current_lp is None or previous_lp is None:
            continue
        drop = float(current_lp - previous_lp)
        if drop > -LOGIC_BREAK_DROP_THRESHOLD:
            continue
        z_low = float((mu - current_lp) / sigma)
        if z_low >= LOGIC_BREAK_Z_THRESHOLD or current_lp <= LOGIC_BREAK_ABS_LOGPROB_THRESHOLD:
            events.append(
                {
                    "position": index,
                    "token": current.get("token", ""),
                    "logprob": current_lp,
                    "drop_from_prev": drop,
                    "z_low": z_low,
                }
            )
    return events


def detect_mutations(tokens_info: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tokens_info) < 2:
        return {
            "mutation_count": 0,
            "mutation_rate": 0.0,
            "high_magnitude_count": 0,
            "events": [],
        }

    events: list[dict[str, Any]] = []
    for index in range(1, len(tokens_info)):
        current = tokens_info[index]
        previous = tokens_info[index - 1]
        current_lp = current.get("logprob")
        previous_lp = previous.get("logprob")
        if current_lp is None or previous_lp is None:
            continue
        delta = float(current_lp - previous_lp)
        if abs(delta) < MUTATION_THRESHOLD:
            continue
        events.append(
            {
                "position": index,
                "token": current.get("token", ""),
                "delta": delta,
                "abs_delta": abs(delta),
                "direction": "drop" if delta < 0 else "jump",
                "high_magnitude": abs(delta) >= MUTATION_HIGH_THRESHOLD,
            }
        )

    return {
        "mutation_count": len(events),
        "mutation_rate": len(events) / max(len(tokens_info) - 1, 1),
        "high_magnitude_count": sum(1 for event in events if event["high_magnitude"]),
        "events": events,
    }


def summarize_token_slice(tokens_slice: list[dict[str, Any]]) -> dict[str, Any]:
    valid_probs = [token["prob"] for token in tokens_slice if token.get("prob") is not None]
    avg_prob = sum(valid_probs) / len(valid_probs) if valid_probs else None
    min_prob = min(valid_probs) if valid_probs else None
    low_confidence_count = sum(1 for value in valid_probs if value < 0.5)
    return {
        "token_count": len(tokens_slice),
        "avg_prob": avg_prob,
        "avg_prob_percent": f"{avg_prob * 100:.1f}%" if avg_prob is not None else "N/A",
        "min_prob": min_prob,
        "min_prob_percent": f"{min_prob * 100:.1f}%" if min_prob is not None else "N/A",
        "low_confidence_count": low_confidence_count,
        "low_confidence_ratio": (low_confidence_count / len(valid_probs)) if valid_probs else 0.0,
    }


def statement_confidence_label(summary: dict[str, Any]) -> tuple[str, str]:
    avg_prob = summary.get("avg_prob")
    min_prob = summary.get("min_prob")
    low_ratio = summary.get("low_confidence_ratio") or 0.0
    if avg_prob is None or min_prob is None:
        return ("Not available", "Detailed confidence scoring is unavailable for this statement.")
    if min_prob < 0.25 or low_ratio >= 0.25:
        return ("Review carefully", "Some wording in this statement was notably uncertain and should be checked manually.")
    if min_prob < 0.5 or avg_prob < 0.72 or low_ratio >= 0.1:
        return ("Review", "Most of the statement is stable, but one part may need a closer check.")
    return ("Stable", "This statement was comparatively stable in the model output.")


def collect_tokens_for_span(
    tokens_info: list[dict[str, Any]], token_spans: list[tuple[int, int]], start: int, end: int
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for token, (token_start, token_end) in zip(tokens_info, token_spans):
        if token_end <= start:
            continue
        if token_start >= end:
            break
        if token_end > start and token_start < end:
            collected.append(token)
    return collected


def build_display_token_groups(tokens_info: list[dict[str, Any]], raw_text: str) -> list[dict[str, Any]]:
    token_spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens_info:
        token_text = str(token.get("token") or "")
        start = cursor
        cursor += len(token_text)
        token_spans.append((start, cursor))

    groups: list[dict[str, Any]] = []
    for match in re.finditer(r"\S+\s*|\n", raw_text):
        segment_text = match.group(0)
        tokens_slice = collect_tokens_for_span(tokens_info, token_spans, match.start(), match.end())
        summary = summarize_token_slice(tokens_slice)
        groups.append(
            {
                "text": segment_text,
                "prob": summary["avg_prob"],
                "prob_percent": summary["avg_prob_percent"],
                "min_prob": summary["min_prob"],
                "min_prob_percent": summary["min_prob_percent"],
                "token_count": summary["token_count"],
            }
        )
    return groups


def build_statement_groups(tokens_info: list[dict[str, Any]], raw_text: str) -> list[dict[str, Any]]:
    token_spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens_info:
        token_text = str(token.get("token") or "")
        start = cursor
        cursor += len(token_text)
        token_spans.append((start, cursor))

    groups: list[dict[str, Any]] = []
    line_cursor = 0
    for raw_line in raw_text.splitlines(keepends=True):
        line_text = raw_line.rstrip("\n")
        line_start = line_cursor
        line_cursor += len(raw_line)
        if not line_text.strip():
            continue
        stripped_line = line_text.strip()
        if re.fullmatch(r"(?:#{1,6}\s+)?(?:\d+\.\s+)?[A-Za-z][A-Za-z /-]{0,40}", stripped_line):
            continue

        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", line_text):
            statement_text = match.group(0).strip()
            if len(re.sub(r"[^A-Za-z0-9]+", "", statement_text)) < 5:
                continue
            start = line_start + match.start()
            end = line_start + match.end()
            tokens_slice = collect_tokens_for_span(tokens_info, token_spans, start, end)
            if not tokens_slice:
                continue
            summary = summarize_token_slice(tokens_slice)
            label, note = statement_confidence_label(summary)
            groups.append(
                {
                    "text": statement_text,
                    "label": label,
                    "note": note,
                    "avg_prob": summary["avg_prob"],
                    "avg_prob_percent": summary["avg_prob_percent"],
                    "min_prob": summary["min_prob"],
                    "min_prob_percent": summary["min_prob_percent"],
                    "token_count": summary["token_count"],
                    "low_confidence_count": summary["low_confidence_count"],
                }
            )
    return groups[:12]


def analyze_logprobs(logprobs_data: Any, fallback_text: str) -> dict[str, Any]:
    cleaned_fallback_text = repair_escaped_bytes(fallback_text)
    if logprobs_data is None or getattr(logprobs_data, "content", None) is None:
        return {
            "available": False,
            "tokens_detail": [],
            "display_tokens": [],
            "statement_groups": [],
            "summary": {
                "total_tokens": 0,
                "avg_logprob": None,
                "avg_prob": None,
                "avg_prob_percent": "N/A",
                "perplexity": None,
                "low_confidence_count": 0,
                "low_confidence_ratio": 0.0,
                "logic_break_count": 0,
                "logic_break_rate": 0.0,
                "mutation_count": 0,
                "mutation_rate": 0.0,
            },
            "bucket_summary": [],
            "logic_breaks": [],
            "mutation_summary": {
                "mutation_count": 0,
                "mutation_rate": 0.0,
                "high_magnitude_count": 0,
                "events": [],
            },
            "text": cleaned_fallback_text,
            "token_text_suspicious": False,
        }

    tokens_info: list[dict[str, Any]] = []
    total_logprob = 0.0
    low_confidence_count = 0

    for token_data in logprobs_data.content:
        token = str(getattr(token_data, "token", ""))
        logprob = getattr(token_data, "logprob", None)
        probability = safe_probability(logprob)
        perplexity = safe_perplexity(logprob)
        if logprob is not None:
            total_logprob += float(logprob)
        if probability is not None and probability < 0.5:
            low_confidence_count += 1

        alternatives: list[dict[str, Any]] = []
        for alt in getattr(token_data, "top_logprobs", []) or []:
            alt_logprob = getattr(alt, "logprob", None)
            alt_prob = safe_probability(alt_logprob)
            alternatives.append(
                {
                    "token": str(getattr(alt, "token", "")),
                    "logprob": alt_logprob,
                    "prob": alt_prob,
                    "prob_percent": f"{(alt_prob or 0.0) * 100:.2f}%" if alt_prob is not None else "N/A",
                }
            )

        tokens_info.append(
            {
                "index": len(tokens_info),
                "token": token,
                "logprob": logprob,
                "prob": probability,
                "perplexity": perplexity,
                "prob_percent": f"{probability * 100:.2f}%" if probability is not None else "N/A",
                "top_alternatives": alternatives,
            }
        )

    total_tokens = len(tokens_info)
    avg_logprob = total_logprob / total_tokens if total_tokens else None
    avg_prob = safe_probability(avg_logprob) if avg_logprob is not None else None
    perplexity = math.exp(-total_logprob / total_tokens) if total_tokens else None
    low_ratio = low_confidence_count / total_tokens if total_tokens else 0.0
    logic_breaks = detect_logic_breaks(tokens_info)
    mutation_summary = detect_mutations(tokens_info)
    bucket_summary = summarize_position_buckets(tokens_info)

    raw_joined_text = "".join(token["token"] for token in tokens_info)
    repaired_joined_text = repair_escaped_bytes(raw_joined_text)
    token_text_suspicious = has_escaped_bytes(raw_joined_text)
    readable_text = cleaned_fallback_text.strip() or repaired_joined_text.strip()
    display_tokens = [] if token_text_suspicious else build_display_token_groups(tokens_info, raw_joined_text)
    statement_groups = [] if token_text_suspicious else build_statement_groups(tokens_info, raw_joined_text)

    return {
        "available": True,
        "tokens_detail": tokens_info,
        "display_tokens": display_tokens,
        "statement_groups": statement_groups,
        "summary": {
            "total_tokens": total_tokens,
            "avg_logprob": avg_logprob,
            "avg_prob": avg_prob,
            "avg_prob_percent": f"{avg_prob * 100:.2f}%" if avg_prob is not None else "N/A",
            "perplexity": perplexity,
            "low_confidence_count": low_confidence_count,
            "low_confidence_ratio": low_ratio,
            "logic_break_count": len(logic_breaks),
            "logic_break_rate": len(logic_breaks) / total_tokens if total_tokens else 0.0,
            "mutation_count": mutation_summary["mutation_count"],
            "mutation_rate": mutation_summary["mutation_rate"],
        },
        "bucket_summary": bucket_summary,
        "logic_breaks": logic_breaks,
        "mutation_summary": mutation_summary,
        "text": readable_text,
        "token_text_suspicious": token_text_suspicious,
    }


def generate_with_logprobs(
    client: OpenAI,
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    enable_token_analytics: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    request_args: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": REQUEST_TIMEOUT,
        "messages": messages,
    }
    if enable_token_analytics:
        request_args["logprobs"] = True
        request_args["top_logprobs"] = TOP_LOGPROBS
    response = client.chat.completions.create(**request_args)
    message = response.choices[0].message
    text = repair_escaped_bytes(normalize_message_content(message.content).strip())
    analysis = analyze_logprobs(getattr(response.choices[0], "logprobs", None), text)
    return {
        "text": text or analysis["text"].strip(),
        "analysis": analysis,
        "model": getattr(response, "model", model_name),
        "token_analytics_enabled": enable_token_analytics,
    }


def build_stage_report(stage: str, analysis: dict[str, Any], skipped: bool = False) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": STAGE_LABELS.get(stage, title_case_stage(stage)),
        "available": bool(analysis.get("available")),
        "skipped": skipped,
        "summary": analysis.get("summary", {}),
        "bucket_summary": analysis.get("bucket_summary", []),
        "logic_breaks": analysis.get("logic_breaks", []),
        "mutation_summary": analysis.get("mutation_summary", {}),
    }


def build_pipeline_analytics(stage_reports: list[dict[str, Any]]) -> dict[str, Any]:
    active_stages = [stage for stage in stage_reports if stage.get("available")]
    if not active_stages:
        return {
            "available": False,
            "stage_count": 0,
            "stages": stage_reports,
            "overall_summary": {},
            "flow_points": [],
            "heatmap_rows": [],
            "logic_break_examples": [],
            "mutation_by_stage": [],
        }

    weighted_logprob_sum = 0.0
    weighted_logprob_count = 0
    total_tokens = 0
    total_logic_breaks = 0
    total_mutations = 0
    flow_points: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    logic_break_examples: list[dict[str, Any]] = []
    mutation_by_stage: list[dict[str, Any]] = []
    stage_count = len(active_stages)

    for stage_index, stage in enumerate(active_stages):
        summary = stage.get("summary", {})
        total_tokens += int(summary.get("total_tokens") or 0)
        total_logic_breaks += int(summary.get("logic_break_count") or 0)
        total_mutations += int(summary.get("mutation_count") or 0)
        buckets = stage.get("bucket_summary", [])
        stage_logic_breaks = stage.get("logic_breaks", [])
        stage_label = stage["label"]
        stage_avg_logprob = summary.get("avg_logprob")
        stage_token_total = int(summary.get("total_tokens") or 0)
        if stage_avg_logprob is not None and stage_token_total:
            weighted_logprob_sum += float(stage_avg_logprob) * stage_token_total
            weighted_logprob_count += stage_token_total

        bin_count = max(len(buckets), 1)
        break_bins: list[int] = []
        stage_total_tokens = max(int(summary.get("total_tokens") or 0), 1)
        for event in stage_logic_breaks:
            event_copy = dict(event)
            event_copy["stage"] = stage["stage"]
            event_copy["stage_label"] = stage_label
            event_copy["global_position"] = (
                stage_index + (event.get("position", 0) / stage_total_tokens)
            ) / stage_count
            logic_break_examples.append(event_copy)
            break_bins.append(
                min(int((event.get("position", 0) / stage_total_tokens) * PIPELINE_BIN_COUNT), PIPELINE_BIN_COUNT - 1)
            )

        for bucket in buckets:
            flow_points.append(
                {
                    "stage": stage["stage"],
                    "label": stage_label,
                    "stage_index": stage_index,
                    "local_bin": bucket["bin"],
                    "global_position": (
                        stage_index + ((bucket["bin"] + 0.5) / bin_count)
                    ) / stage_count,
                    "mean_prob": bucket.get("mean_prob"),
                    "mean_perplexity": bucket.get("mean_perplexity"),
                    "mean_logprob": bucket.get("mean_logprob"),
                }
            )

        heatmap_rows.append(
            {
                "stage": stage["stage"],
                "label": stage_label,
                "bins": [bucket.get("mean_logprob") for bucket in buckets],
                "logic_break_bins": sorted(set(break_bins)),
            }
        )
        mutation_by_stage.append(
            {
                "stage": stage["stage"],
                "label": stage_label,
                "mutation_count": stage.get("mutation_summary", {}).get("mutation_count", 0),
                "mutation_rate": stage.get("mutation_summary", {}).get("mutation_rate", 0.0),
                "high_magnitude_count": stage.get("mutation_summary", {}).get("high_magnitude_count", 0),
            }
        )

    avg_stage_logprob = (
        weighted_logprob_sum / weighted_logprob_count if weighted_logprob_count else None
    )
    overall_perplexity = math.exp(-avg_stage_logprob) if avg_stage_logprob is not None else None

    logic_break_examples.sort(key=lambda item: (item.get("logprob", 0.0), item.get("drop_from_prev", 0.0)))

    return {
        "available": True,
        "stage_count": stage_count,
        "stages": stage_reports,
        "overall_summary": {
            "total_tokens": total_tokens,
            "avg_logprob": avg_stage_logprob,
            "avg_prob": safe_probability(avg_stage_logprob) if avg_stage_logprob is not None else None,
            "avg_prob_percent": (
                f"{safe_probability(avg_stage_logprob) * 100:.2f}%"
                if avg_stage_logprob is not None and safe_probability(avg_stage_logprob) is not None
                else "N/A"
            ),
            "overall_perplexity": overall_perplexity,
            "logic_break_count": total_logic_breaks,
            "logic_break_rate": total_logic_breaks / total_tokens if total_tokens else 0.0,
            "mutation_count": total_mutations,
            "mutation_rate": total_mutations / max(total_tokens - stage_count, 1),
        },
        "flow_points": flow_points,
        "heatmap_rows": heatmap_rows,
        "logic_break_examples": logic_break_examples[:12],
        "mutation_by_stage": mutation_by_stage,
    }


def detect_language(client: OpenAI, text: str, model_name: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=180,
        timeout=REQUEST_TIMEOUT,
        messages=[
            {"role": "system", "content": LANGUAGE_DETECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify the language of the following text. "
                    "Return JSON only.\n\n"
                    f"Text:\n{text}"
                ),
            },
        ],
    )
    raw = normalize_message_content(response.choices[0].message.content)
    payload = extract_json_object(raw)

    language_name = str(payload.get("language_name") or "English").strip() or "English"
    language_code = str(payload.get("language_code") or "en").strip().lower() or "en"
    rationale = str(payload.get("rationale") or "").strip()
    is_english = payload.get("is_english")
    if not isinstance(is_english, bool):
        is_english = language_code == "en" or language_name.lower() == "english"

    return {
        "is_english": is_english,
        "language_name": language_name,
        "language_code": language_code,
        "rationale": rationale,
        "raw": raw,
    }


def answer_with_logprobs(
    client: OpenAI,
    english_question: str,
    images: list[UploadedImage],
    model_name: str,
    enable_token_analytics: bool,
) -> dict[str, Any]:
    return generate_with_logprobs(
        client,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_multimodal_user_content(english_question, images),
            },
        ],
        model_name=model_name,
        enable_token_analytics=enable_token_analytics,
        temperature=0.2,
        max_tokens=1200,
    )


def translation_with_logprobs(
    client: OpenAI, text: str, prompt: str, *, model_name: str, enable_token_analytics: bool
) -> dict[str, Any]:
    return generate_with_logprobs(
        client,
        messages=[
            {
                "role": "system",
                "content": "You are a precise medical translator. Output only the translation.",
            },
            {"role": "user", "content": prompt.format(text=text)},
        ],
        model_name=model_name,
        enable_token_analytics=enable_token_analytics,
        temperature=0.1,
        max_tokens=1000,
    )


def validate_payload(payload: ConsultationRequest) -> None:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter a text question before submitting.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"Input text exceeds the limit of {MAX_INPUT_CHARS} characters.")
    if len(payload.images) > MAX_IMAGE_COUNT:
        raise HTTPException(status_code=400, detail=f"A maximum of {MAX_IMAGE_COUNT} images is supported.")

    for image in payload.images:
        if not image.data_url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail=f"Image {image.name} is not a valid data URL.")
        if ";base64," not in image.data_url:
            raise HTTPException(status_code=400, detail=f"Image {image.name} is missing base64 data.")


def build_translation_prompt(source_language: str) -> str:
    return TRANSLATE_TO_ENGLISH_PROMPT.format(source_language=source_language, text="{text}")


def build_back_translation_prompt(target_language: str) -> str:
    return BACK_TRANSLATION_PROMPT.format(target_language=target_language, text="{text}")


def title_case_stage(stage: str) -> str:
    return stage.replace("_", " ").title()


def ensure_non_empty_result(text: str, step_name: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError(f"{step_name} returned an empty result. Please retry or adjust the prompt.")
    return cleaned


def iter_fallback_tokens(text: str) -> Iterator[str]:
    for token in re.findall(r"\S+\s*|\n", text):
        yield token


app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
async def meta() -> JSONResponse:
    return JSONResponse(
        {
            "app_name": APP_TITLE,
            "model": MODEL_NAME,
            "default_model": MODEL_NAME,
            "token_analytics_model": TOKEN_ANALYTICS_MODEL,
            "model_suggestions": MODEL_SUGGESTIONS,
            "requires_user_api_key": True,
            "api_key_header": USER_API_KEY_HEADER,
            "openrouter_portal_url": OPENROUTER_PORTAL_URL,
            "max_input_chars": MAX_INPUT_CHARS,
            "max_image_count": MAX_IMAGE_COUNT,
            "top_logprobs": TOP_LOGPROBS,
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/consult/stream")
async def consult_stream(request: Request, payload: ConsultationRequest) -> StreamingResponse:
    validate_payload(payload)
    api_key = extract_api_key_from_request(request)
    model_name = resolve_requested_model(payload.model)
    token_analytics_enabled = supports_token_analytics(model_name)
    client = get_client(api_key)

    def event_stream() -> Iterator[str]:
        try:
            source_text = payload.text.strip()
            stage_reports: list[dict[str, Any]] = []

            yield json_line(
                {
                    "type": "status",
                    "stage": "detect_language",
                    "message": (
                        f"Checking whether the input is English with {model_name}."
                        if model_name
                        else "Checking whether the input is English."
                    ),
                }
            )

            detection = detect_language(client, source_text, model_name)
            yield json_line({"type": "detection", **detection})

            source_language = detection["language_name"] or "English"
            english_question = source_text

            if detection["is_english"]:
                stage_reports.append(build_stage_report("translate_input", {"available": False}, skipped=True))
                yield json_line(
                    {
                        "type": "translation_skipped",
                        "stage": "translate_input",
                        "message": "English input detected, skipping question translation.",
                        "english_text": source_text,
                    }
                )
            else:
                yield json_line(
                    {
                        "type": "status",
                        "stage": "translate_input",
                        "message": f"Translating the {source_language} question into English with {model_name}.",
                    }
                )
                prompt = build_translation_prompt(source_language)
                translated = translation_with_logprobs(
                    client,
                    source_text,
                    prompt,
                    model_name=model_name,
                    enable_token_analytics=token_analytics_enabled,
                )
                translated_analysis = translated["analysis"]
                english_question = ensure_non_empty_result(translated["text"], "Input translation")
                if translated_analysis.get("tokens_detail") and not translated_analysis.get("token_text_suspicious"):
                    for token in translated_analysis.get("tokens_detail", []):
                        yield json_line({"type": "translation_chunk", "text": token["token"]})
                else:
                    for piece in iter_fallback_tokens(english_question):
                        yield json_line({"type": "translation_chunk", "text": piece})

                stage_reports.append(build_stage_report("translate_input", translated_analysis))
                yield json_line(
                    {
                        "type": "translation_complete",
                        "stage": "translate_input",
                        "text": english_question,
                        "metrics": translated_analysis["summary"],
                    }
                )
                yield json_line(
                    {
                        "type": "pipeline_analytics",
                        "final": False,
                        "analytics": build_pipeline_analytics(stage_reports),
                    }
                )

            yield json_line(
                {
                    "type": "status",
                    "stage": "answer_in_english",
                    "message": (
                        f"Generating the English answer with {model_name} and {len(payload.images)} attached images."
                        if payload.images
                        else f"Generating the English answer with {model_name}."
                    ),
                }
            )

            answer = answer_with_logprobs(
                client,
                english_question,
                payload.images,
                model_name=model_name,
                enable_token_analytics=token_analytics_enabled,
            )
            english_answer = answer["text"]
            logprob_data = answer["analysis"]

            yield json_line(
                {
                    "type": "answer_ready",
                    "stage": "answer_in_english",
                    "message": (
                        "The model response is ready. Streaming the English answer and detailed confidence data."
                        if token_analytics_enabled
                        else "The model response is ready. Streaming the English answer."
                    ),
                    "model": answer["model"],
                    "logprobs_available": logprob_data["available"],
                    "token_analytics_enabled": token_analytics_enabled,
                }
            )

            stage_reports.append(build_stage_report("answer_in_english", logprob_data))
            if logprob_data["tokens_detail"] and not logprob_data.get("token_text_suspicious"):
                for index, token in enumerate(logprob_data["tokens_detail"]):
                    yield json_line(
                        {
                            "type": "answer_token",
                            "index": index,
                            **token,
                        }
                    )
            else:
                for index, token in enumerate(re.findall(r"\S+\s*|\n", english_answer)):
                    yield json_line(
                        {
                            "type": "answer_token",
                            "index": index,
                            "token": token,
                            "logprob": None,
                            "prob": None,
                            "prob_percent": "N/A",
                            "top_alternatives": [],
                    }
                )
            yield json_line(
                {
                    "type": "pipeline_analytics",
                    "final": False,
                    "analytics": build_pipeline_analytics(stage_reports),
                }
            )

            yield json_line(
                {
                    "type": "answer_complete",
                    "stage": "answer_in_english",
                    "text": english_answer,
                    "metrics": logprob_data["summary"],
                    "analysis": logprob_data,
                }
            )

            if detection["is_english"]:
                stage_reports.append(build_stage_report("translate_answer_back", {"available": False}, skipped=True))
                yield json_line(
                    {
                        "type": "back_translation_skipped",
                        "stage": "translate_answer_back",
                        "message": "The original language is English, so back-translation is skipped.",
                        "text": english_answer,
                    }
                )
            else:
                yield json_line(
                    {
                        "type": "status",
                        "stage": "translate_answer_back",
                        "message": f"Translating the English answer back into {source_language} with {model_name}.",
                    }
                )
                prompt = build_back_translation_prompt(source_language)
                translated_back_result = translation_with_logprobs(
                    client,
                    english_answer,
                    prompt,
                    model_name=model_name,
                    enable_token_analytics=token_analytics_enabled,
                )
                translated_back_analysis = translated_back_result["analysis"]
                translated_back = ensure_non_empty_result(translated_back_result["text"], "Back-translation")
                if translated_back_analysis.get("tokens_detail") and not translated_back_analysis.get("token_text_suspicious"):
                    for token in translated_back_analysis.get("tokens_detail", []):
                        yield json_line({"type": "back_translation_chunk", "text": token["token"]})
                else:
                    for piece in iter_fallback_tokens(translated_back):
                        yield json_line({"type": "back_translation_chunk", "text": piece})

                stage_reports.append(build_stage_report("translate_answer_back", translated_back_analysis))
                yield json_line(
                    {
                        "type": "back_translation_complete",
                        "stage": "translate_answer_back",
                        "text": translated_back,
                        "metrics": translated_back_analysis["summary"],
                    }
                )
                yield json_line(
                    {
                        "type": "pipeline_analytics",
                        "final": False,
                        "analytics": build_pipeline_analytics(stage_reports),
                    }
                )

            yield json_line(
                {
                    "type": "pipeline_analytics",
                    "final": True,
                    "analytics": build_pipeline_analytics(stage_reports),
                }
            )
            yield json_line({"type": "done", "message": "All stages completed."})
        except AuthenticationError:  # pragma: no cover
            yield json_line(
                {
                    "type": "error",
                    "message": (
                        "OpenRouter rejected the provided API key. "
                        "Please check that your account is active and that you pasted a valid key."
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover
            yield json_line({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
