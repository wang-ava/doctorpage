#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-concurrency stability evaluation across 4 languages (EN, ZH, MS, TH).
Processes pre-existing multi-turn medical conversations: same messages, 3 rounds per case.
Output: Healthbench/result/response/{model_name}/{Language}/round{1-3}.json

Checkpoint/Resume: Saves progress periodically and on exit. Re-run without --no-resume
to skip already-completed (prompt_id, language, round) tasks. Use --no-resume to start
fresh and overwrite outputs.

Per-request metrics: original_input, reasoning_tokens (if applicable), output_tokens,
total_tokens, response_time.
"""

import os
import re
import json
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
import requests

# Save checkpoint every N completed tasks (merge existing + new, then write)
CHECKPOINT_INTERVAL = 25

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------------------------------

BASE_URL_API = "https://openrouter.ai/api/v1"
BASE_URL_LOCAL = "http://localhost:8000/v1"
TEMPERATURE = 0.2
TIMEOUT = 180 # Increased timeout for local models which may be slower
MAX_WORKERS = 10  # Reduced default workers for local models to avoid overwhelming the server
NUM_ROUNDS = 3
LANGUAGES = ["EN", "ZH", "MS", "TH"]

# Language code -> (translation filename suffix, messages key)
# EN uses English dataset "prompt"; others use {lang}_translation.jsonl "translation"
LANG_CONFIG = {
    "EN": ("english_only", "prompt"),  # special: from English sample
    "ZH": ("chinese_translation", "translation"),
    "MS": ("malay_translation", "translation"),
    "TH": ("thai_translation", "translation"),
}


def sanitize_model_name(model: str) -> str:
    """Replace invalid path characters and whitespace in model name."""
    s = model.strip()
    # 1) 将不适合作为路径名的字符替换为 "_"
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    # 2) 将任意空白（空格/Tab/换行等）替换为 "_"
    s = re.sub(r"\s+", "_", s)
    # 3) 压缩连续 "_"，并去掉首尾 "_"
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _read_json_objects(path: Path):
    """Yield JSON objects from a file that may be line-delimited or pretty-printed."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    start = 0
    n = len(text)
    while start < n:
        i = text.find("{", start)
        if i < 0:
            break
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[i : j + 1]
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
                    start = j + 1
                    break
            j += 1
        else:
            break


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load line-delimited or pretty-printed JSON objects from path."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass
    if not out:
        out = list(_read_json_objects(path))
    return out


def load_english_cases(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load English sample; return dict prompt_id -> messages (field 'prompt')."""
    rows = load_jsonl(path)
    by_id = {}
    for r in rows:
        pid = r.get("prompt_id")
        if not pid:
            continue
        msg = r.get("prompt")
        if not msg or not isinstance(msg, list):
            continue
        by_id[pid] = msg
    return by_id


def load_translation_by_id(path: Path, messages_key: str) -> Dict[str, List[Dict[str, str]]]:
    """Load translation JSONL; return dict prompt_id -> messages (e.g. 'translation')."""
    rows = load_jsonl(path)
    by_id = {}
    for r in rows:
        pid = r.get("prompt_id")
        if not pid:
            continue
        msg = r.get(messages_key)
        if not msg or not isinstance(msg, list):
            continue
        by_id[pid] = msg
    return by_id


def build_aligned_cases(
    base_dir: Path,
    english_path: Path,
    translation_dir: Path,
) -> Tuple[List[str], Dict[str, Dict[str, List[Dict[str, str]]]]]:
    """
    Align 100 cases by prompt_id across EN + ZH, MS, TH.
    Returns (ordered_ids, id -> {lang -> messages}).
    Only includes cases present in all four sources.
    """
    en_path = base_dir / english_path if not english_path.is_absolute() else Path(english_path)
    trans_dir = base_dir / translation_dir if not translation_dir.is_absolute() else Path(translation_dir)

    en_by_id = load_english_cases(en_path)
    ordered_ids = list(en_by_id.keys())
    if not ordered_ids:
        raise FileNotFoundError(f"No English cases with prompt_id in {en_path}")

    # Load translation files for ZH, MS, TH
    trans_by_lang = {}
    for lang in ["ZH", "MS", "TH"]:
        if lang == "ZH":
            p = trans_dir / "chinese_translation.jsonl"
        elif lang == "MS":
            p = trans_dir / "malay_translation.jsonl"
        else:
            p = trans_dir / "thai_translation.jsonl"
        if not p.exists():
            logger.warning("Translation file not found: %s", p)
            trans_by_lang[lang] = {}
            continue
        trans_by_lang[lang] = load_translation_by_id(p, "translation")

    # Align: only ids that have EN and all requested non-EN languages
    aligned = {}
    for pid in ordered_ids:
        entry = {"EN": en_by_id[pid]}
        ok = True
        for lang in ["ZH", "MS", "TH"]:
            if pid not in trans_by_lang.get(lang, {}):
                ok = False
                break
            entry[lang] = trans_by_lang[lang][pid]
        if ok:
            aligned[pid] = entry

    # Preserve order
    ordered = [i for i in ordered_ids if i in aligned]
    return ordered, aligned


def _coerce_usage(usage_obj: Any) -> Dict[str, int]:
    """
    Normalize API usage to dict: input_tokens, output_tokens, total_tokens, reasoning_tokens.
    Handles OpenAI/OpenRouter shapes and completion_tokens_details.reasoning_tokens.
    """
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    def get_reasoning(d: Any) -> int:
        if d is None:
            return 0
        if isinstance(d, dict):
            return int(d.get("reasoning_tokens") or 0)
        return int(getattr(d, "reasoning_tokens", 0) or 0)

    if isinstance(usage_obj, dict):
        it = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
        ot = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
        tt = usage_obj.get("total_tokens") or (it + ot)
        det = usage_obj.get("completion_tokens_details") or {}
        return {"input_tokens": int(it or 0), "output_tokens": int(ot or 0), "total_tokens": int(tt or 0), "reasoning_tokens": get_reasoning(det)}

    try:
        ud = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "completion_tokens_details": getattr(usage_obj, "completion_tokens_details", None),
        }
        it = ud.get("prompt_tokens") or ud.get("input_tokens") or 0
        ot = ud.get("completion_tokens") or ud.get("output_tokens") or 0
        tt = ud.get("total_tokens") or (it + ot)
        det = ud.get("completion_tokens_details")
        return {"input_tokens": int(it or 0), "output_tokens": int(ot or 0), "total_tokens": int(tt or 0), "reasoning_tokens": get_reasoning(det)}
    except Exception:
        it = int(getattr(usage_obj, "prompt_tokens", 0) or getattr(usage_obj, "input_tokens", 0) or 0)
        ot = int(getattr(usage_obj, "completion_tokens", 0) or getattr(usage_obj, "output_tokens", 0) or 0)
        tt = int(getattr(usage_obj, "total_tokens", 0) or (it + ot))
        det = getattr(usage_obj, "completion_tokens_details", None)
        return {"input_tokens": it, "output_tokens": ot, "total_tokens": tt, "reasoning_tokens": get_reasoning(det)}


def run_one_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    timeout: float,
    max_retries: int = 3,
) -> Tuple[str, Optional[str], Dict[str, int], float]:
    """
    Return (response_text, error, usage_dict, response_time_sec).
    On failure: ("", error_msg, zeroed usage, elapsed time).
    """
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            if not resp or not getattr(resp, "choices", None):
                raise ValueError("Empty API response")
            content = (resp.choices[0].message.content or "").strip()
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
            usage = _coerce_usage(raw.get("usage") or getattr(resp, "usage", None))
            return content, None, usage, elapsed
        except Exception as e:
            elapsed = time.perf_counter() - t0
            # Extract more detailed error information
            err = str(e)
            err_type = type(e).__name__
            
            # Try to get more detailed error information from OpenAI exceptions
            if hasattr(e, 'message'):
                err = str(e.message) if e.message else err
            elif hasattr(e, 'body'):
                try:
                    if isinstance(e.body, dict):
                        err_detail = e.body.get('error', {}).get('message', '')
                        if err_detail:
                            err = f"{err} | {err_detail}"
                except:
                    pass
            
            # Try to get cause information
            if hasattr(e, '__cause__') and e.__cause__:
                cause_str = str(e.__cause__)
                if cause_str and cause_str not in err:
                    err = f"{err} (cause: {cause_str})"
            
            # Build comprehensive error message
            full_err_msg = f"{err_type}: {err}"
            
            # Log more details for connection errors - but only on first attempt to avoid spam
            is_connection_error = (
                "Connection" in err or 
                "connection" in err.lower() or 
                "refused" in err.lower() or 
                "timeout" in err.lower() or
                "unreachable" in err.lower() or
                "network" in err.lower() or
                err_type == "APIConnectionError" or
                err_type == "TimeoutError" or
                err_type == "ConnectionError"
            )
            
            if is_connection_error:
                if attempt == 0:  # Only log on first attempt
                    logger.warning("Connection error on attempt %s/%s: %s", attempt + 1, max_retries, full_err_msg)
                    logger.warning("Server: %s, Model: %s", client.base_url if hasattr(client, 'base_url') else "localhost:8000", model)
            else:
                if attempt == 0:  # Only log on first attempt
                    logger.warning("API error on attempt %s/%s: %s", attempt + 1, max_retries, full_err_msg)
            
            if attempt + 1 >= max_retries:
                # On final failure, return the full error message
                return "", full_err_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}, elapsed
            
            # Exponential backoff with jitter
            wait_time = 1.5 ** (attempt + 1) + (attempt * 0.1)
            if attempt == max_retries - 2:  # Last retry, wait a bit longer
                wait_time *= 1.5
            time.sleep(wait_time)
    return "", "max_retries exceeded", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}, 0.0


def process_task(
    task: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Run one (prompt_id, language, round) completion. Returns record for output."""
    prompt_id = task["prompt_id"]
    lang = task["language"]
    round_idx = task["round"]
    messages = task["messages"]

    text, err, usage, response_time = run_one_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    rec: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "language": lang,
        "round": round_idx,
        "response": text,
        "error": err,
        "original_input": messages,
        "reasoning_tokens": usage["reasoning_tokens"] or None,
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "response_time": round(response_time, 4),
    }
    if rec["reasoning_tokens"] is None or rec["reasoning_tokens"] == 0:
        rec["reasoning_tokens"] = None
    return rec


def load_checkpoint(
    out_root: Path,
    languages: List[str],
    num_rounds: int,
) -> Tuple[Set[Tuple[str, str, int]], Dict[str, Dict[int, List[Dict[str, Any]]]]]:
    """
    Load existing round{N}.json files from out_root.
    Returns (completed_set, existing_results).
    completed_set = {(prompt_id, language, round), ...}
    existing_results[lang][round] = list of records.
    """
    completed: Set[Tuple[str, str, int]] = set()
    existing_results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        lang: {r: [] for r in range(1, num_rounds + 1)} for lang in languages
    }
    for lang in languages:
        lang_dir = out_root / lang
        if not lang_dir.is_dir():
            continue
        for r in range(1, num_rounds + 1):
            path = lang_dir / f"round{r}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load checkpoint %s: %s", path, e)
                continue
            if not isinstance(data, list):
                logger.warning("Invalid checkpoint format in %s: expected list", path)
                continue
            for rec in data:
                pid = rec.get("prompt_id")
                if not pid:
                    continue
                key = (pid, lang, r)
                # Only mark as completed if successful (no error)
                # Failed tasks will be retried on next run
                if not rec.get("error"):
                    completed.add(key)
                    existing_results[lang][r].append(rec)
                else:
                    logger.info("Found failed task to retry: %s/%s/round%s", pid[:8], lang, r)
    return completed, existing_results


def save_checkpoint(
    out_root: Path,
    languages: List[str],
    num_rounds: int,
    existing_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    results_by_lang_round: Dict[str, Dict[int, List[Dict[str, Any]]]],
) -> None:
    """Merge existing + new results and write round{N}.json for each (lang, round)."""
    out_root.mkdir(parents=True, exist_ok=True)
    for lang in languages:
        lang_dir = out_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for r in range(1, num_rounds + 1):
            existing = existing_results.get(lang, {}).get(r, [])
            new_rows = results_by_lang_round.get(lang, {}).get(r, [])
            rows = existing + new_rows
            path = lang_dir / f"round{r}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            logger.info("Checkpoint: wrote %s (%s rows)", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-A-P stability evaluation across EN, ZH, MS, TH (3 rounds per case)."
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="local",
        help="api = OpenRouter; local = local model (default: local)",
    )
    parser.add_argument(
        "--model",
        help="Model name",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL (default: OpenRouter for api, localhost:8000 for local)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: OPENROUTER_API_KEY or OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory containing dataset/ and result/ (default: parent of script's dir)",
    )
    parser.add_argument(
        "--english",
        type=Path,
        default=Path("dataset/hard_2025-05-08-21-00-10_english_only_sample_100.jsonl"),
        help="Path to English sample JSONL (relative to base-dir)",
    )
    parser.add_argument(
        "--translation-dir",
        type=Path,
        default=Path("result/translate/google_gemini-3-pro-preview"),
        help="Directory containing {chinese, malay, thai}_translation.jsonl (relative to base-dir)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="ThreadPoolExecutor max workers",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help="Number of stability rounds (1–3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases (for testing)",
    )
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=None,
        help="Limit number of cases per language (for testing, e.g., 10 for 10 cases per language)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Languages to run (default: EN ZH MS TH)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint/resume; start fresh and overwrite existing outputs",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help="Save checkpoint every N completed tasks (default: %(default)s)",
    )
    args = parser.parse_args()

    base_dir = (args.base_dir or Path(__file__).resolve().parent.parent).resolve()

    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if args.mode == "api" and not api_key:
        raise ValueError("For --mode api, set OPENROUTER_API_KEY or OPENAI_API_KEY or --api-key")

    base_url = args.base_url
    if base_url is None:
        base_url = BASE_URL_API if args.mode == "api" else BASE_URL_LOCAL

    model = args.model

    # If user didn't override --model and is using api mode, switch to a sensible api default.
    if args.mode == "api" and args.model is None:
        args.model = "google/gemini-3-pro-preview"
    
    if args.mode == "local" and args.model is None:
        raise ValueError("For --mode local, please specify --model (e.g.,  --model \"OpenGVLab/InternVL2-8B\")")

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
    temp = args.temperature
    timeout = args.timeout

    # For local mode, limit max_workers to avoid overwhelming the server
    if args.mode == "local":
        max_workers = min(args.max_workers, 15)
        if timeout < 60:
            logger.warning("Timeout is %s seconds. For local models, consider using at least 60-120 seconds.", timeout)
    else:
        max_workers = min(args.max_workers, 25)
    
    if not isinstance(args.rounds, int) or args.rounds < 1:
        raise ValueError("--rounds must be a positive integer")
    num_rounds = max(1, min(3, args.rounds))

    languages = [l.upper() for l in args.languages]

    logger.info("base_dir=%s mode=%s model=%s base_url=%s workers=%s rounds=%s langs=%s",
                base_dir, args.mode, model, base_url, max_workers, num_rounds, languages)
    
    # Check server connectivity for local mode
    if args.mode == "local":
        try:
            # Try to connect to the server
            test_url = base_url + '/models'
            logger.info("Checking server connectivity at %s...", test_url)
            resp = requests.get(test_url, timeout=5)
            if resp.status_code == 200:
                logger.info("✓ Local server is accessible at %s", base_url)
                # Try to list available models
                try:
                    models_data = resp.json()
                    """
                    {'object': 'list', 'data': [{'id': 'OpenGVLab/InternVL2-8B', 'object': 'model', 'created': 1769600653, 'owned_by': 'vllm', 'root': 'OpenGVLab/InternVL2-8B', 'parent': None, 'max_model_len': 65536, 'permission': [{'id': 'modelperm-a9b0760e5019f601', 'object': 'model_permission', 'created': 1769600653, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]}
                    """
                    if 'data' in models_data:
                        model_ids = [m.get('id', 'unknown') for m in models_data.get('data', [])]
                        logger.info("Available models: %s", ', '.join(model_ids) if model_ids else 'none')
                except:
                    pass
            else:
                logger.error("✗ Local server returned status %s at %s", resp.status_code, base_url)
                raise ConnectionError(f"Server returned status {resp.status_code}")
        except requests.exceptions.ConnectionError as e:
            logger.error("=" * 60)
            logger.error("✗ Cannot connect to local server at %s", base_url)
            logger.error("")
            logger.error("The server is not running or not accessible.")
            logger.error("")
            logger.error("To fix this, start your model server first:")
            logger.error("  Example with vLLM (using GPU 2,3):")
            logger.error("    CUDA_VISIBLE_DEVICES=2,3 vllm serve OpenGVLab/InternVL2-8B --port 8000")
            logger.error("  Example with llama.cpp: ./server -m model.gguf --port 8000")
            logger.error("  Or use your OpenAI-compatible server")
            logger.error("")
            logger.error("Note: To use specific GPUs (e.g., 2,3), set CUDA_VISIBLE_DEVICES=2,3")
            logger.error("      before starting the server.")
            logger.error("")
            logger.error("Then run this script again.")
            logger.error("=" * 60)
            raise ConnectionError(f"Local server not accessible at {base_url}. Please start your model server first.") from e
        except Exception as e:
            logger.error("Could not verify server connectivity: %s", e)
            raise ConnectionError(f"Server check failed: {e}") from e

    out_root = base_dir / "result" / "response" / sanitize_model_name(model)
    checkpoint_interval = max(1, args.checkpoint_interval)
    completed: Set[Tuple[str, str, int]] = set()
    existing_results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        lang: {r: [] for r in range(1, num_rounds + 1)} for lang in languages
    }
    """
    {'EN': {1: [], 2: [], 3: []}, 'ZH': {1: [], 2: [], 3: []}, 'MS': {1: [], 2: [], 3: []}, 'TH': {1: [], 2: [], 3: []}}
    """
    if not args.no_resume:
        completed, existing_results = load_checkpoint(out_root, languages, num_rounds)
        if completed:
            logger.info("Resume: loaded %s completed (prompt_id, lang, round) from checkpoint", len(completed))

    ordered_ids, aligned = build_aligned_cases(
        base_dir,
        args.english,
        args.translation_dir,
    )

    if args.limit is not None and args.limit > 0:
        ordered_ids = ordered_ids[:args.limit]
        aligned = {k: v for k, v in aligned.items() if k in ordered_ids}

    logger.info("Aligned %s cases across EN + ZH/MS/TH", len(ordered_ids))

    # Apply limit per language if specified
    if args.limit_per_language is not None and args.limit_per_language > 0:
        # For each language, collect prompt_ids that have that language
        lang_pids: Dict[str, List[str]] = {lang: [] for lang in languages}
        for pid in ordered_ids:
            for lang in languages:
                lang_pids[lang].append(pid)
        
        # Limit each language to N prompt_ids
        limited_pids = set()
        for lang in languages:
            limited = lang_pids[lang][:args.limit_per_language]
            limited_pids.update(limited)
            logger.info("Language %s: limited to %s cases (from %s available)", 
                       lang, len(limited), len(lang_pids[lang]))
        
        # Filter ordered_ids and aligned to only include limited prompt_ids
        ordered_ids = [pid for pid in ordered_ids if pid in limited_pids]
        aligned = {k: v for k, v in aligned.items() if k in limited_pids}
        logger.info("After per-language limit: %s total cases", len(ordered_ids))

    # Build all tasks: (prompt_id, language, round) -> messages
    tasks_all: List[Dict[str, Any]] = []
    for pid in ordered_ids:
        for lang in languages:
            messages = aligned[pid][lang]
            for r in range(1, num_rounds + 1):
                tasks_all.append({
                    "prompt_id": pid,
                    "language": lang,
                    "round": r,
                    "messages": messages,
                })
    tasks = [t for t in tasks_all if (t["prompt_id"], t["language"], t["round"]) not in completed]
    logger.info("Total tasks: %s (%s to run, %s skipped)", len(tasks_all), len(tasks), len(tasks_all) - len(tasks))

    # Final server check before starting tasks (for local mode)
    if args.mode == "local" and tasks:
        if requests is not None:
            try:
                test_url = base_url.rstrip('/v1') + '/v1/models'
                resp = requests.get(test_url, timeout=3)
                if resp.status_code == 200:
                    logger.info("✓ Server is ready. Starting task processing...")
                else:
                    logger.error("✗ Server returned status %s. Please check your server.", resp.status_code)
            except Exception as e:
                logger.error("=" * 60)
                logger.error("✗ Cannot connect to server before starting tasks!")
                logger.error("Error: %s", str(e))
                logger.error("")
                logger.error("Please ensure your server is running:")
                logger.error("  CUDA_VISIBLE_DEVICES=2,3 vllm serve models/internvl2-8b --port 8000")
                logger.error("")
                logger.error("Then run this script again.")
                logger.error("=" * 60)
                raise ConnectionError(f"Server not accessible: {e}") from e

    if not tasks:
        logger.info("Nothing to run. Use --no-resume to start fresh.")
        save_checkpoint(out_root, languages, num_rounds, existing_results, {
            lang: {r: [] for r in range(1, num_rounds + 1)} for lang in languages
        })
        logger.info("Done. Results under %s", out_root)
        return

    # Collect new results by (language, round)
    results_by_lang_round: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        lang: {r: [] for r in range(1, num_rounds + 1)} for lang in languages
    }
    lock = threading.Lock()
    done_count = [0]

    def run_and_store(t: Dict[str, Any]) -> None:
        rec = process_task(t, client, model, temp, timeout)
        with lock:
            results_by_lang_round[rec["language"]][rec["round"]].append(rec)
            done_count[0] += 1
            n = done_count[0]
            # Log errors for debugging
            if rec.get("error"):
                logger.warning("Task failed: prompt_id=%s, lang=%s, round=%s, error=%s", 
                             rec["prompt_id"], rec["language"], rec["round"], rec["error"][:100])
            if n % checkpoint_interval == 0:
                save_checkpoint(out_root, languages, num_rounds, existing_results, results_by_lang_round)
                # Log progress with error count
                error_count = sum(1 for lang_dict in results_by_lang_round.values() 
                                for round_list in lang_dict.values() 
                                for r in round_list if r.get("error"))
                logger.info("Progress: %s/%s tasks completed (%s errors)", n, len(tasks), error_count)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_and_store, t) for t in tasks]
        # Use tqdm progress bar with ETA
        with tqdm(total=len(tasks), desc="Processing", unit="task",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for _ in as_completed(futs):
                pbar.update(1)

    save_checkpoint(out_root, languages, num_rounds, existing_results, results_by_lang_round)
    
    # Final statistics
    total_completed = sum(len(round_list) 
                         for lang_dict in results_by_lang_round.values() 
                         for round_list in lang_dict.values())
    total_errors = sum(1 
                      for lang_dict in results_by_lang_round.values() 
                      for round_list in lang_dict.values() 
                      for r in round_list if r.get("error"))
    total_success = total_completed - total_errors
    
    logger.info("=" * 60)
    logger.info("Final Statistics:")
    logger.info("  Total tasks completed: %s", total_completed)
    logger.info("  Successful: %s", total_success)
    logger.info("  Failed: %s", total_errors)
    if total_errors > 0:
        logger.warning("  Error rate: %.1f%%", (total_errors / total_completed * 100) if total_completed > 0 else 0)
        logger.warning("  Check the result files for detailed error messages")
    logger.info("=" * 60)
    logger.info("Done. Results under %s", out_root)


if __name__ == "__main__":
    main()
