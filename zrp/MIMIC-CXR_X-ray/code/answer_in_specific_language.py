#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import logging
import time
import threading
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# -------------------------------------------


def sanitize_filename(filename: str) -> str:
    """Replace invalid filename characters with underscores."""
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    invalid_chars = r'[<>:"\\|?*]'
    sanitized = re.sub(invalid_chars, '_', basename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return os.path.join(directory, sanitized) if directory else sanitized

def extract_indication(report_text: str) -> str:
    """
    Extract INDICATION section from report text.
    Returns empty string if not found.
    """
    if not report_text:
        return ""
    
    # Look for INDICATION: pattern
    indication_pattern = r'INDICATION:\s*(.*?)(?=\n\s*[A-Z]+:|$)'
    match = re.search(indication_pattern, report_text, re.IGNORECASE | re.DOTALL)
    if match:
        indication = match.group(1).strip()
        # Clean up common patterns
        indication = re.sub(r'\s+', ' ', indication)
        return indication
    
    return ""


def get_patient_group(subject_id: str) -> str:
    """Extract patient group (p10, p11, etc.) from subject_id."""
    if not subject_id:
        return None
    # Extract first 2-3 digits after 'p' or from the number
    subject_str = str(subject_id).lstrip('p')
    if len(subject_str) >= 2:
        prefix = subject_str[:2]
        return f"p{prefix}"
    return None


def find_study_images(dataset_dir: Path, subject_id: str, study_id: str) -> List[str]:
    """
    Find all image files for a given study_id.
    Returns list of absolute paths to image files.
    """
    patient_group = get_patient_group(subject_id)
    if not patient_group:
        return []
    
    # Ensure subject_id has 'p' prefix (remove if already present)
    subject_id_clean = str(subject_id).lstrip('p')
    subject_dir_name = f"p{subject_id_clean}"
    
    study_dir = dataset_dir / "files" / patient_group / subject_dir_name / f"s{study_id}"
    if not study_dir.exists():
        return []
    
    image_files = sorted(study_dir.glob("*.jpg"))
    return [str(img) for img in image_files]


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def resolve_dataset_path(path_str: str) -> Path:
    """
    Resolve dataset path, trying multiple locations if the given path doesn't exist.
    Handles the case where script is run from MIMIC-CXR_X-ray/code/ directory.
    """
    path = Path(path_str)
    
    # If absolute path exists, use it
    if path.is_absolute() and path.exists():
        return path
    
    # If relative path exists from current directory, use it
    if path.exists():
        return path.resolve()
    
    # Try resolving from current directory's parent (if running from code/)
    # This handles: running from MIMIC-CXR_X-ray/code/ with path "MIMIC-CXR_X-ray/dataset"
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    
    # If path_str contains "MIMIC-CXR_X-ray/", extract just the part after it
    if "MIMIC-CXR_X-ray/" in path_str:
        # Extract the part after MIMIC-CXR_X-ray/ (e.g., "dataset" from "MIMIC-CXR_X-ray/dataset")
        suffix = path_str.split("MIMIC-CXR_X-ray/", 1)[1]
        alt_path = parent_dir / suffix
        if alt_path.exists():
            return alt_path.resolve()
    
    # Try: parent_dir / path_str (fallback)
    alt_path = parent_dir / path_str
    if alt_path.exists():
        return alt_path.resolve()
    
    # Try as absolute path from current directory
    resolved = path.resolve()
    if resolved.exists():
        return resolved
    
    # Return the original path (will raise error later if it doesn't exist)
    return path


def build_multimodal_messages(
    indication_text: str,
    image_paths: List[str],
    language: str
) -> List[Dict[str, Any]]:
    """
    Build multimodal messages with images and text prompt.
    
    Args:
        indication_text: Translated INDICATION text
        image_paths: List of paths to chest X-ray images
        language: Target language name
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    # Build the prompt according to requirements
    # 处理临床指征的默认值
    indication = indication_text if indication_text else {
        "English": "No specific clinical indication provided.",
        "Chinese": "未提供具体的临床适应症。",
        "Malay": "Tiada indikasi klinikal khusus disediakan.",
        "Thai": "ไม่มีข้อบ่งชี้ทางคลินิกที่ระบุเฉพาะเจาะจง"
    }.get(language, "No specific clinical indication provided.")

    prompts = {
            "English": """You are a professional radiologist expert in analyzing chest X-ray images and writing standardized clinical reports.

Your task is to generate a diagnostic report based on the provided images and clinical indications.

Clinical Context: {indication}

Instructions:
1. Systematic Observation: Inspect lung fields, cardiac silhouette, mediastinum, pleural spaces, and skeletal structures.
2. Focus: Search for imaging evidence (e.g., pneumothorax, infiltrates, or cardiac abnormalities) related to the symptoms mentioned in the clinical context.
3. Professionalism: Use standard medical terminology in English.

Output Format:
- Findings: [Detailed description of anatomical structures and abnormalities]
- Impression: [Final clinical conclusion]

Please provide a structured response containing identified image details (Findings) and the most likely diagnosis (Impression).""",

            "Chinese": """您是一位专业的放射科医生，擅长分析胸部X光片并撰写标准化临床报告。

您的任务是根据提供的图像和临床适应症生成诊断报告。

临床背景：{indication}

指令：
1. 系统观察：检查肺野、心脏轮廓、纵隔、胸膜腔和骨骼结构。
2. 重点：寻找与临床背景中提到的症状相关的影像学证据（例如，气胸、浸润或心脏异常）。
3. 专业性：使用标准的中文医学术语。

输出格式：
- 检查所见 (Findings)：[解剖结构和异常的详细描述]
- 印象/诊断意见 (Impression)：[最终临床结论]

请提供一个结构化的回答，包含识别出的图像细节（检查所见）和最可能的诊断（印象）。""",

            "Malay": """Anda adalah pakar radiologi profesional yang pakar dalam menganalisis imej X-ray dada dan menulis laporan klinikal piawai.

Tugas anda adalah untuk menghasilkan laporan diagnostik berdasarkan imej yang disediakan dan indikasi klinikal.

Konteks Klinikal: {indication}

Arahan:
1. Pemerhatian Sistematik: Periksa medan paru-paru, siluet jantung, mediastinum, ruang pleura, dan struktur rangka.
2. Fokus: Cari bukti pengimejan (cth., pneumotoraks, infiltrat, atau keabnormalan jantung) yang berkaitan dengan gejala yang dinyatakan dalam konteks klinikal.
3. Profesionalisme: Gunakan istilah perubatan standard dalam Bahasa Melayu.

Format Output:
- Penemuan (Findings): [Penerangan terperinci mengenai struktur anatomi dan keabnormalan]
- Kesan (Impression): [Kesimpulan klinikal akhir]

Sila berikan respons berstruktur yang mengandungi butiran imej yang dikenal pasti (Penemuan) dan diagnosis yang paling mungkin (Kesan).""",

            "Thai": """คุณเป็นรังสีแพทย์มืออาชีพผู้เชี่ยวชาญในการวิเคราะห์ภาพรังสีทรวงอก (Chest X-ray) และเขียนรายงานทางคลินิกที่เป็นมาตรฐาน

งานของคุณคือสร้างรายงานการวินิจฉัยตามภาพและข้อบ่งชี้ทางคลินิกที่ให้มา

บริบททางคลินิก: {indication}

คำแนะนำ:
1. การสังเกตอย่างเป็นระบบ: ตรวจสอบพื้นที่ปอด, เงาหัวใจ, เมดิแอสตินัม, ช่องเยื่อหุ้มปอด และโครงสร้างกระดูก
2. จุดเน้น: ค้นหาหลักฐานทางภาพ (เช่น ภาวะลมรั่วในช่องเยื่อหุ้มปอด, การแทรกซึม, หรือความผิดปกติของหัวใจ) ที่เกี่ยวข้องกับอาการที่ระบุในบริบททางคลินิก
3. ความเป็นมืออาชีพ: ใช้ศัพท์ทางการแพทย์มาตรฐานในภาษาไทย

รูปแบบผลลัพธ์:
- สิ่งที่ตรวจพบ (Findings): [คำอธิบายโดยละเอียดของโครงสร้างทางกายวิภาคและความผิดปกติ]
- ความคิดเห็นแพทย์ (Impression): [สรุปผลทางคลินิกขั้นสุดท้าย]

โปรดให้คำตอบที่มีโครงสร้างประกอบด้วยรายละเอียดภาพที่ระบุ (สิ่งที่ตรวจพบ) และการวินิจฉัยที่เป็นไปได้มากที่สุด (ความคิดเห็นแพทย์)"""
    }

    # 获取模板并填入临床指征
    system_content = prompts.get(language, prompts.get("English")).format(indication=indication)
    messages = [{"role": "system", "content": system_content}]
    
    # Build user message with images
    user_content = []
    
    # Add images
    for img_path in image_paths:
        base64_image = encode_image(img_path)
        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    instructions = {
        "English": "Please analyze these chest X-ray images and provide a diagnostic report in English based on the clinical context provided.",
        "Chinese": "请根据提供的临床背景分析这些胸部X光图像，并提供一份中文诊断报告。",
        "Malay": "Sila analisis imej X-ray dada ini dan sediakan laporan diagnostik dalam Bahasa Melayu berdasarkan konteks klinikal yang disediakan.",
        "Thai": "โปรดวิเคราะห์ภาพรังสีทรวงอกเหล่านี้และจัดทำรายงานการวินิจฉัยเป็นภาษาไทยตามบริบททางคลินิกที่ระบุ"
    }

    # Add text instruction
    user_instruction = instructions.get(language, instructions.get("English"))

    user_content.append({
        "type": "text",
        "text": user_instruction
    })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    return messages


class DiagnosisRunner:
    """
    Multimodal multi-language medical evaluation runner for MIMIC-CXR.
    - Processes chest X-ray images with clinical indications
    - Supports multiple languages (English, Chinese, Malay, Thai)
    - Runs experiments in parallel for each language
    - Supports checkpointing and resume
    """

    def __init__(
        self,
        model: str = "google/gemini-3-pro-preview",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        max_retries: int = 5,
        retry_backoff_base: float = 1.5,
        log_dir: str = "logs",
        max_workers: int = 10,
        dataset_dir: str = "MIMIC-CXR_X-ray/dataset",
        translate_dir: str = "MIMIC-CXR_X-ray/result/translate/google_gemini-3-pro-preview"
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set via --api_key or env var.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.max_workers = max_workers
        
        # Resolve paths, trying multiple locations if needed
        self.dataset_dir = resolve_dataset_path(dataset_dir)
        self.translate_dir = resolve_dataset_path(translate_dir)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        # Use RLock because some code paths call _save_results while already holding the lock.
        self._lock = threading.RLock()

        self._setup_logging_file()

    def _setup_logging_file(self):
        """Configure logging with timestamp-based filename."""
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = sanitize_filename(self.model.replace('/', '_'))
        log_file = self.log_dir / f"mimic_cxr_runner_{sanitized_model}_{ts}.log"
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.propagate = False
        logger.info(f"Logging initialized; file: {log_file}")

    def _chat_call(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Tuple[str, Dict[str, int], float, Dict[str, Any]]:
        """Single chat.completions call with retries."""
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout_s,
                )
                elapsed = time.time() - t0

                if not resp or not resp.choices:
                    raise ValueError("Empty API response")
                message = resp.choices[0].message
                content = (message.content or "").strip()

                # Get usage from resp.model_dump()['usage']
                resp_dict = resp.model_dump()
                raw_usage = resp_dict.get("usage") or getattr(resp, "usage", None)
                usage_map = self._coerce_usage(raw_usage)

                return content, usage_map, elapsed, resp_dict

            except Exception as e:
                logger.error(f"API error attempt {attempt}: {e}")
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_base ** attempt)

    def _coerce_usage(self, usage_obj) -> Dict[str, int]:
        """
        Normalize various 'usage' shapes to a dict with keys:
        input_tokens, output_tokens, total_tokens, reasoning_tokens.
        """
        if usage_obj is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

        # If it's already a dict
        if isinstance(usage_obj, dict):
            it = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
            ot = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
            tt = usage_obj.get("total_tokens") or (it + ot)
            
            reasoning_tokens = 0
            completion_details = usage_obj.get("completion_tokens_details", {})
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens") or 0
            
            return {"input_tokens": it, "output_tokens": ot, "total_tokens": tt, "reasoning_tokens": int(reasoning_tokens or 0)}

        # Try model_dump() (pydantic / openai sdk objects)
        try:
            if hasattr(usage_obj, "model_dump"):
                ud = usage_obj.model_dump()
            else:
                ud = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                }
            it = ud.get("prompt_tokens") or ud.get("input_tokens") or 0
            ot = ud.get("completion_tokens") or ud.get("output_tokens") or 0
            tt = ud.get("total_tokens") or (it + ot)
            
            reasoning_tokens = 0
            completion_details = ud.get("completion_tokens_details", {})
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens") or 0
            
            return {"input_tokens": int(it or 0), "output_tokens": int(ot or 0), "total_tokens": int(tt or 0), "reasoning_tokens": int(reasoning_tokens or 0)}
        except Exception:
            try:
                it = int(getattr(usage_obj, "prompt_tokens", 0) or getattr(usage_obj, "input_tokens", 0) or 0)
                ot = int(getattr(usage_obj, "completion_tokens", 0) or getattr(usage_obj, "output_tokens", 0) or 0)
                tt = int(getattr(usage_obj, "total_tokens", 0) or (it + ot))
                
                reasoning_tokens = 0
                if hasattr(usage_obj, "completion_tokens_details"):
                    details = getattr(usage_obj, "completion_tokens_details", None)
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                
                return {"input_tokens": it, "output_tokens": ot, "total_tokens": tt, "reasoning_tokens": int(reasoning_tokens or 0)}
            except Exception:
                return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    def _run_one_sample(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        For one sample:
        - Make API call with multimodal messages
        - Return the assistant response
        """
        # Make API call
        reply, usage, elapsed, raw = self._chat_call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        reasoning_info = ""
        if usage.get("reasoning_tokens", 0) > 0:
            reasoning_info = f" (reasoning: {usage['reasoning_tokens']})"
        logger.info(
            f"API call completed | "
            f"Tokens in/out/total: {usage['input_tokens']}/{usage['output_tokens']}{reasoning_info}/{usage['total_tokens']} | "
            f"Time={elapsed:.2f}s"
        )

        call_logs = [{
            "response_time_sec": elapsed,
            "usage": usage,
            "response_preview": reply[:200],
            "raw": raw,
        }]
        
        return {
            "model_response": reply,
            "total_response_time_sec": round(elapsed, 3),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "call_logs": call_logs,
        }

    def _load_study_data(self) -> pd.DataFrame:
        """Load study data from metadata CSV."""
        metadata_file = self.dataset_dir / "mimic-cxr-2.0.0-metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded {len(df)} records from metadata file")
        
        # Get unique study_id and subject_id pairs
        study_df = df[['subject_id', 'study_id']].drop_duplicates()
        logger.info(f"Found {len(study_df)} unique studies")
        
        return study_df

    def _get_indication(
        self,
        subject_id: str,
        study_id: str,
        language: str
    ) -> str:
        """
        Get INDICATION text for a study in the specified language.
        Returns empty string if not found.
        """
        patient_group = get_patient_group(subject_id)
        if not patient_group:
            return ""
        
        # Ensure subject_id has 'p' prefix (remove if already present)
        subject_id_clean = str(subject_id).lstrip('p')
        subject_dir_name = f"p{subject_id_clean}"
        
        if language == "English":
            # Read from original dataset
            report_path = self.dataset_dir / "reports" / patient_group / subject_dir_name / f"s{study_id}.txt"
        else:
            # Read from translated directory
            report_path = self.translate_dir / language / patient_group / subject_dir_name / f"s{study_id}.txt"
        
        if not report_path.exists():
            logger.warning(f"Report not found: {report_path}")
            return ""
        
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
            indication = extract_indication(report_text)
            return indication
        except Exception as e:
            logger.error(f"Failed to read report {report_path}: {e}")
            return ""

    def _process_single_sample(
        self,
        subject_id: str,
        study_id: str,
        language: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Process a single sample (study_id + language combination)."""
        try:
            # Get indication text in the specified language
            indication = self._get_indication(subject_id, study_id, language)
            logger.debug(f"Processing study {study_id} in {language}, indication length: {len(indication)}")
            
            # Find images for this study
            image_paths = find_study_images(self.dataset_dir, subject_id, study_id)
            
            if not image_paths:
                logger.warning(f"No images found for study {study_id}, subject {subject_id}")
                return None
            
            # Build multimodal messages with language-specific prompt
            messages = build_multimodal_messages(indication, image_paths, language)
            
            # Make API call
            res = self._run_one_sample(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return {
                "study_id": str(study_id),
                "subject_id": str(subject_id),
                "language": language,
                "indication": indication,
                "image_count": len(image_paths),
                "image_paths": image_paths,
                "result": res,
            }
        except Exception as e:
            logger.error(f"Failed study {study_id}, language {language}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def process(
        self,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        checkpoint_interval: int = 5,
        rounds: int = 3,
        limit: Optional[int] = None,
        limit_per_language: Optional[int] = None,
        target_languages: Optional[List[str]] = None
    ) -> None:
        """
        Process multimodal multi-language medical evaluation task.
        """
        # Language configuration
        all_languages = ["English", "Chinese", "Malay", "Thai"]
        
        # Filter languages if target_languages is specified
        if target_languages is not None:
            target_languages_set = set(target_languages)
            all_languages = [lang for lang in all_languages if lang in target_languages_set]
            logger.info(f"Filtered to target languages: {all_languages}")
        
        # Load study data
        study_df = self._load_study_data()
        
        # Create output directory structure
        # Format: MIMIC-CXR_X-ray/result/response/{model_name}/{Language}/round{1-3}.json
        base_output_dir = resolve_dataset_path("MIMIC-CXR_X-ray/result/response")
        sanitized_model = sanitize_filename(self.model.replace('/', '_'))
        
        # Perform multiple rounds
        for rnd in range(1, int(rounds) + 1):
            logger.info(f"Starting round {rnd}/{rounds}")
            
            # Process each language separately
            for language in all_languages:
                logger.info(f"Processing language: {language}, round {rnd}")
                
                # Output file for this language and round
                lang_output_dir = base_output_dir / sanitized_model / language
                lang_output_dir.mkdir(parents=True, exist_ok=True)
                output_file = lang_output_dir / f"round{rnd}.json"
                
                # Load existing results for checkpointing
                results = []
                processed_studies = set()
                
                if output_file.exists():
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        processed_studies = {r["study_id"] for r in results if "study_id" in r}
                        logger.info(f"Round {rnd}, {language}: Resuming from checkpoint, found {len(processed_studies)} already processed studies")
                    except Exception as e:
                        logger.warning(f"Could not read existing JSON for round {rnd}, {language}: {e}. Starting fresh.")
                        results = []
                        processed_studies = set()
                
                # Prepare tasks
                tasks = []
                for _, row in study_df.iterrows():
                    subject_id = str(row['subject_id'])
                    study_id = str(row['study_id'])
                    
                    if study_id in processed_studies:
                        continue
                    
                    tasks.append((subject_id, study_id, language))
                
                # Apply limit if specified (limit_per_language takes precedence over limit)
                if limit_per_language is not None and limit_per_language > 0:
                    tasks = tasks[:limit_per_language]
                    logger.info(f"Round {rnd}, {language}: Limited to {limit_per_language} cases per language")
                elif limit is not None and limit > 0:
                    tasks = tasks[:limit]
                
                logger.info(f"Round {rnd}, {language}: Prepared {len(tasks)} tasks")
                
                if len(tasks) == 0:
                    logger.info(f"Round {rnd}, {language}: No new samples to process")
                    continue
                
                # Process in parallel
                processed = 0
                skipped = 0
                errors = 0
                
                def write_result_threadsafe(sample_result: Dict[str, Any]):
                    """Thread-safe result writing."""
                    nonlocal processed, skipped, errors
                    
                    if sample_result is None:
                        with self._lock:
                            skipped += 1
                        return
                    
                    with self._lock:
                        # Verify language matches to ensure data isolation
                        if sample_result.get("language") != language:
                            logger.error(f"Language mismatch! Expected {language}, got {sample_result.get('language')}")
                        
                        results.append(sample_result)
                        processed += 1
                        processed_studies.add(sample_result["study_id"])
                        
                        # Checkpoint: save every N results (5 by default)
                        if processed % checkpoint_interval == 0:
                            self._save_results(output_file, results)
                            logger.info(f"Round {rnd}, {language}: Checkpoint saved ({processed} processed, {skipped} skipped, {errors} errors)")
                
                try:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_task = {
                            executor.submit(
                                self._process_single_sample,
                                subject_id, study_id, language,
                                temperature, max_tokens
                            ): (subject_id, study_id)
                            for subject_id, study_id, _ in tasks
                        }
                        
                        # Process completed tasks with progress bar
                        for future in tqdm(as_completed(future_to_task.keys()), total=len(tasks), desc=f"Round {rnd}, {language}", unit="task"):
                            try:
                                result = future.result()
                                write_result_threadsafe(result)
                            except Exception as e:
                                with self._lock:
                                    errors += 1
                                    logger.error(f"Error processing result: {e}")
                
                except KeyboardInterrupt:
                    logger.info(f"Round {rnd}, {language} interrupted by user - saving current progress...")
                    self._save_results(output_file, results)
                    logger.info(f"Round {rnd}, {language}: Saved {processed} newly processed samples (skipped {skipped}, errors {errors})")
                    raise
                finally:
                    # Final save
                    self._save_results(output_file, results)
                
                logger.info(f"Round {rnd}, {language} completed. Output: {output_file}")
                logger.info(f"Round {rnd}, {language}: Processed {processed} new samples, skipped {skipped} existing, errors {errors}")

        logger.info(f"All {rounds} rounds completed. Results saved in per-language JSON files.")

    def _save_results(self, output_file: Path, results: List[Dict[str, Any]]):
        """Save results to JSON file (thread-safe)."""
        with self._lock:
            try:
                # Create backup if file exists
                if output_file.exists():
                    backup_file = output_file.with_suffix('.json.bak')
                    import shutil
                    shutil.copy2(output_file, backup_file)
                
                # Write results
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save results to {output_file}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Multi-language Medical Evaluation Runner (MIMIC-CXR)")
    parser.add_argument("--model", default="google/gemini-3-pro-preview",
                        help="Model name to use")
    parser.add_argument("--api_key", 
                        default=None, 
                        help="API key (will use OPENAI_API_KEY env var if not specified)")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                        help="Base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum tokens in response")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="API timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum number of API retries")
    parser.add_argument("--retry-backoff-base", type=float, default=1.5,
                        help="Exponential backoff base for retries")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Checkpoint interval (save every N results)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Maximum number of parallel workers (default: 10)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of rounds to process (default: 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing, default: None = process all)")
    parser.add_argument("--limit-per-language", type=int, default=None,
                        help="Limit number of samples per language (for testing, e.g., 10 for 10 cases per language)")
    parser.add_argument("--target-languages", nargs="+", default=None,
                        help="Only process specified languages (e.g., --target-languages English Chinese Malay Thai)")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("--translate-dir", default="result/translate/google_gemini-3-pro-preview",
                        help="Path to translated reports directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get API key from environment variable if not specified
    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY")
        if args.api_key is None:
            raise ValueError("API key must be provided via --api_key argument or OPENAI_API_KEY environment variable")

    runner = DiagnosisRunner(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff_base,
        log_dir=args.log_dir,
        max_workers=args.max_workers,
        dataset_dir=args.dataset_dir,
        translate_dir=args.translate_dir,
    )

    try:
        runner.process(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            checkpoint_interval=args.checkpoint_interval,
            rounds=args.rounds,
            limit=args.limit,
            limit_per_language=args.limit_per_language,
            target_languages=args.target_languages
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(0)
