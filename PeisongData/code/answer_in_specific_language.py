#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-turn LMM evaluation for re-intubation prediction across zh, en, ms, th.

Workflow per image (single context window):
  Step 1 (Data Extraction): Extract key data + time-series trends → structured output.
  Step 2 (Clinical Prediction): Based on Step 1, answer "re-intubation within 6h?" with Analysis + Conclusion (Yes/No).
"""

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

from tqdm import tqdm

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).resolve().parent
    # 依次尝试：工作区根 .env、PeisongData/.env、脚本目录 .env（override=True 确保 .env 覆盖已有空值）
    for _d in (_script_dir.parent.parent, _script_dir.parent, _script_dir):
        _f = _d / ".env"
        if _f.exists():
            load_dotenv(_f, override=True)
            logger.info(f"Loaded .env from {_f}")
            break
except ImportError:
    pass
# -------------------------------------------


def sanitize_filename(filename: str) -> str:
    """Replace invalid filename characters with underscores."""
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    invalid_chars = r'[<>:"\\|?*]'
    sanitized = re.sub(invalid_chars, '_', basename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return os.path.join(directory, sanitized) if directory else sanitized


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def get_trend_types_definition(lang: str) -> str:
    """Get time-series trend type definitions for structured extraction."""
    definitions = {
        "zh": """
时序趋势类型定义（共13种）：

1 = 长时间平稳（数值波动<10%）
2 = 长时间波动（数值波动≥10%但无明显方向）
3 = 持续下降（整体下降趋势>20%）
4 = 持续上升（整体上升趋势>20%）
5 = 先平稳后上升（前段平稳，后段上升）
6 = 先平稳后下降（前段平稳，后段下降）
7 = 先上升后平稳（前段上升，后段平稳）
8 = 先下降后平稳（前段下降，后段平稳）
9 = 先下降后上升（V型）
10 = 先上升后下降（倒V型）
11 = 先波动后上升（前段波动，后段上升）
12 = 先波动后下降（前段波动，后段下降）
13 = 缺失数据或无法判断""",
        "en": """
Time-series Trend Type Definitions (13 Types):

1 = Long-term stable (fluctuation <10%)
2 = Long-term fluctuating (fluctuation ≥10%, no clear direction)
3 = Continuous decline (overall declining >20%)
4 = Continuous rise (overall rising >20%)
5 = Stable then rising (first phase stable, second phase rising)
6 = Stable then declining (first phase stable, second phase declining)
7 = Rising then stable (first phase rising, second phase stable)
8 = Declining then stable (first phase declining, second phase stable)
9 = Declining then rising (V-shape)
10 = Rising then declining (inverted V-shape)
11 = Fluctuating then rising (first phase fluctuating, second phase rising)
12 = Fluctuating then declining (first phase fluctuating, second phase declining)
13 = Missing data or indeterminate""",
        "ms": """
Definisi Jenis Trend Siri Masa (13 Jenis):

1 = Stabil jangka panjang (turun naik <10%)
2 = Berfluktuasi jangka panjang (turun naik ≥10%, tanpa arah jelas)
3 = Penurunan berterusan (trend menurun >20%)
4 = Peningkatan berterusan (trend meningkat >20%)
5 = Stabil kemudian meningkat (fasa pertama stabil, fasa kedua meningkat)
6 = Stabil kemudian menurun (fasa pertama stabil, fasa kedua menurun)
7 = Meningkat kemudian stabil (fasa pertama meningkat, fasa kedua stabil)
8 = Menurun kemudian stabil (fasa pertama menurun, fasa kedua stabil)
9 = Menurun kemudian meningkat (bentuk V)
10 = Meningkat kemudian menurun (bentuk V terbalik)
11 = Berfluktuasi kemudian meningkat (fasa pertama berfluktuasi, fasa kedua meningkat)
12 = Berfluktuasi kemudian menurun (fasa pertama berfluktuasi, fasa kedua menurun)
13 = Data hilang atau tidak dapat ditentukan""",
        "th": """
คำนิยามประเภทแนวโน้มอนุกรมเวลา (13 ประเภท):

1 = มีเสถียรภาพระยะยาว (ความผันผวน <10%)
2 = ผันผวนระยะยาว (ความผันผวน ≥10% ไม่มีทิศทางชัดเจน)
3 = ลดลงอย่างต่อเนื่อง (แนวโน้มลดลง >20%)
4 = เพิ่มขึ้นอย่างต่อเนื่อง (แนวโน้มเพิ่มขึ้น >20%)
5 = มีเสถียรภาพแล้วเพิ่มขึ้น (ระยะแรกมีเสถียรภาพ ระยะที่สองเพิ่มขึ้น)
6 = มีเสถียรภาพแล้วลดลง (ระยะแรกมีเสถียรภาพ ระยะที่สองลดลง)
7 = เพิ่มขึ้นแล้วมีเสถียรภาพ (ระยะแรกเพิ่มขึ้น ระยะที่สองมีเสถียรภาพ)
8 = ลดลงแล้วมีเสถียรภาพ (ระยะแรกลดลง ระยะที่สองมีเสถียรภาพ)
9 = ลดลงแล้วเพิ่มขึ้น (รูปตัว V)
10 = เพิ่มขึ้นแล้วลดลง (รูปตัว V กลับหัว)
11 = ผันผวนแล้วเพิ่มขึ้น (ระยะแรกผันผวน ระยะที่สองเพิ่มขึ้น)
12 = ผันผวนแล้วลดลง (ระยะแรกผันผวน ระยะที่สองลดลง)
13 = ข้อมูลสูญหายหรือไม่สามารถกำหนดได้"""
    }
    return definitions.get(lang, definitions["en"])


def get_medical_prompts() -> Dict[str, Dict[str, str]]:
    """
    Monolingual medical prompts for Step 1 (Extraction) and Step 2 (Prediction)
    for zh, en, ms, th. System + step1_user + step2_user per language.
    """
    prompts = {
        "zh": {
            "system": "你是一位经验丰富的重症监护医生，专门负责评估患者是否需要重新插管。请用中文与用户交流。",
            "step1_user": f"""请仔细查看这张包含患者12小时监测数据的图像。图像中包含多个时序监测子图（从左到右、从上到下排列）。

请完成以下任务并以JSON格式输出：

**任务1：提取关键数据（必须包含具体数值范围）**
请提取以下信息：
- patient_info: 患者基本信息（性别、年龄等）
- vital_signs: 生命体征数据，每项必须包含：
  - parameter_name: 参数名称
  - min_value: 最小值
  - max_value: 最大值
  - unit: 单位
  - mean_value: 平均值（如可计算）
  
示例格式：
{{
  "patient_info": {{"age": "XX岁", "gender": "男/女"}},
  "vital_signs": [
    {{"parameter_name": "心率", "min_value": 60, "max_value": 100, "mean_value": 80, "unit": "bpm"}},
    {{"parameter_name": "血压收缩压", "min_value": 110, "max_value": 140, "mean_value": 125, "unit": "mmHg"}},
    ...
  ]
}}

**任务2：识别时序变化趋势（结构化标签）**
{get_trend_types_definition("zh")}

请按照图像中子图的顺序（从左到右、从上到下，编号1-16），为每个时序图分配趋势类型。

输出格式：
{{
  "time_series_trends": {{
    "1": {{"parameter": "参数名", "trend_type": 类型编号, "description": "简短描述"}},
    "2": {{"parameter": "参数名", "trend_type": 类型编号, "description": "简短描述"}},
    ...
    "16": {{"parameter": "参数名", "trend_type": 类型编号, "description": "简短描述"}}
  }}
}}

**最终输出必须是完整的JSON格式，包含patient_info、vital_signs和time_series_trends三个部分。**""",
            "step2_user": """基于你刚才提取的数据，请回答以下问题：

如果患者现在拔管，是否会在接下来的6小时内需要重新插管？

**重要提示：如果患者年龄大于等于90岁，提取的年龄会显示为300。请根据这个信息和上面提取的所有数据来进行判断。**

**请严格按照以下JSON格式输出：**
{{
  "analysis": "基于生命体征的详细推理过程，包括对关键指标的分析和风险评估",
  "conclusion": "是" 或 "否"
}}

**注意：输出必须是有效的JSON格式，conclusion字段必须严格为「是」或「否」。**"""
        },
        "en": {
            "system": "You are an experienced intensive care physician specializing in assessing whether patients require re-intubation. Communicate in English.",
            "step1_user": f"""Please carefully examine this image containing 12 hours of patient monitoring data. The image contains multiple time-series subplots (arranged left-to-right, top-to-bottom).

Complete the following tasks and output in JSON format:

**Task 1: Extract Key Data (must include specific value ranges)**
Extract the following information:
- patient_info: Basic patient information (gender, age, etc.)
- vital_signs: Vital sign data, each entry must include:
  - parameter_name: Parameter name
  - min_value: Minimum value
  - max_value: Maximum value
  - unit: Unit of measurement
  - mean_value: Average value (if calculable)
  
Example format:
{{
  "patient_info": {{"age": "XX years", "gender": "Male/Female"}},
  "vital_signs": [
    {{"parameter_name": "Heart Rate", "min_value": 60, "max_value": 100, "mean_value": 80, "unit": "bpm"}},
    {{"parameter_name": "Systolic BP", "min_value": 110, "max_value": 140, "mean_value": 125, "unit": "mmHg"}},
    ...
  ]
}}

**Task 2: Identify Time-Series Trends (structured labels)**
{get_trend_types_definition("en")}

According to the order of subplots in the image (left-to-right, top-to-bottom, numbered 1-16), assign a trend type to each time-series plot.

Output format:
{{
  "time_series_trends": {{
    "1": {{"parameter": "parameter name", "trend_type": type_number, "description": "brief description"}},
    "2": {{"parameter": "parameter name", "trend_type": type_number, "description": "brief description"}},
    ...
    "16": {{"parameter": "parameter name", "trend_type": type_number, "description": "brief description"}}
  }}
}}

**Final output must be complete JSON format, containing patient_info, vital_signs, and time_series_trends sections.**""",
            "step2_user": """Based on the data you just extracted, please answer:

If the patient is extubated now, will they require re-intubation within the next 6 hours?

**Important Note: If the patient's age is 90 years or older, the extracted age will be shown as 300. Please make your judgment based on this information and all the data extracted above.**

**Please provide your response strictly in the following JSON format:**
{{
  "analysis": "Detailed reasoning based on the vital signs, including analysis of key indicators and risk assessment",
  "conclusion": "Yes" or "No"
}}

**Note: Output must be valid JSON format, and the conclusion field must strictly be "Yes" or "No"."""
        },
        "ms": {
            "system": "Anda ialah doktor penjagaan rapi yang berpengalaman, pakar menilai sama ada pesakit memerlukan intubasi semula. Berkomunikasi dalam Bahasa Melayu.",
            "step1_user": f"""Sila periksa imej ini dengan teliti yang mengandungi data pemantauan pesakit 12 jam. Imej mengandungi beberapa subplot siri masa (disusun kiri-ke-kanan, atas-ke-bawah).

Selesaikan tugasan berikut dan output dalam format JSON:

**Tugasan 1: Ekstrak Data Utama (mesti termasuk julat nilai khusus)**
Ekstrak maklumat berikut:
- patient_info: Maklumat asas pesakit (jantina, umur, dll.)
- vital_signs: Data tanda vital, setiap entri mesti termasuk:
  - parameter_name: Nama parameter
  - min_value: Nilai minimum
  - max_value: Nilai maksimum
  - unit: Unit pengukuran
  - mean_value: Nilai purata (jika boleh dikira)
  
Format contoh:
{{
  "patient_info": {{"age": "XX tahun", "gender": "Lelaki/Perempuan"}},
  "vital_signs": [
    {{"parameter_name": "Kadar Jantung", "min_value": 60, "max_value": 100, "mean_value": 80, "unit": "bpm"}},
    {{"parameter_name": "Tekanan Darah Sistolik", "min_value": 110, "max_value": 140, "mean_value": 125, "unit": "mmHg"}},
    ...
  ]
}}

**Tugasan 2: Kenal pasti Trend Siri Masa (label berstruktur)**
{get_trend_types_definition("ms")}

Mengikut susunan subplot dalam imej (kiri-ke-kanan, atas-ke-bawah, bernombor 1-16), berikan jenis trend kepada setiap plot siri masa.

Format output:
{{
  "time_series_trends": {{
    "1": {{"parameter": "nama parameter", "trend_type": nombor_jenis, "description": "penerangan ringkas"}},
    "2": {{"parameter": "nama parameter", "trend_type": nombor_jenis, "description": "penerangan ringkas"}},
    ...
    "16": {{"parameter": "nama parameter", "trend_type": nombor_jenis, "description": "penerangan ringkas"}}
  }}
}}

**Output akhir mesti dalam format JSON lengkap, mengandungi bahagian patient_info, vital_signs, dan time_series_trends.**""",
            "step2_user": """Berdasarkan data yang anda baru ekstrak, sila jawab:

Jika pesakit dinyah-tiub sekarang, adakah mereka akan memerlukan intubasi semula dalam tempoh 6 jam akan datang?

**Nota Penting: Jika umur pesakit adalah 90 tahun atau lebih, umur yang diekstrak akan ditunjukkan sebagai 300. Sila buat penilaian anda berdasarkan maklumat ini dan semua data yang diekstrak di atas.**

**Sila berikan respons anda dengan ketat dalam format JSON berikut:**
{{
  "analysis": "Penaakulan terperinci berdasarkan tanda vital, termasuk analisis penunjuk utama dan penilaian risiko",
  "conclusion": "Ya" atau "Tidak"
}}

**Nota: Output mesti dalam format JSON yang sah, dan medan conclusion mesti dengan ketat "Ya" atau "Tidak"."""
        },
        "th": {
            "system": "คุณเป็นแพทย์ผู้เชี่ยวชาญด้านการดูแลผู้ป่วยหนักที่มีประสบการณ์ เชี่ยวชาญการประเมินว่าผู้ป่วยต้องการการใส่ท่อช่วยหายใจซ้ำหรือไม่ ตอบเป็นภาษาไทย",
            "step1_user": f"""กรุณาตรวจสอบภาพนี้อย่างละเอียดที่มีข้อมูลการติดตามผู้ป่วย 12 ชั่วโมง ภาพประกอบด้วยแผนภูมิย่อยอนุกรมเวลาหลายรายการ (เรียงจากซ้ายไปขวา บนลงล่าง)

ทำภารกิจต่อไปนี้และให้ผลลัพธ์ในรูปแบบ JSON:

**ภารกิจ 1: แยกข้อมูลสำคัญ (ต้องรวมช่วงค่าที่เฉพาะเจาะจง)**
แยกข้อมูลต่อไปนี้:
- patient_info: ข้อมูลพื้นฐานของผู้ป่วย (เพศ อายุ ฯลฯ)
- vital_signs: ข้อมูลสัญญาณชีพ แต่ละรายการต้องรวม:
  - parameter_name: ชื่อพารามิเตอร์
  - min_value: ค่าต่ำสุด
  - max_value: ค่าสูงสุด
  - unit: หน่วยวัด
  - mean_value: ค่าเฉลี่ย (ถ้าคำนวณได้)
  
ตัวอย่างรูปแบบ:
{{
  "patient_info": {{"age": "XX ปี", "gender": "ชาย/หญิง"}},
  "vital_signs": [
    {{"parameter_name": "อัตราการเต้นของหัวใจ", "min_value": 60, "max_value": 100, "mean_value": 80, "unit": "bpm"}},
    {{"parameter_name": "ความดันโลหิตซิสโตลิก", "min_value": 110, "max_value": 140, "mean_value": 125, "unit": "mmHg"}},
    ...
  ]
}}

**ภารกิจ 2: ระบุแนวโน้มอนุกรมเวลา (ป้ายกำกับที่มีโครงสร้าง)**
{get_trend_types_definition("th")}

ตามลำดับของแผนภูมิย่อยในภาพ (ซ้ายไปขวา บนลงล่าง หมายเลข 1-16) กำหนดประเภทแนวโน้มให้กับแต่ละแผนภูมิอนุกรมเวลา

รูปแบบผลลัพธ์:
{{
  "time_series_trends": {{
    "1": {{"parameter": "ชื่อพารามิเตอร์", "trend_type": หมายเลขประเภท, "description": "คำอธิบายสั้นๆ"}},
    "2": {{"parameter": "ชื่อพารามิเตอร์", "trend_type": หมายเลขประเภท, "description": "คำอธิบายสั้นๆ"}},
    ...
    "16": {{"parameter": "ชื่อพารามิเตอร์", "trend_type": หมายเลขประเภท, "description": "คำอธิบายสั้นๆ"}}
  }}
}}

**ผลลัพธ์สุดท้ายต้องเป็นรูปแบบ JSON ที่สมบูรณ์ ประกอบด้วยส่วน patient_info, vital_signs และ time_series_trends**""",
            "step2_user": """จากข้อมูลที่คุณแยกไว้ กรุณาตอบ:

หากผู้ป่วยถูกถอดท่อช่วยหายใจตอนนี้ พวกเขาจะต้องได้รับการใส่ท่อช่วยหายใจซ้ำภายใน 6 ชั่วโมงหรือไม่?

**หมายเหตุสำคัญ: หากอายุของผู้ป่วยเป็น 90 ปีหรือมากกว่า อายุที่แยกออกมาจะแสดงเป็น 300 กรุณาตัดสินใจโดยอิงจากข้อมูลนี้และข้อมูลทั้งหมดที่แยกไว้ข้างต้น**

**กรุณาให้คำตอบอย่างเข้มงวดในรูปแบบ JSON ต่อไปนี้:**
{{
  "analysis": "การให้เหตุผลโดยละเอียดตามสัญญาณชีพ รวมถึงการวิเคราะห์ตัวชี้วัดหลักและการประเมินความเสี่ยง",
  "conclusion": "ใช่" หรือ "ไม่ใช่"
}}

**หมายเหตุ: ผลลัพธ์ต้องเป็นรูปแบบ JSON ที่ถูกต้อง และฟิลด์ conclusion ต้องเป็น "ใช่" หรือ "ไม่ใช่" อย่างเข้มงวด**"""
        }
    }
    return prompts


class APIError(Exception):
    """Base class for API errors."""
    pass


class AuthenticationError(APIError):
    """API authentication failed."""
    pass


class RateLimitError(APIError):
    """API rate limit exceeded."""
    pass


class TimeoutError(APIError):
    """API request timeout."""
    pass


def call_lmm_multi_turn_api(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout_s: float = 120.0,
) -> Tuple[str, Dict[str, int], float, Dict[str, Any]]:
    """
    Call LMM with a history of messages/images; returns the next assistant reply.

    messages: list of {"role": "system"|"user"|"assistant", "content": str | list}
      For user messages with image, content = [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:..."}}].

    Returns:
      response_text: str
      usage: {input_tokens, output_tokens, total_tokens, reasoning_tokens}
      elapsed: float (seconds)
      raw_response: dict
      
    Raises:
      AuthenticationError: Invalid API key
      RateLimitError: Rate limit exceeded
      TimeoutError: Request timeout
      APIError: Other API errors
    """
    try:
        from openai import OpenAI
        import openai
    except ImportError:
        raise ImportError("openai package is required. Install: pip install openai")

    key = (api_key or "").strip()
    if not key:
        raise ValueError("API key is required")

    client = OpenAI(api_key=key, base_url=base_url)
    t0 = time.time()
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
    except openai.AuthenticationError as e:
        raise AuthenticationError(f"Authentication failed: {e}")
    except openai.RateLimitError as e:
        raise RateLimitError(f"Rate limit exceeded: {e}")
    except (openai.APITimeoutError, openai.Timeout) as e:
        raise TimeoutError(f"Request timeout: {e}")
    except openai.APIError as e:
        raise APIError(f"API error: {e}")
    
    elapsed = time.time() - t0

    if not resp or not resp.choices:
        raise ValueError("Empty API response")

    content = (resp.choices[0].message.content or "").strip()
    u = getattr(resp, "usage", None)
    if u:
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
            "reasoning_tokens": 0,
        }
        if hasattr(u, "completion_tokens_details"):
            d = getattr(u, "completion_tokens_details", None)
            if d and hasattr(d, "reasoning_tokens"):
                usage["reasoning_tokens"] = getattr(d, "reasoning_tokens", 0) or 0
    else:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    return content, usage, elapsed, resp.model_dump()


def parse_step2_response(text: str, lang: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse Step 2 response which should be in JSON format with 'analysis' and 'conclusion' fields.
    Falls back to legacy parsing if JSON parsing fails.
    
    Returns:
        (analysis, conclusion) tuple where conclusion is 'Yes'|'No' or None
    """
    if not text:
        return None, None
    
    t = text.strip()
    
    # Try to parse as JSON first (new structured format)
    try:
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found")
        
        data = json.loads(json_str)
        
        # Extract analysis and conclusion from JSON
        analysis = data.get("analysis", "").strip()
        conclusion_raw = data.get("conclusion", "").strip()
        
        # Normalize conclusion to Yes/No
        conclusion = None
        cl = conclusion_raw.lower()
        if cl in ("是", "yes", "ya", "ใช่"):
            conclusion = "Yes"
        elif cl in ("否", "no", "tidak", "ไม่ใช่"):
            conclusion = "No"
        
        if analysis and conclusion:
            logger.info(f"Successfully parsed structured JSON response: conclusion={conclusion}")
            return analysis, conclusion
        else:
            logger.warning(f"JSON parsed but missing fields: analysis={bool(analysis)}, conclusion={bool(conclusion)}")
            # Fall through to legacy parsing
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"JSON parsing failed, falling back to legacy parsing: {e}")
        # Fall through to legacy parsing
    
    # Legacy parsing: extract conclusion from free text
    conclusion = parse_conclusion_legacy(t, lang)
    return t, conclusion  # Return full text as analysis for backward compatibility


def parse_conclusion_legacy(text: str, lang: str) -> Optional[str]:
    """
    Legacy function: Extract strict Yes/No conclusion from Step 2 response with enhanced tolerance.
    Returns 'Yes'|'No' or None if cannot parse.
    """
    if not text:
        return None
        
    t = text.strip()
    tl = t.lower()
    
    # Pattern 1: Explicit "结论：X" / "Conclusion: X" etc. (highest priority)
    patterns = [
        r"(?:结论|Conclusion|Kesimpulan|สรุป)\s*[：:\s]+\s*(ไม่ใช่|ใช่|是|否|Yes|No|Ya|Tidak)",
        r"(?:结论|Conclusion|Kesimpulan|สรุป)\s*[：:\s]+.*?\b(Yes|No|是|否|Ya|Tidak|ใช่|ไม่ใช่)\b",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, t, re.I)
        if m:
            x = m.group(1).strip().lower()
            # Positive responses
            if x in ("是", "yes", "ya", "ใช่"):
                return "Yes"
            # Negative responses
            if x in ("否", "no", "tidak", "ไม่ใช่"):
                return "No"
    
    # Pattern 2: Language-specific fallback with context
    if lang == "zh":
        # Look for "是" or "否" near conclusion-related words
        if re.search(r"(?:结论|答案|判断)[：:\s]*(?:.*?)是\b", t):
            return "Yes"
        if re.search(r"(?:结论|答案|判断)[：:\s]*(?:.*?)否\b", t):
            return "No"
        # Last line check
        last_line = t.split('\n')[-1]
        if "是" in last_line and "否" not in last_line:
            return "Yes"
        if "否" in last_line:
            return "No"
    
    elif lang == "en":
        # Look for Yes/No in conclusion context
        if re.search(r"(?:conclusion|answer|judgment)[:\s]*(?:.*?)\byes\b", tl) and "no" not in tl:
            return "Yes"
        if re.search(r"(?:conclusion|answer|judgment)[:\s]*(?:.*?)\bno\b", tl):
            return "No"
        # Last line check
        last_line = t.split('\n')[-1].lower()
        if re.search(r"\byes\b", last_line) and not re.search(r"\bno\b", last_line):
            return "Yes"
        if re.search(r"\bno\b", last_line):
            return "No"
    
    elif lang == "ms":
        if re.search(r"(?:kesimpulan|jawapan)[:\s]*(?:.*?)\bya\b", tl) and "tidak" not in tl:
            return "Yes"
        if re.search(r"(?:kesimpulan|jawapan)[:\s]*(?:.*?)tidak", tl):
            return "No"
        last_line = t.split('\n')[-1].lower()
        if "ya" in last_line and "tidak" not in last_line:
            return "Yes"
        if "tidak" in last_line:
            return "No"
    
    elif lang == "th":
        if re.search(r"(?:สรุป|คำตอบ)[:\s]*(?:.*?)ใช่", t) and "ไม่ใช่" not in t:
            return "Yes"
        if re.search(r"(?:สรุป|คำตอบ)[:\s]*(?:.*?)ไม่ใช่", t):
            return "No"
        last_line = t.split('\n')[-1]
        if "ใช่" in last_line and "ไม่ใช่" not in last_line:
            return "Yes"
        if "ไม่ใช่" in last_line:
            return "No"
    
    # Pattern 3: Generic fallback (lowest priority)
    yes_count = len(re.findall(r"\b(?:是|yes|ya|ใช่)\b", tl))
    no_count = len(re.findall(r"\b(?:否|no|tidak|ไม่ใช่)\b", tl))
    
    if yes_count > no_count and yes_count > 0:
        logger.warning(f"Fuzzy match: parsed as 'Yes' (yes_count={yes_count}, no_count={no_count})")
        return "Yes"
    if no_count > yes_count and no_count > 0:
        logger.warning(f"Fuzzy match: parsed as 'No' (yes_count={yes_count}, no_count={no_count})")
        return "No"
    
    logger.warning(f"Failed to parse conclusion from text (lang={lang}): {t[:200]}...")
    return None


class DiagnosisRunner:
    """
    Multi-turn, multi-language LMM evaluation for re-intubation prediction.
    - Step 1: extraction (structured JSON); Step 2: prediction (Analysis + Conclusion) in same context.
    - Monolingual prompts per language (zh, en, ms, th).
    - Parallel execution via ThreadPoolExecutor; 3 rounds; output to PeisongData/result/response/{model_name}/{Language}/round{i}.json.
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
        image_dir: str = "PeisongData/dataset/data/figure",
    ):
        self.model = model
        raw = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        self.api_key = raw or None
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.max_workers = max_workers

        self.image_dir = self._resolve_path(image_dir, "image directory")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.language_mapping = {"zh": "chinese", "en": "english", "ms": "malay", "th": "thai"}
        self.prompts = get_medical_prompts()
        
        # Statistics tracking
        self.stats = {
            "total_tokens": 0,
            "total_cost_estimate": 0.0,
            "successful_samples": 0,
            "failed_samples": 0,
            "unparsed_conclusions": 0,
        }

        logger.info(f"Image directory: {self.image_dir} (exists: {self.image_dir.exists()})")
        logger.info(f"CWD: {Path.cwd()}")
        self._setup_logging_file()
    
    def _resolve_path(self, rel_path: str, description: str = "path") -> Path:
        """
        Resolve relative path by trying multiple root directories.
        Returns resolved absolute path.
        """
        path = Path(rel_path)
        
        # Already absolute
        if path.is_absolute():
            return path.resolve() if path.exists() else path
        
        # Try in order: current dir, workspace root, script parent
        search_roots = [
            Path.cwd(),
            Path("/mnt/data3/yuqian"),
            Path(__file__).parent.parent,
        ]
        
        for root in search_roots:
            potential = root / path
            if potential.exists():
                logger.info(f"Resolved {description}: {potential.resolve()}")
                return potential.resolve()
        
        # Not found, default to workspace root
        default = Path("/mnt/data3/yuqian") / path
        logger.warning(f"{description} not found, using default: {default}")
        return default

    def _setup_logging_file(self) -> None:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = sanitize_filename(self.model.replace("/", "_"))
        log_file = self.log_dir / f"lmm_eval_{sanitized_model}_{ts}.log"
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        logger.info(f"Logging initialized; file: {log_file}")

    def _find_image_files(self, language_code: str) -> List[Path]:
        lang_dir_name = self.language_mapping.get(language_code)
        if not lang_dir_name:
            return []
        lang_dir = self.image_dir / lang_dir_name
        if not lang_dir.exists():
            logger.warning(f"Language directory missing: {lang_dir}")
            return []
        out = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            out.extend(lang_dir.glob(ext))
        return sorted(out)

    def _extract_patient_id(self, image_path: Path) -> str:
        parts = image_path.stem.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else image_path.stem

    def _run_one_sample(
        self,
        image_path: Path,
        language_code: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Multi-turn: Step 1 (extraction) then Step 2 (prediction) in one context.
        Returns dict with step1_response, step2_response, analysis, conclusion, usage, timing.
        """
        if language_code not in self.prompts:
            raise ValueError(f"Unknown language: {language_code}")

        pd = self.prompts[language_code]
        system = pd["system"]
        step1_user = pd["step1_user"]
        step2_user = pd["step2_user"]

        image_b64 = encode_image_to_base64(str(image_path))
        mime = _mime_type_for_path(image_path)
        url = f"data:{mime};base64,{image_b64}"

        user1_content: List[Dict[str, Any]] = [
            {"type": "text", "text": step1_user},
            {"type": "image_url", "image_url": {"url": url}},
        ]

        def add_usage(u: Dict[str, int], tot: Dict[str, int]) -> None:
            for k in tot:
                tot[k] = tot.get(k, 0) + u.get(k, 0)

        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
            total_elapsed = 0.0
            
            try:
                # Rebuild messages each attempt so retries don't double-append
                msgs: List[Dict[str, Any]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user1_content},
                ]
                
                # Step 1: extraction
                step1_text, u1, e1, _ = call_lmm_multi_turn_api(
                    messages=msgs,
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=self.timeout_s,
                )
                add_usage(u1, total_usage)
                total_elapsed += e1
                msgs.append({"role": "assistant", "content": step1_text})
                msgs.append({"role": "user", "content": step2_user})

                # Step 2: prediction (Analysis + Conclusion)
                step2_text, u2, e2, raw2 = call_lmm_multi_turn_api(
                    messages=msgs,
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=self.timeout_s,
                )
                add_usage(u2, total_usage)
                total_elapsed += e2

                # Parse step2 response (now with structured JSON support)
                analysis, conclusion = parse_step2_response(step2_text, language_code)
                
                # Parse step1 into structured dict for direct use in result
                step1_structured = self._parse_step1_json(step1_text)
                step1_valid = step1_structured is not None
                
                # Track unparsed conclusions
                if conclusion is None:
                    logger.warning(f"Failed to parse conclusion for {image_path.name}")
                    with threading.Lock():
                        self.stats["unparsed_conclusions"] += 1

                logger.info(
                    f"Multi-turn OK | Tokens in/out/total: {total_usage['input_tokens']}/{total_usage['output_tokens']}/{total_usage['total_tokens']} | "
                    f"Time={total_elapsed:.2f}s | Step1_valid={step1_valid}"
                )

                return {
                    "step1_response": step1_text,
                    "step1_valid": step1_valid,
                    "step1_structured": step1_structured,
                    "step2_response": step2_text,
                    "Analysis": analysis,
                    "Conclusion": conclusion,
                    "conclusion_parsed": conclusion is not None,
                    "total_response_time_sec": round(total_elapsed, 3),
                    "input_tokens": total_usage["input_tokens"],
                    "output_tokens": total_usage["output_tokens"],
                    "total_tokens": total_usage["total_tokens"],
                    "reasoning_tokens": total_usage.get("reasoning_tokens", 0),
                    "raw_step2": raw2,
                    "attempts": attempt,
                }
                
            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"Authentication failed: {e}")
                raise
                
            except RateLimitError as e:
                last_error = e
                wait_time = (self.retry_backoff_base ** attempt) * 2  # Longer wait for rate limits
                logger.warning(f"Rate limit hit (attempt {attempt}/{self.max_retries}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                
            except TimeoutError as e:
                last_error = e
                logger.warning(f"Timeout (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.retry_backoff_base ** attempt)
                
            except APIError as e:
                last_error = e
                logger.error(f"API error (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(self.retry_backoff_base ** attempt)
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error (attempt {attempt}/{self.max_retries}): {e}")
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_base ** attempt)
        
        # Max retries exhausted
        raise Exception(f"Max retries ({self.max_retries}) exhausted. Last error: {last_error}")
    
    def _parse_step1_json(self, step1_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse Step 1 response into structured dict (patient_info, vital_signs, time_series_trends).
        Returns the parsed dict if valid, None otherwise.
        """
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", step1_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\}", step1_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return None

            data = json.loads(json_str)

            # Check required fields
            required_fields = ["patient_info", "vital_signs", "time_series_trends"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Step1 JSON missing required field: {field}")
                    return None

            if not isinstance(data.get("vital_signs"), list) or len(data["vital_signs"]) == 0:
                logger.warning("Step1 JSON: vital_signs should be a non-empty list")
                return None

            if not isinstance(data.get("time_series_trends"), dict):
                logger.warning("Step1 JSON: time_series_trends should be a dict")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Step1 JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Step1 parse error: {e}")
            return None

    def _validate_step1_json(self, step1_text: str) -> bool:
        """
        Validate that Step 1 response contains valid JSON with required fields.
        Returns True if valid, False otherwise.
        """
        return self._parse_step1_json(step1_text) is not None

    def _process_single_sample(
        self,
        image_path: Path,
        language_code: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        patient_id = self._extract_patient_id(image_path)
        try:
            res = self._run_one_sample(
                image_path=image_path,
                language_code=language_code,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Update statistics
            with threading.Lock():
                self.stats["successful_samples"] += 1
                self.stats["total_tokens"] += res.get("total_tokens", 0)
            
            return {
                "patient_id": patient_id,
                "image_path": str(image_path),
                "language": language_code,
                "result": res,
            }
        except AuthenticationError as e:
            logger.error(f"Authentication failed for {image_path}: {e}")
            raise  # Don't continue if auth fails
        except Exception as e:
            logger.error(f"Failed {image_path} ({language_code}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            with threading.Lock():
                self.stats["failed_samples"] += 1
            
            return None

    def process(
        self,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        rounds: int = 3,
        limit: Optional[int] = None,
        target_languages: Optional[List[str]] = None,
    ) -> None:
        all_languages = list(self.language_mapping.keys())
        if target_languages:
            s = set(target_languages)
            all_languages = [x for x in all_languages if x in s]
            logger.info(f"Target languages: {all_languages}")

        # Use _resolve_path for output directory
        base_output_dir = self._resolve_path("PeisongData/result/response", "output directory")
        sanitized_model = sanitize_filename(self.model.replace("/", "_"))
        output_dir = base_output_dir / sanitized_model
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.resolve()}")

        for rnd in range(1, int(rounds) + 1):
            logger.info(f"Round {rnd}/{rounds}")
            for language_code in all_languages:
                lang_name = self.language_mapping[language_code]
                logger.info(f"Round {rnd}: {lang_name} ({language_code})")

                image_files = self._find_image_files(language_code)
                if not image_files:
                    logger.warning(f"Round {rnd}: No images for {lang_name}")
                    (output_dir / lang_name.capitalize()).mkdir(parents=True, exist_ok=True)
                    continue

                lang_out = output_dir / lang_name.capitalize()
                lang_out.mkdir(parents=True, exist_ok=True)
                round_json = lang_out / f"round{rnd}.json"

                existing_results = {}
                if round_json.exists():
                    try:
                        with open(round_json, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        items = data if isinstance(data, list) else list(data.values())
                        for it in items:
                            k = (it.get("patient_id"), it.get("language"))
                            if k:
                                existing_results[k] = it
                        logger.info(f"Round {rnd}: Loaded {len(existing_results)} existing for {lang_name}")
                    except Exception as e:
                        logger.warning(f"Could not read {round_json}: {e}")

                tasks = []
                for im in image_files:
                    k = (self._extract_patient_id(im), language_code)
                    if k not in existing_results:
                        tasks.append((im, language_code))

                if limit and limit > 0:
                    tasks = tasks[:limit]
                    logger.info(f"Round {rnd}: Limited to {len(tasks)} tasks (limit={limit})")

                if not tasks:
                    logger.info(f"Round {rnd}: No new samples for {lang_name}")
                    continue

                results = list(existing_results.values())
                processed = skipped = errors = 0

                try:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                        f2task = {
                            ex.submit(self._process_single_sample, im, lc, temperature, max_tokens): im
                            for im, lc in tasks
                        }
                        for fut in tqdm(as_completed(f2task), total=len(tasks), desc=f"R{rnd} {lang_name}", unit="img"):
                            try:
                                r = fut.result()
                                if r:
                                    results.append(r)
                                    processed += 1
                                else:
                                    skipped += 1
                            except Exception as e:
                                errors += 1
                                logger.error(f"Task error: {e}")
                except KeyboardInterrupt:
                    logger.info(f"Round {rnd} {lang_name} interrupted; saving partial...")
                    with open(round_json, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    raise

                with open(round_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                # Calculate statistics for this round
                valid_results = [r for r in results if r.get("result")]
                avg_time = sum(r["result"]["total_response_time_sec"] for r in valid_results) / len(valid_results) if valid_results else 0
                parsed_count = sum(1 for r in valid_results if r["result"].get("conclusion_parsed", False))
                parse_rate = (parsed_count / len(valid_results) * 100) if valid_results else 0
                
                logger.info(
                    f"Round {rnd} {lang_name} done. {round_json} | "
                    f"processed={processed} skipped={skipped} errors={errors} total={len(results)} | "
                    f"avg_time={avg_time:.2f}s parse_rate={parse_rate:.1f}%"
                )

        # Print overall statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"All {rounds} rounds completed.")
        logger.info(f"Overall Statistics:")
        logger.info(f"  Successful samples: {self.stats['successful_samples']}")
        logger.info(f"  Failed samples: {self.stats['failed_samples']}")
        logger.info(f"  Unparsed conclusions: {self.stats['unparsed_conclusions']}")
        logger.info(f"  Total tokens used: {self.stats['total_tokens']:,}")
        if self.stats['successful_samples'] > 0:
            avg_tokens = self.stats['total_tokens'] / self.stats['successful_samples']
            logger.info(f"  Average tokens per sample: {avg_tokens:.1f}")
        logger.info(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-turn Multi-Language LMM Evaluation (re-intubation)")
    p.add_argument("--model", default=os.getenv("MODEL", "google/gemini-3-pro-preview"), help="Model name")
    p.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    p.add_argument("--base-url", default=os.getenv("BASE_URL", "https://openrouter.ai/api/v1"), help="API base URL")
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None)
    p.add_argument("--timeout", type=float, default=float(os.getenv("TIMEOUT", "120.0")))
    p.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    p.add_argument("--retry-backoff-base", type=float, default=float(os.getenv("RETRY_BACKOFF_BASE", "1.5")))
    p.add_argument("--log-dir", default=os.getenv("LOG_DIR", "logs"))
    p.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "10")))
    p.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "3")))
    p.add_argument("--limit", type=int, default=int(os.getenv("LIMIT")) if os.getenv("LIMIT") else None)
    p.add_argument("--target-languages", nargs="+", default=os.getenv("TARGET_LANGUAGES", "").split() or None)
    p.add_argument("--image-dir", default=os.getenv("IMAGE_DIR", "PeisongData/dataset/data/figure"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.api_key = (args.api_key or os.getenv("OPENAI_API_KEY") or "").strip() or None
    if not args.api_key:
        raise ValueError("Provide --api_key or OPENAI_API_KEY")

    # 启动时打码打印，便于确认 key 已正确加载（401 常因 key 未传或无效）
    _k = args.api_key
    _mask = f"sk-...{_k[-4:]}" if len(_k) > 4 else "***"
    logger.info("API key configured: %s | base_url=%s", _mask, args.base_url)

    runner = DiagnosisRunner(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff_base,
        log_dir=args.log_dir,
        max_workers=args.max_workers,
        image_dir=args.image_dir,
    )
    try:
        runner.process(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rounds=args.rounds,
            limit=args.limit,
            target_languages=args.target_languages,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
        raise SystemExit(0)
