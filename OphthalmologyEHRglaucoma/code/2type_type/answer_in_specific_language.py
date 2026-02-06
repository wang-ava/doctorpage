#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import logging
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TypedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ----------------- Type Definitions -----------------
class GroundTruth(TypedDict):
    diagnosis_code: Optional[int]
    diagnosis_label: str


class Measurements(TypedDict):
    Age: Any
    Gender: str
    IOP: Any
    CCT: Any
    Average_RNFL_Thickness: Any
    ACD: Any
    Visual_Field_MD: Any
    Visual_Field_PSD: Any


# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# -------------------------------------------


# ----------------- Language Configuration Module -----------------
class LanguageConfig:
    """Centralized language configuration for labels and translations"""
    
    SUPPORTED_LANGUAGES = ["English", "Chinese", "Malay", "Thai"]
    
    @staticmethod
    @lru_cache(maxsize=4)
    def get_section_display_map(language: str) -> Dict[str, str]:
        """Get section display labels for a specific language"""
        base = {
            "Chief Complaint": "Chief Complaint",
            "History of Present Illness": "History of Present Illness",
            "Past Medical History": "Past Medical History",
            "Physical Examination": "Physical Examination",
        }

        language_maps = {
            "Chinese": {
                "Chief Complaint": "主诉",
                "History of Present Illness": "现病史",
                "Past Medical History": "既往史",
                "Physical Examination": "体格检查",
            },
            "Malay": {
                "Chief Complaint": "Aduan Utama",
                "History of Present Illness": "Sejarah Penyakit Semasa",
                "Past Medical History": "Sejarah Perubatan Lepas",
                "Physical Examination": "Pemeriksaan Fizikal",
            },
            "Thai": {
                "Chief Complaint": "อาการสำคัญ",
                "History of Present Illness": "ประวัติป่วยปัจจุบัน",
                "Past Medical History": "ประวัติการเจ็บป่วยในอดีต",
                "Physical Examination": "การตรวจร่างกาย",
            }
        }
        
        if language in language_maps:
            return {**base, **language_maps[language]}
        return base
    
    @staticmethod
    @lru_cache(maxsize=4)
    def get_measurement_labels(language: str) -> Dict[str, str]:
        """Get measurement labels for a specific language"""
        labels = {
            "English": {
                "Age": "Age",
                "Gender": "Gender",
                "IOP": "IOP (Intraocular Pressure, mmHg)",
                "CCT": "CCT (Central Corneal Thickness, mm)",
                "Average RNFL Thickness": "Average RNFL Thickness (μm)",
                "ACD": "ACD (Anterior Chamber Depth, mm)",
                "Visual Field MD": "Visual Field MD (Mean Deviation, dB)",
                "Visual Field PSD": "Visual Field PSD (Pattern Standard Deviation, dB)",
            },
            "Chinese": {
                "Age": "年龄",
                "Gender": "性别",
                "IOP": "眼压 (毫米汞柱)",
                "CCT": "中央角膜厚度 (毫米)",
                "Average RNFL Thickness": "平均视网膜神经纤维层厚度 (微米)",
                "ACD": "前房深度 (毫米)",
                "Visual Field MD": "视野平均偏差 (分贝)",
                "Visual Field PSD": "视野模式标准差 (分贝)",
            },
            "Malay": {
                "Age": "Umur",
                "Gender": "Jantina",
                "IOP": "Tekanan Intraokular (milimeter merkuri)",
                "CCT": "Ketebalan Kornea Pusat (milimeter)",
                "Average RNFL Thickness": "Ketebalan Lapisan Serat Saraf Retina Purata (mikrometer)",
                "ACD": "Kedalaman Ruang Anterior (milimeter)",
                "Visual Field MD": "Medan Visual Penyimpangan Purata (desibel)",
                "Visual Field PSD": "Medan Visual Sisihan Piawai Corak (desibel)",
            },
            "Thai": {
                "Age": "อายุ",
                "Gender": "เพศ",
                "IOP": "ความดันลูกตา (มิลลิเมตรปรอท)",
                "CCT": "ความหนาของกระจกตาส่วนกลาง (มิลลิเมตร)",
                "Average RNFL Thickness": "ความหนาเฉลี่ยของชั้นเส้นใยประสาทจอตา (ไมโครเมตร)",
                "ACD": "ความลึกของช่องหน้าลูกตา (มิลลิเมตร)",
                "Visual Field MD": "ลานสายตาค่าเบี่ยงเบนเฉลี่ย (เดซิเบล)",
                "Visual Field PSD": "ลานสายตาค่าเบี่ยงเบนมาตรฐานแบบ (เดซิเบล)",
            }
        }
        return labels.get(language, labels["English"])
    
    @staticmethod
    @lru_cache(maxsize=4)
    def get_ui_text(language: str) -> Dict[str, str]:
        """Get UI text for a specific language"""
        ui_texts = {
            "English": {
                "right_eye": "**Right Eye:**",
                "left_eye": "**Left Eye:**",
                "no_measurements": "No measurements available",
                "textual_summary": "=== Textual Case Summary ===",
                "structured_measurements": "=== Structured Clinical Measurements ==="
            },
            "Chinese": {
                "right_eye": "**右眼:**",
                "left_eye": "**左眼:**",
                "no_measurements": "无可用测量数据",
                "textual_summary": "=== 文本病例摘要 ===",
                "structured_measurements": "=== 结构化临床测量 ==="
            },
            "Malay": {
                "right_eye": "**Mata Kanan:**",
                "left_eye": "**Mata Kiri:**",
                "no_measurements": "Tiada pengukuran tersedia",
                "textual_summary": "=== Ringkasan Kes Teks ===",
                "structured_measurements": "=== Pengukuran Klinikal Berstruktur ==="
            },
            "Thai": {
                "right_eye": "**ตาขวา:**",
                "left_eye": "**ตาซ้าย:**",
                "no_measurements": "ไม่มีข้อมูลการวัด",
                "textual_summary": "=== สรุปเคสแบบข้อความ ===",
                "structured_measurements": "=== การวัดทางคลินิกแบบมีโครงสร้าง ==="
            }
        }
        return ui_texts.get(language, ui_texts["English"])


# ----------------- Utility Functions -----------------
def sanitize_filename(filename: str) -> str:
    """Replace invalid filename characters with underscores."""
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    invalid_chars = r'[<>:"\\|?*]'
    sanitized = re.sub(invalid_chars, '_', basename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return os.path.join(directory, sanitized) if directory else sanitized


def validate_diagnosis_code(code: Any) -> Optional[int]:
    """
    Validate diagnosis code is in valid range [0-6].
    
    Args:
        code: The diagnosis code to validate
        
    Returns:
        Valid integer code or None if invalid
    """
    try:
        code_int = int(code)
        if 0 <= code_int <= 6:
            return code_int
        else:
            logger.warning(f"Invalid diagnosis code: {code}, not in range [0,6]")
            return None
    except (ValueError, TypeError):
        logger.warning(f"Cannot convert diagnosis code to integer: {code}")
        return None


def validate_language(language: str) -> str:
    """
    Validate language is supported.
    
    Args:
        language: Language name to validate
        
    Returns:
        Validated language name
        
    Raises:
        ValueError: If language is not supported
    """
    if language not in LanguageConfig.SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Invalid language: {language}. "
            f"Must be one of {LanguageConfig.SUPPORTED_LANGUAGES}"
        )
    return language


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON string from text, handling markdown code blocks.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
    
    # Try to find JSON in markdown code block
    json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    
    # Try to find JSON object directly
    brace_pattern = r'\{[\s\S]*\}'
    match = re.search(brace_pattern, text)
    if match:
        return match.group(0).strip()
    
    return text.strip()


def parse_model_response(response_text: str, language: str) -> Dict[str, Any]:
    """
    Parse model response JSON, keeping original field names.
    
    Args:
        response_text: Raw model response text
        language: Language of the response (not used, kept for compatibility)
        
    Returns:
        Dictionary with parsed response, or error info if parsing failed
    """
    result = {
        "parse_success": False,
        "parse_error": None,
        "parsed_response": {}
    }
    
    try:
        # Extract JSON from response
        json_str = extract_json_from_text(response_text)
        if not json_str:
            result["parse_error"] = "No JSON found in response"
            return result
        
        # Parse JSON - keep original structure and field names
        parsed = json.loads(json_str)
        
        result["parse_success"] = True
        result["parsed_response"] = parsed
        
    except json.JSONDecodeError as e:
        result["parse_error"] = f"JSON decode error: {str(e)}"
        logger.warning(f"Failed to parse JSON response: {e}")
    except Exception as e:
        result["parse_error"] = f"Parsing error: {str(e)}"
        logger.warning(f"Error parsing model response: {e}")
    
    return result


# ----------------- Main Class -----------------
class GlaucomaDiagnosisRunner:
    """
    Multi-language glaucoma diagnosis runner (2 modalities, no images).
    - Merges textual case summaries with structured clinical measurements
    - Processes in multiple languages (English, Chinese, Malay, Thai)
    - Saves results in directory structure: response_2type/{model_name}/{Language}/round{1-3}.json

    NOTE (language consistency):
    - Translated JSON keeps English section keys (e.g., "Chief Complaint"), but content is localized.
    - To keep prompts language-consistent, we localize DISPLAY labels of sections based on target language.
    """

    # Constants
    DIAGNOSIS_CODE_RANGE = (0, 6)
    DEFAULT_CHECKPOINT_INTERVAL = 5
    
    def __init__(
        self,
        model: str = "google/gemini-3-pro-preview",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 120.0,
        max_retries: int = 5,
        retry_backoff_base: float = 1.5,
        log_dir: str = "logs",
        max_workers: int = 10,
        case_dir: str = "OphthalmologyEHRglaucoma/dataset/case",
        translate_dir: str = "OphthalmologyEHRglaucoma/result/translate/google_gemini-3-pro-preview",
        csv_path: str = "OphthalmologyEHRglaucoma/dataset/patient-state-diagnosis.csv",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set via --api_key argument or OPENAI_API_KEY environment variable."
            )
        
        # Use default OpenAI client (same as 3type_data) to avoid connection pool
        # issues that can cause hangs with high max_workers under rate limiting.
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.max_workers = max_workers

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data paths
        self.case_dir = Path(case_dir)
        self.translate_dir = Path(translate_dir)
        self.csv_path = Path(csv_path)

        # Validate paths exist
        self._validate_paths()

        # Thread safety - use RLock to allow nested locking (e.g., checkpoint during write)
        self._lock = threading.RLock()

        # Load CSV data
        self.csv_data = self._load_csv_data()

        self._setup_logging_file()

    def _validate_paths(self) -> None:
        """Validate that required paths exist"""
        if not self.case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Note: translate_dir may not exist for English-only processing
        logger.debug(f"Validated paths - case_dir: {self.case_dir}, csv_path: {self.csv_path}")

    # ----------------- PROMPTS -----------------
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_prompt_template(language: str) -> str:
        """
        Get prompt template for specified language.
        
        Prompt must stay consistent with the 3-type runner:
        - Same output fields & ordering:
          diagnosis_label, diagnosis_code, prescription_medication, treatment_plan, confidence, rationale
        - Only difference: this runner has NO images, so modalities list & rationale guidance exclude fundus findings.
        """
        prompts = {
            "English": """You are an ophthalmologist specializing in glaucoma. For each patient you will receive two modalities in this order: (1) a textual case summary with ground-truth labels removed, and (2) structured per-eye clinical measurements. Using ALL modalities, predict for BOTH eyes:
1. diagnosis_label: free-text clinical diagnosis (may include non-glaucoma findings).
2. diagnosis_code: choose one code from {0=Not glaucoma, 1=PACS, 2=PAC, 3=PACG, 4=APAC, 5=POAG, 6=Secondary glaucoma}.
3. prescription_medication: regimen or "None".
4. treatment_plan: recommended procedure / follow-up or "None". Consider whether surgical intervention is indicated, but do not add a separate surgery_procedures field.
5. confidence: number in [0,1].
6. rationale: <=300 words citing key tabular metrics (e.g., IOP, OCT/visual field metrics where present) plus case-text cues.

Please provide your response in the following JSON format:
{
  "right_eye": {
    "diagnosis_label": "string",
    "diagnosis_code": 0-6,
    "prescription_medication": "string or None",
    "treatment_plan": "string or None",
    "confidence": 0.0-1.0,
    "rationale": "string (max 300 words)"
  },
  "left_eye": {
    "diagnosis_label": "string",
    "diagnosis_code": 0-6,
    "prescription_medication": "string or None",
    "treatment_plan": "string or None",
    "confidence": 0.0-1.0,
    "rationale": "string (max 300 words)"
  }
}""",

            "Chinese": """您是一位专门研究青光眼的眼科医生。对于每位患者，您将按以下顺序收到两种模态的数据：(1) 已移除真实标签的文本病例摘要，(2) 每只眼睛的结构化临床测量数据。使用所有模态，为双眼进行预测：
1. 诊断标签：自由文本的临床诊断（可能包括非青光眼发现）。
2. 诊断代码：从 {0=非青光眼, 1=原发性房角关闭疑似, 2=原发性房角关闭, 3=原发性闭角型青光眼, 4=急性原发性闭角型青光眼, 5=原发性开角型青光眼, 6=继发性青光眼} 中选择一个代码。
3. 处方药物：用药方案或"无"。
4. 治疗计划：推荐的程序或随访或"无"。考虑是否需要手术干预，但不要添加单独的手术程序字段。
5. 置信度：[0,1]范围内的数字。
6. 理由：不超过300字，引用关键的表格指标（例如，眼压、光学相干断层扫描或视野指标（如果存在））以及病例文本线索。

请按以下JSON格式提供您的回答：
{
  "右眼": {
    "诊断标签": "字符串",
    "诊断代码": 0-6之间的整数,
    "处方药物": "字符串或无",
    "治疗计划": "字符串或无",
    "置信度": 0.0-1.0之间的数字,
    "理由": "字符串（最多300字）"
  },
  "左眼": {
    "诊断标签": "字符串",
    "诊断代码": 0-6之间的整数,
    "处方药物": "字符串或无",
    "治疗计划": "字符串或无",
    "置信度": 0.0-1.0之间的数字,
    "理由": "字符串（最多300字）"
  }
}""",

            "Malay": """Anda adalah pakar oftalmologi yang pakar dalam glaukoma. Bagi setiap pesakit, anda akan menerima dua modaliti mengikut susunan berikut: (1) ringkasan kes teks dengan label kebenaran sebenar dibuang, dan (2) pengukuran klinikal berstruktur untuk setiap mata. Menggunakan SEMUA modaliti, ramalkan untuk KEDUA-DUA mata:
1. label diagnosis: diagnosis klinikal teks bebas (boleh termasuk penemuan bukan glaukoma).
2. kod diagnosis: pilih satu kod daripada {0=Bukan glaukoma, 1=Suspek Penutupan Sudut Primer, 2=Penutupan Sudut Primer, 3=Glaukoma Sudut Tertutup Primer, 4=Glaukoma Sudut Tertutup Primer Akut, 5=Glaukoma Sudut Terbuka Primer, 6=Glaukoma sekunder}.
3. ubat preskripsi: rejimen atau "Tiada".
4. pelan rawatan: prosedur yang disyorkan atau susulan atau "Tiada". Pertimbangkan sama ada campur tangan pembedahan ditunjukkan, tetapi jangan tambah medan prosedur pembedahan yang berasingan.
5. tahap keyakinan: nombor dalam [0,1].
6. rasional: tidak lebih daripada 300 perkataan yang memetik metrik jadual utama (cth., Tekanan Intraokular, metrik Tomografi Koheren Optik atau medan visual jika ada) ditambah petunjuk teks kes.

Sila berikan respons anda dalam format JSON berikut:
{
  "mata_kanan": {
    "label_diagnosis": "string",
    "kod_diagnosis": 0-6,
    "ubat_preskripsi": "string atau Tiada",
    "pelan_rawatan": "string atau Tiada",
    "tahap_keyakinan": 0.0-1.0,
    "rasional": "string (maksimum 300 perkataan)"
  },
  "mata_kiri": {
    "label_diagnosis": "string",
    "kod_diagnosis": 0-6,
    "ubat_preskripsi": "string atau Tiada",
    "pelan_rawatan": "string atau Tiada",
    "tahap_keyakinan": 0.0-1.0,
    "rasional": "string (maksimum 300 perkataan)"
  }
}""",

            "Thai": """คุณเป็นจักษุแพทย์ผู้เชี่ยวชาญด้านต้อหิน สำหรับผู้ป่วยแต่ละราย คุณจะได้รับข้อมูลสองรูปแบบตามลำดับนี้: (1) สรุปเคสแบบข้อความที่ลบป้ายกำกับความจริงออกแล้ว (2) การวัดทางคลินิกแบบมีโครงสร้างสำหรับแต่ละตา ใช้ข้อมูลทั้งสองรูปแบบ ทำนายสำหรับทั้งสองตา:
1. ป้ายกำกับการวินิจฉัย: การวินิจฉัยทางคลินิกแบบข้อความอิสระ (อาจรวมถึงการค้นพบที่ไม่ใช่ต้อหิน)
2. รหัสการวินิจฉัย: เลือกหนึ่งรหัสจาก {0=ไม่ใช่ต้อหิน, 1=สงสัยมุมปิดปฐมภูมิ, 2=มุมปิดปฐมภูมิ, 3=ต้อหินมุมปิดปฐมภูมิ, 4=ต้อหินมุมปิดปฐมภูมิเฉียบพลัน, 5=ต้อหินมุมเปิดปฐมภูมิ, 6=ต้อหินทุติยภูมิ}
3. ยาที่สั่งจ่าย: แผนการรักษาหรือ "ไม่มี"
4. แผนการรักษา: ขั้นตอนที่แนะนำหรือติดตามผลหรือ "ไม่มี" พิจารณาว่าจำเป็นต้องมีการผ่าตัดหรือไม่ แต่ไม่ต้องเพิ่มฟิลด์ขั้นตอนการผ่าตัดแยก
5. ความมั่นใจ: ตัวเลขในช่วง [0,1]
6. เหตุผล: ไม่เกิน 300 คำ อ้างอิงถึงตัวชี้วัดตารางหลัก (เช่น ความดันลูกตา, ตัวชี้วัดการสแกนด้วยแสงหรือลานสายตาหากมี) รวมถึงคำใบ้จากข้อความเคส

กรุณาให้คำตอบในรูปแบบ JSON ดังนี้:
{
  "ตาขวา": {
    "ป้ายกำกับการวินิจฉัย": "string",
    "รหัสการวินิจฉัย": 0-6,
    "ยาที่สั่งจ่าย": "string หรือ ไม่มี",
    "แผนการรักษา": "string หรือ ไม่มี",
    "ความมั่นใจ": 0.0-1.0,
    "เหตุผล": "string (ไม่เกิน 300 คำ)"
  },
  "ตาซ้าย": {
    "ป้ายกำกับการวินิจฉัย": "string",
    "รหัสการวินิจฉัย": 0-6,
    "ยาที่สั่งจ่าย": "string หรือ ไม่มี",
    "แผนการรักษา": "string หรือ ไม่มี",
    "ความมั่นใจ": 0.0-1.0,
    "เหตุผล": "string (ไม่เกิน 300 คำ)"
  }
}"""
        }
        return prompts.get(language, prompts["English"])

    def _setup_logging_file(self) -> None:
        """Configure logging with timestamp-based filename."""
        # Remove existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_model = sanitize_filename(self.model.replace('/', '_'))
        log_file = self.log_dir / f"glaucoma_2type_{sanitized_model}_{ts}.log"
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        logger.propagate = False
        logger.info(f"Logging initialized; file: {log_file}")

    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load CSV data with structured measurements.
        
        Returns:
            DataFrame with patient data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is empty or has invalid format
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path, dtype={'Patient ID': str})
            
            if len(df) == 0:
                raise ValueError(f"CSV file is empty: {self.csv_path}")
            
            # Validate required columns
            required_columns = ['Patient ID']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")
            
            logger.info(f"Loaded {len(df)} rows from CSV file: {self.csv_path}")
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty or invalid: {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def _load_case_file(self, patient_id: str, language: str) -> Dict[str, Any]:
        """
        Load case file for a specific patient and language.
        
        Args:
            patient_id: Patient ID
            language: Target language
            
        Returns:
            Case data dictionary
            
        Raises:
            FileNotFoundError: If case file doesn't exist
            json.JSONDecodeError: If case file has invalid JSON
        """
        # Validate language first
        validate_language(language)
        
        if language == "English":
            case_file = self.case_dir / "1Done_json" / f"{patient_id}.json"
        else:
            lang_dir_map = {"Chinese": "Chinese", "Malay": "Malay", "Thai": "Thai"}
            case_file = self.translate_dir / lang_dir_map[language] / f"{patient_id}.json"

        if not case_file.exists():
            raise FileNotFoundError(
                f"Case file not found for patient {patient_id}, language {language}: {case_file}"
            )

        try:
            with open(case_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate case data has required structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid case file format for {patient_id}: expected dict")
                
            logger.debug(f"Loaded case file: {case_file}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in case file {case_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading case file {case_file}: {e}")
            raise

    @staticmethod
    def _strip_translation_prefix(text: str) -> str:
        """Remove common translation instruction prefixes from text"""
        if not text:
            return text
        text = re.sub(r'^请提供完整的中文翻译：\s*\n\s*', '', str(text))
        text = re.sub(r'^Please provide.*?:\s*\n\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_textual_summary(self, case_data: Dict[str, Any], language: str) -> str:
        """
        Extract and concatenate textual summary, excluding Diagnosis section, with localized display labels.
        
        Args:
            case_data: Case data dictionary
            language: Target language for display labels
            
        Returns:
            Formatted textual summary
        """
        sections = case_data.get("sections", {})

        field_order = [
            "Chief Complaint",
            "History of Present Illness",
            "Past Medical History",
            "Physical Examination"
        ]

        display_map = LanguageConfig.get_section_display_map(language)

        parts = []
        for field in field_order:
            if field in sections:
                content = self._strip_translation_prefix(str(sections[field]).strip())
                if content:
                    display_label = display_map.get(field, field)
                    parts.append(f"{display_label}:\n{content}")

        return "\n\n".join(parts)

    def _extract_structured_measurements(self, patient_id: str) -> Dict[str, Measurements]:
        """
        Extract structured measurements for both eyes from CSV.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Dictionary with 'OD' and 'OS' keys containing measurements
        """
        patient_rows = self.csv_data[self.csv_data['Patient ID'].astype(str) == str(patient_id)]

        if len(patient_rows) == 0:
            logger.warning(f"No CSV data found for patient {patient_id}")
            return {"OD": {}, "OS": {}}

        measurements: Dict[str, Measurements] = {"OD": {}, "OS": {}}

        # Demographics (same for both eyes)
        first_row = patient_rows.iloc[0]
        age = first_row.get("Age", "")
        gender_map = {1: "Male", 2: "Female"}
        gender_val = first_row.get("Gender (1=Male, 2=Female)", "")
        
        try:
            gender = gender_map.get(int(gender_val) if pd.notna(gender_val) else 0, "")
        except (ValueError, TypeError):
            gender = ""
            logger.debug(f"Invalid gender value for patient {patient_id}: {gender_val}")

        measurements["OD"]["Age"] = age
        measurements["OS"]["Age"] = age
        measurements["OD"]["Gender"] = gender
        measurements["OS"]["Gender"] = gender

        # Per-eye measurements
        for _, row in patient_rows.iterrows():
            eye_code = row.get("Included Eye (1=Right,2=Left)", "")
            
            try:
                eye_code_int = int(eye_code) if pd.notna(eye_code) else 0
                if eye_code_int == 1:
                    eye = "OD"
                elif eye_code_int == 2:
                    eye = "OS"
                else:
                    logger.debug(f"Invalid eye code for patient {patient_id}: {eye_code}")
                    continue
            except (ValueError, TypeError):
                logger.debug(f"Cannot parse eye code for patient {patient_id}: {eye_code}")
                continue

            measurements[eye]["IOP"] = row.get("Intraocular Pressure", "")
            measurements[eye]["CCT"] = row.get("Central Corneal Thickness (mm)", "")
            measurements[eye]["Average RNFL Thickness"] = row.get("Average RNFL Thickness (μm)", "")
            measurements[eye]["ACD"] = row.get("Central Anterior Chamber Depth (mm)", "")
            measurements[eye]["Visual Field MD"] = row.get("MD24-2 (dB)", "")
            measurements[eye]["Visual Field PSD"] = row.get("PSD24-2 (dB)", "")

        return measurements

    def _extract_ground_truth(self, patient_id: str) -> Dict[str, GroundTruth]:
        """
        Extract ground truth diagnosis for both eyes from CSV (for evaluation; NOT included in prompt).
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Dictionary with 'OD' and 'OS' keys containing ground truth
        """
        patient_rows = self.csv_data[self.csv_data['Patient ID'].astype(str) == str(patient_id)]
        ground_truth: Dict[str, GroundTruth] = {"OD": {}, "OS": {}}

        for _, row in patient_rows.iterrows():
            eye_code = row.get("Included Eye (1=Right,2=Left)", "")
            
            try:
                eye_code_int = int(eye_code) if pd.notna(eye_code) else 0
                if eye_code_int == 1:
                    eye = "OD"
                elif eye_code_int == 2:
                    eye = "OS"
                else:
                    continue
            except (ValueError, TypeError):
                continue

            # Validate and extract diagnosis code
            diagnosis_code_raw = row.get(
                "Diagnosis（1=PACS, 2=PAC, 3=PACG, 4=APAC, 5=POAG, 6=Secondary glaucoma, 0=Not glaucoma）",
                ""
            )
            ground_truth[eye]["diagnosis_code"] = validate_diagnosis_code(diagnosis_code_raw)

            # Extract diagnosis label
            diagnosis_label = row.get("Right Eye Diagnosis", "") if eye == "OD" else row.get("Left Eye Diagnosis", "")
            ground_truth[eye]["diagnosis_label"] = str(diagnosis_label) if pd.notna(diagnosis_label) else ""

        return ground_truth

    def _format_structured_measurements(
        self, 
        measurements: Dict[str, Measurements], 
        language: str
    ) -> str:
        """
        Format structured measurements as text for the prompt, in target language.
        
        Args:
            measurements: Measurements dictionary for both eyes
            language: Target language
            
        Returns:
            Formatted measurement text
        """
        labels = LanguageConfig.get_measurement_labels(language)
        ui_text = LanguageConfig.get_ui_text(language)

        lines = []
        
        # Right eye
        lines.append(ui_text["right_eye"])
        od_lines = []
        for key, label in labels.items():
            value = measurements["OD"].get(key, '')
            if value != '' and pd.notna(value):
                od_lines.append(f"{label}: {value}")
        lines.append("\n".join(od_lines) if od_lines else ui_text["no_measurements"])

        lines.append("")
        
        # Left eye
        lines.append(ui_text["left_eye"])
        os_lines = []
        for key, label in labels.items():
            value = measurements["OS"].get(key, '')
            if value != '' and pd.notna(value):
                os_lines.append(f"{label}: {value}")
        lines.append("\n".join(os_lines) if os_lines else ui_text["no_measurements"])

        return "\n".join(lines)

    def _build_prompt_messages(
        self, 
        textual_summary: str, 
        structured_measurements: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Build system+user messages.
        
        Args:
            textual_summary: Formatted textual case summary
            structured_measurements: Formatted measurements
            language: Target language
            
        Returns:
            List of message dictionaries
        """
        system_prompt = self._get_prompt_template(language)
        ui_text = LanguageConfig.get_ui_text(language)

        user_content_parts = [
            ui_text["textual_summary"],
            textual_summary,
            "",
            ui_text["structured_measurements"],
            structured_measurements
        ]

        user_content = "\n".join(user_content_parts)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _chat_call(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Tuple[str, Dict[str, int], float, Dict[str, Any]]:
        """
        Single chat.completions call with retries.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (content, usage_dict, elapsed_time, raw_response)
            
        Raises:
            Exception: If all retries fail
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
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

                resp_dict = resp.model_dump()
                raw_usage = resp_dict.get("usage") or getattr(resp, "usage", None)
                usage_map = self._coerce_usage(raw_usage)

                return content, usage_map, elapsed, resp_dict

            except Exception as e:
                last_exception = e
                logger.warning(f"API error attempt {attempt}/{self.max_retries}: {e}")
                
                if attempt < self.max_retries:
                    backoff_time = self.retry_backoff_base ** attempt
                    logger.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
        
        # All retries failed
        logger.error(f"All {self.max_retries} API retry attempts failed")
        raise last_exception

    @staticmethod
    def _coerce_usage(usage_obj: Any) -> Dict[str, int]:
        """
        Normalize usage object to dictionary.
        
        Args:
            usage_obj: Usage object from API response
            
        Returns:
            Normalized usage dictionary with standard keys
        """
        default_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0
        }
        
        if usage_obj is None:
            return default_usage

        try:
            # Handle dict-like objects
            if isinstance(usage_obj, dict):
                ud = usage_obj
            elif hasattr(usage_obj, "model_dump"):
                ud = usage_obj.model_dump()
            else:
                ud = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                }
            
            # Extract tokens with fallbacks
            it = ud.get("prompt_tokens") or ud.get("input_tokens") or 0
            ot = ud.get("completion_tokens") or ud.get("output_tokens") or 0
            tt = ud.get("total_tokens") or (it + ot)

            # Extract reasoning tokens if available
            reasoning_tokens = 0
            completion_details = ud.get("completion_tokens_details", {})
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens") or 0

            return {
                "input_tokens": int(it or 0),
                "output_tokens": int(ot or 0),
                "total_tokens": int(tt or 0),
                "reasoning_tokens": int(reasoning_tokens or 0)
            }
            
        except Exception as e:
            logger.warning(f"Error parsing usage object: {e}")
            return default_usage

    def _run_one_sample(
        self,
        patient_id: str,
        language: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process one patient sample for one language.
        
        Args:
            patient_id: Patient ID
            language: Target language
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Result dictionary with prediction and metadata
            
        Raises:
            Various exceptions if processing fails
        """
        # Load case data
        case_data = self._load_case_file(patient_id, language)

        # Extract textual summary
        textual_summary = self._extract_textual_summary(case_data, language)

        # Extract structured measurements
        measurements = self._extract_structured_measurements(patient_id)
        structured_measurements = self._format_structured_measurements(measurements, language)

        # Extract ground truth
        ground_truth = self._extract_ground_truth(patient_id)

        # Build prompt
        messages = self._build_prompt_messages(
            textual_summary=textual_summary,
            structured_measurements=structured_measurements,
            language=language
        )

        # Call API
        reply, usage, elapsed, raw = self._chat_call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Log with appropriate level
        reasoning_info = ""
        if usage.get("reasoning_tokens", 0) > 0:
            reasoning_info = f" (reasoning: {usage['reasoning_tokens']})"
        
        logger.debug(
            f"API call completed | Patient {patient_id}, {language} | "
            f"Tokens in/out/total: {usage['input_tokens']}/{usage['output_tokens']}{reasoning_info}/{usage['total_tokens']} | "
            f"Time={elapsed:.2f}s"
        )

        # Parse model response
        parse_result = parse_model_response(reply, language)
        
        if parse_result["parse_success"]:
            logger.debug(f"Successfully parsed response for patient {patient_id}")
        else:
            logger.warning(
                f"Failed to parse response for patient {patient_id}: "
                f"{parse_result['parse_error']}"
            )

        return {
            "patient_id": patient_id,
            "language": language,
            "input": {
                "textual_summary": textual_summary,
                "structured_measurements": measurements,
            },
            "output": {
                "model_response_raw": reply,
                "parsed_response": parse_result["parsed_response"],
                "parse_success": parse_result["parse_success"],
                "parse_error": parse_result["parse_error"],
            },
            "ground_truth": ground_truth,
            "metadata": {
                "total_response_time_sec": round(elapsed, 3),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            },
            "raw_response": raw,
        }

    def _process_single_sample(
        self,
        patient_id: str,
        language: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Process single sample with error handling.
        
        Args:
            patient_id: Patient ID
            language: Target language
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Result dictionary or None if processing fails
        """
        try:
            return self._run_one_sample(
                patient_id=patient_id,
                language=language,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}, language {language}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_all_patient_ids(self) -> List[str]:
        """
        Get all patient IDs from English case directory.
        
        Returns:
            Sorted list of patient IDs
        """
        patient_ids = set()
        case_json_dir = self.case_dir / "1Done_json"
        
        if not case_json_dir.exists():
            logger.warning(f"Case JSON directory not found: {case_json_dir}")
            return []
        
        for case_file in case_json_dir.glob("*.json"):
            patient_ids.add(case_file.stem)
        
        sorted_ids = sorted(list(patient_ids))
        logger.debug(f"Found {len(sorted_ids)} patient IDs in {case_json_dir}")
        return sorted_ids

    def _create_checkpoint(
        self,
        output_file: Path,
        results: List[Dict[str, Any]],
        round_num: int,
        language: str
    ) -> None:
        """
        Create checkpoint with metadata.
        
        Args:
            output_file: Output file path
            results: List of result dictionaries
            round_num: Current round number
            language: Current language
        """
        checkpoint_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "round": round_num,
                "language": language,
                "total_results": len(results),
                "model": self.model
            },
            "results": results
        }
        
        self._save_results(output_file, checkpoint_data)
        logger.info(
            f"Checkpoint created: Round {round_num}, {language}, "
            f"{len(results)} results at {output_file}"
        )

    def process(
        self,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        rounds: int = 3,
        limit: Optional[int] = None,
        target_language: Optional[str] = None,
        target_languages: Optional[str] = None,
        output_base_dir: str = "OphthalmologyEHRglaucoma/result/response_2type",
    ) -> None:
        """
        Process multi-language glaucoma diagnosis task.
        
        Output path: {output_base_dir}/{model_name}/{Language}/round{1-3}.json
        
        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            checkpoint_interval: Save checkpoint every N results
            rounds: Number of rounds to process
            limit: Limit number of patients (for testing)
            target_language: Single target language
            target_languages: Multiple target languages (comma or space separated)
            output_base_dir: Base output directory
        """
        # Determine which languages to process
        languages_to_process = []
        
        if target_languages is not None:
            # Parse multiple languages (support comma or space separated)
            lang_list = re.split(r'[,\s]+', target_languages.strip())
            languages_to_process = [lang.strip() for lang in lang_list if lang.strip()]
        elif target_language is not None:
            languages_to_process = [target_language]
        else:
            # Default to English if not specified
            languages_to_process = ["English"]
            logger.info("No target language specified, defaulting to: English")

        # Validate all languages
        for lang in languages_to_process:
            validate_language(lang)

        logger.info(f"Processing languages: {', '.join(languages_to_process)}")

        # Get all patient IDs (shared across all languages)
        all_patient_ids = self._get_all_patient_ids()
        
        if not all_patient_ids:
            raise ValueError("No patient IDs found. Please check case directory.")
        
        logger.info(f"Found {len(all_patient_ids)} patient IDs")

        if limit is not None and limit > 0:
            all_patient_ids = all_patient_ids[:limit]
            logger.info(f"Limited to {len(all_patient_ids)} patients (limit={limit})")

        # Create output directory structure
        base_output_dir = Path(output_base_dir)
        sanitized_model = sanitize_filename(self.model.replace('/', '_'))

        # Process each language
        for language in languages_to_process:
            logger.info(f"{'=' * 60}")
            logger.info(f"Processing language: {language}")
            logger.info(f"{'=' * 60}")
            
            for round_num in range(1, rounds + 1):
                logger.info(f"Starting round {round_num}/{rounds} for language: {language}")

                output_dir = base_output_dir / sanitized_model / language
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Output directory: {output_dir}")

                output_file = output_dir / f"round{round_num}.json"

                # Load existing results
                existing_results = []
                if output_file.exists():
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            loaded_data = json.load(f)
                        
                        # Handle both old format (list) and new format (dict with metadata)
                        if isinstance(loaded_data, dict) and "results" in loaded_data:
                            existing_results = loaded_data["results"]
                        elif isinstance(loaded_data, dict):
                            existing_results = list(loaded_data.values())
                        elif isinstance(loaded_data, list):
                            existing_results = loaded_data
                        
                        logger.info(
                            f"Round {round_num}, {language}: "
                            f"Found {len(existing_results)} existing results"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not read existing results file: {e}. Starting fresh."
                        )
                        existing_results = []

                # Determine which samples still need processing
                processed_ids = {
                    r.get("patient_id") 
                    for r in existing_results 
                    if isinstance(r, dict) and "patient_id" in r
                }

                tasks = [
                    (pid, language) 
                    for pid in all_patient_ids 
                    if pid not in processed_ids
                ]

                if len(tasks) == 0:
                    logger.info(f"Round {round_num}, {language}: No new samples to process")
                    continue

                logger.info(f"Round {round_num}, {language}: Prepared {len(tasks)} tasks")

                # Initialize counters
                results = existing_results.copy()
                processed = 0
                skipped = 0

                def write_result_threadsafe(result: Optional[Dict[str, Any]]) -> None:
                    """Thread-safe result writer"""
                    nonlocal processed, skipped
                    
                    if result is None:
                        with self._lock:
                            skipped += 1
                        return

                    with self._lock:
                        results.append(result)
                        processed += 1
                        
                        # Checkpoint at intervals
                        if len(results) % checkpoint_interval == 0:
                            self._create_checkpoint(output_file, results, round_num, language)

                try:
                    # Process with thread pool
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_task = {
                            executor.submit(
                                self._process_single_sample,
                                patient_id, language,
                                temperature, max_tokens
                            ): patient_id
                            for patient_id, _ in tasks
                        }

                        # Process as completed with progress bar
                        for future in tqdm(
                            as_completed(future_to_task.keys()),
                            total=len(tasks),
                            desc=f"{language} Round {round_num}",
                            unit="task"
                        ):
                            try:
                                result = future.result()
                                write_result_threadsafe(result)
                            except Exception as e:
                                logger.error(f"Error processing result: {e}")

                except KeyboardInterrupt:
                    logger.info(
                        f"Round {round_num}, {language}: "
                        f"Interrupted by user - saving current progress..."
                    )
                    self._create_checkpoint(output_file, results, round_num, language)
                    logger.info(
                        f"Round {round_num}, {language}: Saved {len(results)} results "
                        f"(processed {processed} new, skipped {skipped})"
                    )
                    raise

                # Final save
                self._create_checkpoint(output_file, results, round_num, language)
                logger.info(
                    f"Round {round_num}, {language} completed. "
                    f"Processed {processed} new samples, skipped {skipped}. "
                    f"Total: {len(results)} results"
                )

            logger.info(f"All rounds completed for language: {language}")

        logger.info(f"{'=' * 60}")
        logger.info(f"All languages completed: {', '.join(languages_to_process)}")
        logger.info(f"{'=' * 60}")

    def _save_results(self, output_file: Path, data: Any) -> None:
        """
        Save results to JSON file (thread-safe).
        
        Args:
            output_file: Output file path
            data: Data to save (can be list or dict)
        """
        def _json_sanitize(obj: Any) -> Any:
            """Recursively replace NaN/inf with None so JSON is valid (RFC 8259)."""
            if isinstance(obj, dict):
                return {k: _json_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_sanitize(v) for v in obj]
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            try:
                if pd.isna(obj):
                    return None
            except (TypeError, ValueError):
                pass
            return obj

        with self._lock:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            data_clean = _json_sanitize(data)

            # Create temp file for atomic write
            temp_file = output_file.with_suffix('.tmp')

            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data_clean, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                temp_file.replace(output_file)
                logger.debug(f"Saved results to {output_file}")
                
            except Exception as e:
                logger.error(f"Error saving results to {output_file}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'client'):
                # OpenAI client doesn't need explicit close in current version
                # but we keep this for future compatibility
                pass
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Language Glaucoma Diagnosis Runner (2 Modalities, No Images)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process English only (default)
  python %(prog)s --model google/gemini-3-pro-preview
  
  # Process multiple languages
  python %(prog)s --target-languages "English,Chinese"
  
  # Process with custom settings
  python %(prog)s --model google/gemini-3-pro-preview --rounds 5 --limit 100
  
  # Test run with single language and limited samples
  python %(prog)s --target-language Chinese --limit 10 --rounds 1
        """
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        default="google/gemini-3-pro-preview",
        help="Model name to use (default: %(default)s)"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key (will use OPENAI_API_KEY env var if not specified)"
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for API (default: %(default)s)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: %(default)s)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens in response (default: None)"
    )
    
    # API settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="API timeout in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of API retries (default: %(default)s)"
    )
    parser.add_argument(
        "--retry-backoff-base",
        type=float,
        default=1.5,
        help="Exponential backoff base for retries (default: %(default)s)"
    )
    
    # Processing settings
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=GlaucomaDiagnosisRunner.DEFAULT_CHECKPOINT_INTERVAL,
        help="Save checkpoint every N results (default: %(default)s)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers (default: %(default)s)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds to process (default: %(default)s)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients to process for testing (default: None = all)"
    )
    
    # Language settings
    parser.add_argument(
        "--target-language",
        type=str,
        default=None,
        choices=LanguageConfig.SUPPORTED_LANGUAGES,
        help="Process specified language (default: English if neither option set)"
    )
    parser.add_argument(
        "--target-languages",
        type=str,
        default=None,
        help=(
            "Process multiple languages, comma or space separated "
            "(e.g., 'English,Chinese' or 'English Chinese'). "
            "Overrides --target-language if set."
        )
    )
    
    # Directory settings
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files (default: %(default)s)"
    )
    parser.add_argument(
        "--case-dir",
        default="OphthalmologyEHRglaucoma/dataset/case",
        help="Directory containing English case JSON files (default: %(default)s)"
    )
    parser.add_argument(
        "--translate-dir",
        default="OphthalmologyEHRglaucoma/result/translate/google_gemini-3-pro-preview",
        help="Directory containing translated case files (default: %(default)s)"
    )
    parser.add_argument(
        "--csv-path",
        default="OphthalmologyEHRglaucoma/dataset/patient-state-diagnosis.csv",
        help="Path to CSV file with structured measurements (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default="OphthalmologyEHRglaucoma/result/response_2type",
        help="Base output directory (will create model_name/language/roundX.json) (default: %(default)s)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point"""
    args = parse_args()

    # Validate API key
    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY")
        if args.api_key is None:
            raise ValueError(
                "API key must be provided via --api_key argument or "
                "OPENAI_API_KEY environment variable"
            )

    # Create runner
    try:
        runner = GlaucomaDiagnosisRunner(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_base=args.retry_backoff_base,
            log_dir=args.log_dir,
            max_workers=args.max_workers,
            case_dir=args.case_dir,
            translate_dir=args.translate_dir,
            csv_path=args.csv_path,
        )
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        raise SystemExit(1)

    # Process
    try:
        runner.process(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            checkpoint_interval=args.checkpoint_interval,
            rounds=args.rounds,
            limit=args.limit,
            target_language=args.target_language,
            target_languages=args.target_languages,
            output_base_dir=args.output_dir
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()