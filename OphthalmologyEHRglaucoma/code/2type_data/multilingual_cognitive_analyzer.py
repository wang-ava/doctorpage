#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Multilingual Cognitive Pattern Analyzer

Core capabilities:
1. Dynamic medical keyword extraction from ground truth and model responses.
2. Automatic confusion pair detection from `top_logprobs`.
3. Diagnosis-stratified performance analysis.
4. Cross-lingual semantic alignment checks.
5. Confidence analysis for correct vs incorrect predictions.
6. Error attribution (knowledge gap / reasoning failure / language barrier).
7. Token-level analysis of medical-term activation and connector usage.
8. Cross-lingual code-switching and concept consistency analysis.
9. Cognitive stability analysis via mutation detection on token logprob traces.
10. Visualization of trajectories, distributions, and heatmaps.
"""

import json
import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Agg')
# Use DejaVu first (full Latin/numbers); CJK and Thai as fallback for multilingual labels
# Script-only fonts (Noto Sans Thai, etc.) lack Latin and must not be primary
_cjk_candidates = ['Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
_thai_candidates = ['Noto Sans Thai', 'Noto Sans Thai Looped']
_available = {f.name for f in fm.fontManager.ttflist}
_font_list = ['DejaVu Sans']
_font_list += [f for f in _cjk_candidates if f in _available]
_font_list += [f for f in _thai_candidates if f in _available and f not in _font_list]
plt.rcParams['font.sans-serif'] = _font_list
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.antialiased'] = True

# Plot text style
_PLOT_FONTSIZE_LABEL = 10
_PLOT_FONTSIZE_TITLE = 11
_PLOT_FONTSIZE_SUPTITLE = 12
_BAR_LABEL_GAP_FRAC = 0.02
_BAR_LABEL_GAP_SMALL = 0.02

# ==========================================
# 1. Dynamic keyword extractor
# ==========================================

class DynamicKeywordExtractor:
    """Extract diagnosis-specific medical keywords from data."""
    
    def __init__(self):
        self.diagnosis_keywords = defaultdict(set)  # {diagnosis_code: {keywords}}
        self.language_medical_terms = defaultdict(Counter)  # {lang: Counter(terms)}
        
    def extract_from_ground_truth(self, ground_truth: Dict, lang_code: str):
        """Extract key medical terms from ground truth with English fallback."""
        keywords = set()
        
        for eye in ['OD', 'OS']:
            if eye in ground_truth:
                diag_code = ground_truth[eye].get('diagnosis_code', '')
                diag_desc = ground_truth[eye].get('diagnosis_description') or ground_truth[eye].get('diagnosis_label', '')
                diag_desc = diag_desc or ''
                
                if diag_code:
                    words = self._tokenize(diag_desc, lang_code)
                    keywords.update(words)
                    self.diagnosis_keywords[diag_code].update(words)
                    # Also tokenize as English for multilingual robustness.
                    if lang_code != 'en':
                        words_en = self._tokenize(diag_desc, 'en')
                        keywords.update(words_en)
                        self.diagnosis_keywords[diag_code].update(words_en)
                    
        return keywords
    
    def extract_from_response(self, response_text: str, lang_code: str) -> Set[str]:
        """Extract medical terms from model output with English fallback."""
        terms = set(self._tokenize(response_text, lang_code))
        if lang_code != 'en':
            terms.update(self._tokenize(response_text, 'en'))
        return terms
    
    def _tokenize(self, text: str, lang_code: str) -> List[str]:
        """Simple multilingual tokenization for Chinese/English/Malay/Thai."""
        if not text:
            return []
        
        text = text.lower()
        
        if lang_code == 'zh':
            # Chinese: extract 2-4 character medical n-grams.
            words = []
            for length in [4, 3, 2]:
                for i in range(len(text) - length + 1):
                    word = text[i:i+length]
                    if self._is_medical_term(word, 'zh'):
                        words.append(word)
            return words
        else:
            # Other languages: split by word boundaries.
            words = re.findall(r'\b\w+\b', text)
            return [w for w in words if len(w) > 2 and self._is_medical_term(w, lang_code)]
    
    def _is_medical_term(self, word: str, lang_code: str) -> bool:
        """Heuristic medical term filter."""
        medical_roots = {
            'zh': ['\u773c', '\u672f', '\u5149', '\u538b', '\u89d2', '\u819c', '\u795e\u7ecf', '\u89c6'],
            'en': ['ophthalm', 'glaucoma', 'surgery', 'pressure', 'cornea', 'retina',
                   'trabec', 'cataract', 'pacg', 'pacs', 'apac', 'lens', 'subluxation',
                   'trabeculectomy', 'remission', 'secondary'],
            'ms': ['mata', 'pembedahan', 'tekanan', 'glaukoma', 'katarak', 'lensa'],
            'th': ['\u0e15\u0e32', '\u0e1c\u0e48\u0e32\u0e15\u0e31\u0e14', '\u0e04\u0e27\u0e32\u0e21\u0e14\u0e31\u0e19', '\u0e15\u0e49\u0e2d', '\u0e41\u0e01\u0e49\u0e27\u0e15\u0e32']
        }
        roots = medical_roots.get(lang_code, [])
        return any(root in word for root in roots)

# ==========================================
# 2. Competitive confusion detector
# ==========================================

class CompetitiveConfusionDetector:
    """Detect near-tie candidate token pairs from top logprob candidates."""
    
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.confusion_pairs = Counter()  # {(token1, token2): count}
        self.semantic_clusters = defaultdict(list)  # {main_token: [competitor_tokens]}
        
    def detect_from_logprobs(self, top_logprobs: List[Dict], selected_token: str):
        """Inspect one token's top candidates and record strong competitions."""
        if len(top_logprobs) < 2:
            return
        
        sorted_tops = sorted(top_logprobs, key=_get_probability, reverse=True)
        top1 = sorted_tops[0]
        top2 = sorted_tops[1]
        
        p1 = _get_probability(top1)
        p2 = _get_probability(top2)
        
        # Top-2 are very close => strong competition.
        if p1 - p2 < self.threshold:
            t1 = top1.get('token', '').strip()
            t2 = top2.get('token', '').strip()
            
            if t1 and t2:
                pair = tuple(sorted([t1, t2]))
                self.confusion_pairs[pair] += 1
                
                # Sampling selected top-2 instead of top-1.
                if selected_token.strip() == t2:
                    self.semantic_clusters[t1].append(t2)
    
    def get_top_confusions(self, n: int = 20) -> List[Tuple[str, str, int]]:
        """Return the most frequent confusion pairs."""
        return [(t1, t2, cnt) for (t1, t2), cnt in self.confusion_pairs.most_common(n)]

# ==========================================
# 3. Diagnosis stratifier
# ==========================================

# Dataset numeric diagnosis schema used for disease-stratified analysis/plots.
DIAGNOSIS_CODE_TO_CATEGORY = {
    0: 'diag_0_not_glaucoma',
    1: 'diag_1_pacs',
    2: 'diag_2_pac',
    3: 'diag_3_pacg',
    4: 'diag_4_apac',
    5: 'diag_5_poag',
    6: 'diag_6_secondary_glaucoma',
}
DIAGNOSIS_CATEGORY_ORDER = [DIAGNOSIS_CODE_TO_CATEGORY[i] for i in [0, 1, 2, 3, 4, 5, 6]]
DIAGNOSIS_CATEGORY_DISPLAY = {
    'diag_0_not_glaucoma': '0=Not glaucoma',
    'diag_1_pacs': '1=PACS',
    'diag_2_pac': '2=PAC',
    'diag_3_pacg': '3=PACG',
    'diag_4_apac': '4=APAC',
    'diag_5_poag': '5=POAG',
    'diag_6_secondary_glaucoma': '6=Secondary glaucoma',
}


class DiagnosisStratifier:
    """Stratify results by diagnosis type and compute per-group metrics."""
    
    def __init__(self):
        # ICD-like patterns (legacy schema).
        self.categories = {
            'primary_open': r'H40\.1[0-9]',
            'primary_closed': r'H40\.2[0-9]',
            'secondary': r'H40\.[3-6][0-9]',
            'postoperative': r'H40\.7[0-9]',
        }
        # Numeric code schema used by this dataset.
        self.numeric_categories = dict(DIAGNOSIS_CODE_TO_CATEGORY)
        self.stats_by_category = defaultdict(lambda: {
            'total': 0, 'correct': 0, 
            'avg_prob': [], 'avg_entropy': [],
            'ppl_scores': []
        })
    
    def categorize(self, diagnosis_code: str) -> str:
        """Map diagnosis code to a category for stratified analysis."""
        if diagnosis_code is None:
            return 'unknown'

        if isinstance(diagnosis_code, (int, np.integer)):
            return self.numeric_categories.get(int(diagnosis_code), 'numeric_other')

        diagnosis_code_str = str(diagnosis_code).strip()
        if diagnosis_code_str == '':
            return 'unknown'

        # Numeric string support.
        if re.fullmatch(r'[-+]?\d+(\.0+)?', diagnosis_code_str):
            return self.numeric_categories.get(int(float(diagnosis_code_str)), 'numeric_other')
        
        for cat, pattern in self.categories.items():
            if re.match(pattern, diagnosis_code_str):
                return cat
        return 'other'
    
    def update(self, category: str, is_correct: bool, prob: float, entropy: float, ppl: float):
        """Update accumulator for one sample."""
        s = self.stats_by_category[category]
        s['total'] += 1
        if is_correct:
            s['correct'] += 1
        s['avg_prob'].append(prob)
        s['avg_entropy'].append(entropy)
        s['ppl_scores'].append(ppl)
    
    def get_summary(self) -> Dict:
        """Compute summary metrics and uncertainty estimates per category."""
        summary = {}
        for cat, s in self.stats_by_category.items():
            if s['total'] > 0:
                n = s['total']
                p = s['correct'] / n
                # Standard error for proportion (binomial)
                acc_se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
                summary[cat] = {
                    'accuracy': p,
                    'accuracy_std': acc_se,
                    'avg_prob': np.mean(s['avg_prob']),
                    'avg_entropy': np.mean(s['avg_entropy']),
                    'avg_entropy_std': np.std(s['avg_entropy']) if len(s['avg_entropy']) > 1 else 0,
                    'avg_ppl': np.mean(s['ppl_scores']),
                    'avg_ppl_std': np.std(s['ppl_scores']) if len(s['ppl_scores']) > 1 else 0,
                    'sample_count': s['total']
                }
        return summary

# ==========================================
# 4. Constants: medical terms and logical connectors
# ==========================================

# Cross-lingual synonym map per concept.
SEMANTIC_ALIGNMENT_TERMS = {
    'glaucoma': ['glaucoma', 'glaukoma', '\u9752\u5149\u773c', '\u0e15\u0e49\u0e2d\u0e2b\u0e34\u0e19', 'glaukoma'],
    'optic_nerve': ['optic nerve', 'optic nerve', '\u89c6\u795e\u7ecf', '\u0e40\u0e2a\u0e49\u0e19\u0e1b\u0e23\u0e30\u0e2a\u0e32\u0e17\u0e15\u0e32', 'saraf optik'],
    'IOP': ['IOP', 'intraocular pressure', '\u773c\u538b', '\u0e04\u0e27\u0e32\u0e21\u0e14\u0e31\u0e19\u0e15\u0e32', 'tekanan intraokular'],
    'trabeculectomy': ['trabeculectomy', 'trabeculectomy', '\u5c0f\u6881\u5207\u9664\u672f', '\u0e1c\u0e48\u0e32\u0e15\u0e31\u0e14 trabeculectomy', 'trabeculectomy'],
}

# Logical connectors used for reasoning-chain analysis.
LOGICAL_CONNECTORS = {
    'zh': ['\u56e0\u4e3a', '\u6240\u4ee5', '\u56e0\u6b64', '\u7531\u4e8e', '\u5bfc\u81f4', '\u8868\u660e', '\u8bf4\u660e', '\u53ef\u89c1', '\u7efc\u4e0a'],
    'en': ['because', 'therefore', 'thus', 'hence', 'so', 'suggests', 'indicates', 'shows', 'thus'],
    'ms': ['kerana', 'oleh itu', 'jadi', 'menunjukkan', 'menandakan'],
    'th': ['\u0e40\u0e1e\u0e23\u0e32\u0e30', '\u0e14\u0e31\u0e07\u0e19\u0e31\u0e49\u0e19', '\u0e08\u0e36\u0e07', '\u0e41\u0e2a\u0e14\u0e07', '\u0e1a\u0e48\u0e07\u0e0a\u0e35\u0e49'],
}

# Medical terms used for activation and stability attribution.
MEDICAL_TERM_PATTERNS = {
    'zh': ['\u89c6\u795e\u7ecf', '\u773c\u538b', '\u9752\u5149\u773c', '\u5c0f\u6881', '\u623f\u89d2', '\u6676\u72b6\u4f53', '\u767d\u5185\u969c', '\u624b\u672f'],
    'en': ['glaucoma', 'IOP', 'optic nerve', 'trabeculectomy', 'angle', 'lens', 'cataract', 'surgery'],
    'ms': ['glaukoma', 'tekanan', 'saraf', 'pembedahan', 'katarak', 'lensa'],
    'th': ['\u0e15\u0e49\u0e2d\u0e2b\u0e34\u0e19', '\u0e04\u0e27\u0e32\u0e21\u0e14\u0e31\u0e19', '\u0e40\u0e2a\u0e49\u0e19\u0e1b\u0e23\u0e30\u0e2a\u0e32\u0e17', '\u0e1c\u0e48\u0e32\u0e15\u0e31\u0e14', '\u0e15\u0e49\u0e2d\u0e01\u0e23\u0e30\u0e08\u0e01', '\u0e41\u0e01\u0e49\u0e27\u0e15\u0e32'],
}

# Reasoning phases by normalized token position.
STABILITY_PHASE_BOUNDS = [(0.0, 0.2, 'problem'), (0.2, 0.6, 'retrieval'), (0.6, 1.0, 'answer')]
# Answer-option patterns used for key-position mutation statistics.
ANSWER_OPTION_PATTERNS = {
    'zh': ['\u9009\u9879', 'A', 'B', 'C', 'D', '\u4e00', '\u4e8c', '\u4e09', '\u56db'],
    'en': ['option', 'A', 'B', 'C', 'D', 'first', 'second', 'choice'],
    'ms': ['pilihan', 'A', 'B', 'C', 'D', 'pertama', 'kedua'],
    'th': ['\u0e15\u0e31\u0e27\u0e40\u0e25\u0e37\u0e2d\u0e01', 'A', 'B', 'C', 'D'],
}
# Mutation detection thresholds over token logprob deltas.
MUTATION_ABSOLUTE_THRESHOLDS = [2.0, 2.5, 3.0]
MUTATION_RELATIVE_STD_MULTIPLIER = 1.5
STABILITY_ALPHA = 0.1   # stability = 1/(1 + alpha*count + beta*avg_magnitude)
STABILITY_BETA = 0.05

# Pairwise gap-source attribution thresholds (tunable for analysis sensitivity).
GAIN_DIRECT_LOW_PROB = 0.55
GAIN_REASONING_HIGH_PROB = 0.75
GAIN_PROB_DELTA_MIN = 0.08
GAIN_ENTROPY_DELTA_MIN = 0.08
GAIN_MUTATION_DELTA_MIN = 2

LOSS_OVERCONFIDENT_PROB = 0.85
LOSS_PROB_DELTA_MIN = 0.05
LOSS_CONFIDENCE_DROP_MIN = 0.08
LOSS_ENTROPY_INCREASE_MIN = 0.08
LOSS_MUTATION_DELTA_MIN = 2
LOSS_CODE_SWITCH_DELTA_MIN = 0.15

# CoT uncertainty analysis thresholds.
CONCLUSION_LOW_LOGPROB_THRESHOLD = -0.5
COT_PPL_BINS = 20
LOGIC_BREAK_DROP_THRESHOLD = 1.5
LOGIC_BREAK_Z_THRESHOLD = 2.0
LOGIC_BREAK_ABS_LOGPROB_THRESHOLD = -2.5
LOGIC_BREAK_MAX_RECORDS = 30
CONCLUSION_CODE_FIELD_KEYS = ['诊断代码', 'diagnosis_code', 'kod_diagnosis', 'รหัสการวินิจฉัย']

# English is used as reference-only line in figures.
ENGLISH_AS_REFERENCE_LINE = True
DISPLAY_LANGUAGES_EXCLUDE = ['English']
RED_LINE_LABEL = 'reasoning in target language'
LANGUAGE_LABEL_COLORS = {
    'Chinese': '#9B59B6',  # purple
    'Malay': '#3498DB',    # blue
    'Thai': '#27AE60',     # green
}


def _safe_float(value, default: float = 0.0) -> float:
    """Safely cast to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_probability(item: Dict) -> float:
    """Read probability from token item with logprob fallback."""
    if isinstance(item, dict):
        if item.get('probability') is not None:
            return float(np.clip(_safe_float(item.get('probability'), 0.0), 0.0, 1.0))
        if item.get('logprob') is not None:
            return float(np.clip(np.exp(_safe_float(item.get('logprob'), -999.0)), 0.0, 1.0))
    return 0.0


def _get_logprob(item: Dict) -> float:
    """Read logprob from token item with probability fallback."""
    if isinstance(item, dict):
        if item.get('logprob') is not None:
            return _safe_float(item.get('logprob'), -999.0)
        return float(np.log(max(_get_probability(item), 1e-12)))
    return -999.0


def _prepare_token_view(logprobs_data: List[Dict]) -> Tuple[List[str], List[float], str, List[Tuple[int, int]]]:
    """Build token text/probability arrays and char spans in concatenated token text."""
    tokens = [(item.get('token') or '') for item in logprobs_data]
    probs = [_get_probability(item) for item in logprobs_data]
    joined = ''.join(tokens).lower()
    spans: List[Tuple[int, int]] = []
    offset = 0
    for token in tokens:
        start = offset
        offset += len(token)
        spans.append((start, offset))
    return tokens, probs, joined, spans


def _find_pattern_token_indices(pattern: str, joined_text: str, spans: List[Tuple[int, int]]) -> List[int]:
    """Return token indices whose character spans overlap any pattern match."""
    pattern = (pattern or '').strip().lower()
    if not pattern:
        return []
    matches = list(re.finditer(re.escape(pattern), joined_text))
    if not matches:
        return []
    indices: Set[int] = set()
    for match in matches:
        start = match.start()
        end = match.end()
        for idx, (token_start, token_end) in enumerate(spans):
            if token_end <= start or token_start >= end:
                continue
            indices.add(idx)
    return sorted(indices)


def _find_next_nonempty_token_index(start_char: int, spans: List[Tuple[int, int]], tokens: List[str]) -> int:
    """Find first non-empty token whose start char offset is >= start_char."""
    for idx, (token_start, _) in enumerate(spans):
        if token_start < start_char:
            continue
        if (tokens[idx] or '').strip() == '':
            continue
        return idx
    return -1


def _collect_connector_next_token_probabilities(logprobs_data: List[Dict], lang_code: str) -> List[Dict]:
    """
    Collect confidence for the token immediately after each logical connector.
    This operationalizes "key-step confidence" in CoT.
    """
    if not logprobs_data:
        return []
    tokens, probs, joined_text, spans = _prepare_token_view(logprobs_data)
    out: List[Dict] = []
    connector_patterns = LOGICAL_CONNECTORS.get(lang_code, []) + LOGICAL_CONNECTORS.get('en', [])
    for connector in connector_patterns:
        connector_l = (connector or '').strip().lower()
        if not connector_l:
            continue
        for m in re.finditer(re.escape(connector_l), joined_text):
            next_idx = _find_next_nonempty_token_index(m.end(), spans, tokens)
            if next_idx < 0 or next_idx >= len(tokens):
                continue
            out.append({
                'connector': connector,
                'next_token': tokens[next_idx],
                'token_index': next_idx,
                'probability': float(probs[next_idx]),
                'logprob': float(_get_logprob(logprobs_data[next_idx])),
            })
    return out


def _extract_conclusion_code_token_confidence(logprobs_data: List[Dict]) -> List[Dict]:
    """
    Extract confidence for diagnosis-code tokens in final JSON answers.
    Used for low-probability warnings on conclusion numbers.
    """
    if not logprobs_data:
        return []
    tokens, probs, joined_text, spans = _prepare_token_view(logprobs_data)
    code_key_group = '|'.join(re.escape(k.lower()) for k in CONCLUSION_CODE_FIELD_KEYS)
    pattern = re.compile(rf'"(?:{code_key_group})"\s*:\s*(?P<code>[0-6])')
    out: List[Dict] = []
    for m in pattern.finditer(joined_text):
        c_start, c_end = m.span('code')
        token_idx = -1
        for idx, (token_start, token_end) in enumerate(spans):
            if token_end <= c_start or token_start >= c_end:
                continue
            token_idx = idx
            break
        if token_idx < 0 or token_idx >= len(tokens):
            continue
        lp = float(_get_logprob(logprobs_data[token_idx]))
        out.append({
            'code': int(m.group('code')),
            'token': tokens[token_idx],
            'token_index': token_idx,
            'probability': float(probs[token_idx]),
            'logprob': lp,
            'low_probability_warning': bool(lp <= CONCLUSION_LOW_LOGPROB_THRESHOLD),
        })
    return out


def _detect_logic_breaks(logprobs_data: List[Dict]) -> List[Dict]:
    """
    Detect abrupt low-confidence valleys in CoT token trace.
    A logic-break candidate requires a sharp drop plus unusually low logprob.
    """
    if not logprobs_data or len(logprobs_data) < 3:
        return []
    logprobs = np.array([_get_logprob(item) for item in logprobs_data], dtype=float)
    tokens = [(item.get('token') or '') for item in logprobs_data]
    valid = logprobs[np.isfinite(logprobs)]
    if valid.size == 0:
        return []
    mu = float(np.mean(valid))
    sigma = float(np.std(valid))
    sigma = sigma if sigma > 1e-9 else 1.0
    events: List[Dict] = []
    for i in range(1, len(logprobs)):
        tok = (tokens[i] or '').strip()
        if tok == '':
            continue
        drop = float(logprobs[i] - logprobs[i - 1])
        if drop > -LOGIC_BREAK_DROP_THRESHOLD:
            continue
        z_low = float((mu - logprobs[i]) / sigma)
        if z_low >= LOGIC_BREAK_Z_THRESHOLD or logprobs[i] <= LOGIC_BREAK_ABS_LOGPROB_THRESHOLD:
            events.append({
                'position': i,
                'token': tokens[i],
                'logprob': float(logprobs[i]),
                'drop_from_prev': drop,
                'z_low': z_low,
            })
    return events

# ==========================================
# 5. Confidence analyzer
# ==========================================

class ConfidenceAnalyzer:
    """Confidence summary for correct/incorrect and uncertainty."""
    
    def get_summary(self, stats: Dict) -> Dict:
        """Build confidence summary from aggregated stats."""
        correct_probs = stats.get('correct_sample_probs', [])
        incorrect_probs = stats.get('incorrect_sample_probs', [])
        entropy_all = stats.get('entropy_all', [])
        ppl_by_correct = stats.get('ppl_by_correct', [])
        ppl_correct = [p for ok, p in ppl_by_correct if ok]
        ppl_incorrect = [p for ok, p in ppl_by_correct if not ok]
        return {
            'avg_logprob_correct': np.mean([np.log(p+1e-12) for p in correct_probs]) if correct_probs else 0,
            'avg_logprob_incorrect': np.mean([np.log(p+1e-12) for p in incorrect_probs]) if incorrect_probs else 0,
            'avg_prob_correct': np.mean(correct_probs) if correct_probs else 0,
            'avg_prob_incorrect': np.mean(incorrect_probs) if incorrect_probs else 0,
            'knows_uncertainty': np.mean(incorrect_probs) < np.mean(correct_probs) if (correct_probs and incorrect_probs) else False,
            'avg_entropy': np.mean(entropy_all) if entropy_all else 0,
            'entropy_std': np.std(entropy_all) if len(entropy_all) > 1 else 0,
            'avg_ppl_correct': np.mean(ppl_correct) if ppl_correct else 0,
            'avg_ppl_incorrect': np.mean(ppl_incorrect) if ppl_incorrect else 0,
        }

# ==========================================
# 6. Error attribution analyzer
# ==========================================

class ErrorAttributionAnalyzer:
    """Classify errors as knowledge gap, reasoning failure, or language barrier."""
    
    def __init__(self, low_prob_threshold: float = 0.5, high_prob_threshold: float = 0.85):
        self.low_threshold = low_prob_threshold
        self.high_threshold = high_prob_threshold
    
    def classify_error(self, avg_prob: float, is_direct: bool, direct_prob: float = None, 
                       reasoning_prob: float = None) -> str:
        """Classify one error sample."""
        if avg_prob < self.low_threshold:
            return 'knowledge_gap'
        if avg_prob >= self.high_threshold:
            return 'reasoning_failure'
        if is_direct and direct_prob is not None and reasoning_prob is not None:
            if direct_prob < self.low_threshold and reasoning_prob >= self.high_threshold:
                return 'language_barrier'
        return 'mixed'
    
    def analyze_errors(self, results: List[Dict], language: str, 
                       direct_results_by_pid: Dict = None, reasoning_results_by_pid: Dict = None) -> Dict:
        """Compute error-type distribution over incorrect samples."""
        counts = defaultdict(int)
        for r in results:
            pred, gt = extract_diagnosis_codes(r)
            is_correct = (pred.get('OD') == gt.get('OD') and pred.get('OS') == gt.get('OS') and pred.get('OD') is not None)
            if is_correct:
                continue
            logprobs_data = r.get('output', {}).get('logprobs', [])
            if not logprobs_data:
                continue
            probs = [_get_probability(item) for item in logprobs_data]
            avg_prob = np.mean(probs) if probs else 0
            direct_prob = reasoning_prob = None
            pid = r.get('patient_id')
            if direct_results_by_pid and pid in direct_results_by_pid:
                lp = direct_results_by_pid[pid].get('output', {}).get('logprobs', [])
                direct_prob = np.mean([_get_probability(x) for x in lp]) if lp else None
            if reasoning_results_by_pid and pid in reasoning_results_by_pid:
                lp = reasoning_results_by_pid[pid].get('output', {}).get('logprobs', [])
                reasoning_prob = np.mean([_get_probability(x) for x in lp]) if lp else None
            err_type = self.classify_error(avg_prob, True, direct_prob, reasoning_prob)
            counts[err_type] += 1
        total = sum(counts.values())
        dist = {k: (v / total if total else 0) for k, v in counts.items()}
        return {'distribution': dist, 'counts': dict(counts), 'total_errors': total}

# ==========================================
# 7. Token-level analyzer
# ==========================================

class TokenLevelAnalyzer:
    """Track medical term activation, connectors, and confidence decay."""
    
    def __init__(self):
        self.medical_term_logprobs = defaultdict(list)  # {term_pattern: [prob]}
        self.connector_logprobs = defaultdict(list)     # {connector: [prob]}
        self.connector_next_token_logprobs = defaultdict(list)  # {connector: [next_token_prob]}
        self.position_vs_prob = []  # [(position_frac, prob)]
    
    def analyze_sample(self, logprobs_data: List[Dict], lang_code: str):
        if not logprobs_data:
            return
        tokens, probs, joined_text, spans = _prepare_token_view(logprobs_data)
        n = len(tokens)
        for pos, prob in enumerate(probs):
            pos_frac = pos / n if n > 0 else 0
            self.position_vs_prob.append((pos_frac, prob))

        # Match patterns on joined token stream to avoid false zeros with subword tokenization.
        medical_patterns = MEDICAL_TERM_PATTERNS.get(lang_code, []) + MEDICAL_TERM_PATTERNS.get('en', [])
        for pattern in medical_patterns:
            idxs = _find_pattern_token_indices(pattern, joined_text, spans)
            if idxs:
                self.medical_term_logprobs[pattern].append(float(np.mean([probs[i] for i in idxs])))

        connector_patterns = LOGICAL_CONNECTORS.get(lang_code, []) + LOGICAL_CONNECTORS.get('en', [])
        for connector in connector_patterns:
            idxs = _find_pattern_token_indices(connector, joined_text, spans)
            if idxs:
                self.connector_logprobs[connector].append(float(np.mean([probs[i] for i in idxs])))

        # Key-step confidence: token right after connector.
        next_token_entries = _collect_connector_next_token_probabilities(logprobs_data, lang_code)
        for entry in next_token_entries:
            self.connector_next_token_logprobs[entry['connector']].append(entry['probability'])
    
    def get_summary(self) -> Dict:
        decay_bins = defaultdict(list)  # bin -> [prob]
        for pos_frac, prob in self.position_vs_prob:
            bin_idx = min(int(pos_frac * 10), 9)
            decay_bins[bin_idx].append(prob)
        decay_curve = [np.mean(decay_bins[i]) if decay_bins[i] else 0 for i in range(10)]
        return {
            'medical_term_avg_prob': {k: np.mean(v) if v else 0 for k, v in self.medical_term_logprobs.items()},
            'connector_avg_prob': {k: np.mean(v) if v else 0 for k, v in self.connector_logprobs.items()},
            'connector_next_token_avg_prob': {k: np.mean(v) if v else 0 for k, v in self.connector_next_token_logprobs.items()},
            'confidence_decay_curve': decay_curve,
            'decay_slope': (decay_curve[-1] - decay_curve[0]) if len(decay_curve) >= 2 else 0,
        }

# ==========================================
# 8. Cross-lingual analyzer
# ==========================================

class CrossLingualAnalyzer:
    """Analyze semantic alignment, language bias, and code-switching."""
    
    def __init__(self):
        self.concept_logprobs = defaultdict(lambda: defaultdict(list))  # {concept: {lang: [prob]}}
        self.code_switch_ratio = defaultdict(list)  # {lang: [ratio_per_sample]}
    
    def analyze_sample(self, logprobs_data: List[Dict], response_text: str, lang_code: str, lang_name: str):
        if not logprobs_data:
            return
        tokens, probs, joined_text, spans = _prepare_token_view(logprobs_data)
        full_text = (response_text or '').lower()
        # Semantic alignment by concept terms.
        for concept, terms in SEMANTIC_ALIGNMENT_TERMS.items():
            for term in terms:
                term_l = term.lower()
                if term_l not in full_text and term_l not in joined_text:
                    continue
                idxs = _find_pattern_token_indices(term_l, joined_text, spans)
                if idxs:
                    self.concept_logprobs[concept][lang_name].append(float(np.mean([probs[i] for i in idxs])))

        # Code-switching ratio in non-English target languages.
        if lang_code != 'en':
            en_pattern = re.compile(r'[a-zA-Z]{3,}')
            total = len([t for t in tokens if len(t.strip()) > 0])
            en_tokens = len([t for t in tokens if en_pattern.search(t)])
            ratio = en_tokens / total if total > 0 else 0
            self.code_switch_ratio[lang_name].append(ratio)
    
    def get_summary(self) -> Dict:
        alignment = {}
        for concept, lang_probs in self.concept_logprobs.items():
            alignment[concept] = {lang: np.mean(probs) if probs else 0 
                                 for lang, probs in lang_probs.items()}
        code_switch = {lang: np.mean(ratios) if ratios else 0 
                      for lang, ratios in self.code_switch_ratio.items()}
        return {'semantic_alignment': alignment, 'code_switch_ratio': code_switch}


# ==========================================
# 9. Cognitive stability analyzer
# ==========================================

def _token_matches_any_pattern(token_idx: int, patterns: List[str], joined_text: str, spans: List[Tuple[int, int]]) -> bool:
    """Check whether token at token_idx overlaps any pattern match."""
    if token_idx < 0 or token_idx >= len(spans):
        return False
    token_start, token_end = spans[token_idx]
    if token_start == token_end:
        return False
    for pattern in patterns:
        for match in re.finditer(re.escape(pattern.lower()), joined_text):
            if match.end() <= token_start or match.start() >= token_end:
                continue
            return True
    return False


def _get_token_content_type(token_idx: int, token_text: str, lang_code: str, joined_text: str, spans: List[Tuple[int, int]]) -> str:
    """Label token as medical/logical/answer/other."""
    t = (token_text or '').strip().lower()
    if t == '':
        return 'other'
    if _token_matches_any_pattern(
        token_idx,
        MEDICAL_TERM_PATTERNS.get(lang_code, []) + MEDICAL_TERM_PATTERNS.get('en', []),
        joined_text,
        spans
    ):
        return 'medical'
    if _token_matches_any_pattern(
        token_idx,
        LOGICAL_CONNECTORS.get(lang_code, []) + LOGICAL_CONNECTORS.get('en', []),
        joined_text,
        spans
    ):
        return 'logical'
    if _token_matches_any_pattern(
        token_idx,
        ANSWER_OPTION_PATTERNS.get(lang_code, []) + ANSWER_OPTION_PATTERNS.get('en', []),
        joined_text,
        spans
    ):
        return 'answer'
    return 'other'


def _get_phase_at_position(pos_frac: float) -> str:
    """Map normalized token position to problem/retrieval/answer phase."""
    for lo, hi, name in STABILITY_PHASE_BOUNDS:
        if lo <= pos_frac < hi:
            return name
    return 'answer'


class CognitiveStabilityAnalyzer:
    """
    Detect reasoning-path mutations from token-level logprob traces.
    Low mutation count usually indicates smoother reasoning dynamics.
    """
    
    def __init__(self, absolute_thresholds: List[float] = None, relative_std_mult: float = MUTATION_RELATIVE_STD_MULTIPLIER):
        self.absolute_thresholds = absolute_thresholds or MUTATION_ABSOLUTE_THRESHOLDS
        self.relative_std_mult = relative_std_mult
        self.per_sample_records = []
        self.mutation_counts_by_threshold = defaultdict(list)  # threshold -> [count per sample]
        self.by_correct = defaultdict(list)  # correct: [mutation_count], incorrect: [...]
        self.mutation_at_medical = []
        self.mutation_at_answer_phase = []
        self.all_magnitudes = []
        self.all_intervals = []  # interval between adjacent mutations
        self.binned_logprobs = [[] for _ in range(10)]  # used by stability heatmap
    
    def analyze_sample(self, logprobs_data: List[Dict], lang_code: str, is_correct: bool, sample_idx: int = 0) -> Dict:
        """
        Analyze one sample and return mutation summary plus trace details.
        """
        if not logprobs_data or len(logprobs_data) < 2:
            return {}
        n = len(logprobs_data)
        tokens_raw, _, joined_text, spans = _prepare_token_view(logprobs_data)
        # Use native token logprob when available. This preserves granularity that can be
        # lost in rounded probability fields (many values become exactly 1.0 -> log(1)=0).
        raw_logprobs = [_get_logprob(item) for item in logprobs_data]
        logprobs = np.array([(lp if np.isfinite(lp) else -999.0) for lp in raw_logprobs], dtype=float)
        tokens = [t.strip() for t in tokens_raw]
        for pos, lp in enumerate(logprobs):
            bin_idx = min(int((pos / n) * 10), 9)
            self.binned_logprobs[bin_idx].append(float(lp))
        deltas = np.diff(logprobs)  # negative: drop, positive: jump
        std_lp = np.std(logprobs)
        relative_threshold = (std_lp * self.relative_std_mult) if std_lp > 1e-9 else 2.0
        
        # Primary threshold is the first absolute threshold.
        primary_threshold = self.absolute_thresholds[0]
        mutations_primary = []  # [(pos, delta, direction, phase, content_type), ...]
        
        for i in range(len(deltas)):
            d = float(deltas[i])
            abs_d = abs(d)
            pos_frac = (i + 0.5) / n
            phase = _get_phase_at_position(pos_frac)
            content = _get_token_content_type(i + 1, tokens[i + 1], lang_code, joined_text, spans) if i + 1 < len(tokens) else 'other'
            
            is_mutation_primary = abs_d >= primary_threshold
            if is_mutation_primary:
                direction = 'down' if d < 0 else 'up'
                mutations_primary.append((i + 1, d, direction, phase, content))
        
        # Multi-threshold mutation counts.
        for th in self.absolute_thresholds:
            cnt = sum(1 for i in range(len(deltas)) if abs(float(deltas[i])) >= th)
            self.mutation_counts_by_threshold[th].append(cnt)
        rel_cnt = sum(1 for i in range(len(deltas)) if abs(float(deltas[i])) >= relative_threshold)
        self.mutation_counts_by_threshold['relative'].append(rel_cnt)
        
        # Oscillation: consecutive mutations with opposite directions.
        oscillation_count = 0
        for j in range(1, len(mutations_primary)):
            if (mutations_primary[j - 1][1] * mutations_primary[j][1]) < 0:
                oscillation_count += 1
        
        mutation_count_primary = len(mutations_primary)
        self.by_correct[is_correct].append(mutation_count_primary)
        self.mutation_at_medical.append(
            sum(1 for m in mutations_primary if m[4] == 'medical') / max(mutation_count_primary, 1)
        )
        self.mutation_at_answer_phase.append(
            sum(1 for m in mutations_primary if m[3] == 'answer') / max(mutation_count_primary, 1)
        )
        intervals = []
        if mutations_primary:
            self.all_magnitudes.extend([abs(m[1]) for m in mutations_primary])
            positions = [m[0] for m in mutations_primary]
            intervals = [positions[k] - positions[k - 1] for k in range(1, len(positions))]
            self.all_intervals.extend(intervals)
        
        # Stability score.
        avg_mag = np.mean([abs(m[1]) for m in mutations_primary]) if mutations_primary else 0
        stability = 1.0 / (1.0 + STABILITY_ALPHA * mutation_count_primary + STABILITY_BETA * avg_mag)
        mean_interval = np.mean(intervals) if intervals else n
        
        per_record = {
            'sample_idx': sample_idx,
            'is_correct': is_correct,
            'mutation_count': mutation_count_primary,
            'mutation_rate': mutation_count_primary / n if n else 0,
            'mean_interval': mean_interval,
            'max_magnitude': max([abs(m[1]) for m in mutations_primary]) if mutations_primary else 0,
            'stability_score': stability,
            'mutation_at_medical_ratio': self.mutation_at_medical[-1] if mutations_primary else 0,
            'mutation_at_answer_ratio': self.mutation_at_answer_phase[-1] if mutations_primary else 0,
            'down_count': sum(1 for m in mutations_primary if m[2] == 'down'),
            'up_count': sum(1 for m in mutations_primary if m[2] == 'up'),
            'oscillation_count': oscillation_count,
        }
        self.per_sample_records.append(per_record)
        
        return {
            'mutation_count': mutation_count_primary,
            'mutation_rate': mutation_count_primary / n if n else 0,
            'mean_interval': mean_interval,
            'max_magnitude': per_record['max_magnitude'],
            'magnitude_std': np.std([abs(m[1]) for m in mutations_primary]) if len(mutations_primary) > 1 else 0,
            'stability_score': stability,
            'mutation_at_medical_ratio': per_record['mutation_at_medical_ratio'],
            'mutation_at_answer_ratio': per_record['mutation_at_answer_ratio'],
            'down_count': per_record['down_count'],
            'up_count': per_record['up_count'],
            'oscillation_count': oscillation_count,
            'mutations_detail': mutations_primary,
            'logprobs': logprobs.tolist(),
            'tokens': tokens,
            'n_tokens': n,
        }
    
    def get_summary(self) -> Dict:
        """Aggregate mutation statistics for plotting and reporting."""
        primary = self.absolute_thresholds[0]
        counts_primary = []
        for r in self.per_sample_records:
            counts_primary.append(r['mutation_count'])
        correct_counts = self.by_correct.get(True, [])
        incorrect_counts = self.by_correct.get(False, [])
        return {
            'mutation_count_mean': np.mean(counts_primary) if counts_primary else 0,
            'mutation_count_std': np.std(counts_primary) if len(counts_primary) > 1 else 0,
            'mutation_rate_mean': np.mean([r['mutation_rate'] for r in self.per_sample_records]) if self.per_sample_records else 0,
            'mean_interval_mean': np.mean(self.all_intervals) if self.all_intervals else 0,
            'max_magnitude_mean': np.mean([r['max_magnitude'] for r in self.per_sample_records]) if self.per_sample_records else 0,
            'magnitude_std_mean': np.std(self.all_magnitudes) if len(self.all_magnitudes) > 1 else 0,
            'stability_score_mean': np.mean([r['stability_score'] for r in self.per_sample_records]) if self.per_sample_records else 0,
            'mutation_at_medical_ratio_mean': np.mean(self.mutation_at_medical) if self.mutation_at_medical else 0,
            'mutation_at_answer_ratio_mean': np.mean(self.mutation_at_answer_phase) if self.mutation_at_answer_phase else 0,
            'by_correct': {
                'correct_mean_count': np.mean(correct_counts) if correct_counts else 0,
                'incorrect_mean_count': np.mean(incorrect_counts) if incorrect_counts else 0,
                'correct_mean_stability': np.mean([r['stability_score'] for r in self.per_sample_records if r['is_correct']]) or 0,
                'incorrect_mean_stability': np.mean([r['stability_score'] for r in self.per_sample_records if not r['is_correct']]) or 0,
            },
            'by_threshold': {str(th): np.mean(self.mutation_counts_by_threshold[th]) if self.mutation_counts_by_threshold[th] else 0 for th in self.absolute_thresholds},
            'relative_threshold_mean': np.mean(self.mutation_counts_by_threshold['relative']) if self.mutation_counts_by_threshold['relative'] else 0,
            'primary_threshold': primary,
            'per_sample_records': self.per_sample_records,
            'binned_mean_logprob': [np.mean(self.binned_logprobs[i]) if self.binned_logprobs[i] else 0 for i in range(10)],
        }


# ==========================================
# 10. Core analysis function
# ==========================================

def analyze_dataset_v2(results: List[Dict], language: str) -> Dict:
    """
    Main analysis pipeline:
    - confusion detection and diagnosis stratification
    - confidence/error/token-level/cross-lingual metrics
    - cognitive stability (mutation detection)
    """
    lang_code = get_language_code(language)
    
    # Initialize analyzers.
    confusion_detector = CompetitiveConfusionDetector(threshold=0.15)
    diagnosis_stratifier = DiagnosisStratifier()
    token_level = TokenLevelAnalyzer()
    cross_lingual = CrossLingualAnalyzer()
    
    stats = {
        'total_samples': len(results),
        'correct_samples': 0,
        'incorrect_samples': 0,
        'correct_samples_right_eye': 0,
        'correct_samples_left_eye': 0,
        'correct_samples_any_eye': 0,
        'evaluated_samples_right_eye': 0,
        'evaluated_samples_left_eye': 0,
        'evaluated_samples_any_eye': 0,
        
        # Base metrics.
        'all_probs': [],
        'correct_sample_probs': [],
        'incorrect_sample_probs': [],
        'ppl_scores': [],
        'ppl_by_correct': [],  # [(is_correct, ppl)]
        'entropy_all': [],
        'top1_dominance': [],
        'token_lengths': [],
        'cot_ppl_bins': defaultdict(list),  # 0..COT_PPL_BINS-1 -> [token_ppl]
        'logic_break_total': 0,
        'logic_break_samples': 0,
        'logic_break_examples': [],
        'connector_next_token_confidences': [],
        'conclusion_code_confidences': [],
        'low_probability_warnings': [],
        
        # Extended metrics.
        'auto_detected_confusions': None,
        'diagnosis_stratified': None,
        'confidence_analysis': None,
        'token_level_analysis': None,
        'cross_lingual_analysis': None,
        'cognitive_stability': None,
        'stability_trajectory_high_mutation': None,  # for single-case visualization
        'stability_trajectory_low_mutation': None,
        
        # Position analysis.
        'token_position_probs': defaultdict(list),
        'sampling_errors_by_position': defaultdict(int),
    }
    
    stability_analyzer = CognitiveStabilityAnalyzer()
    high_mutation_trajectory = None
    low_mutation_trajectory = None
    
    for sample_idx, result in enumerate(results):
        # Diagnosis correctness.
        pred, gt = extract_diagnosis_codes(result)
        pred_od = pred.get('OD')
        pred_os = pred.get('OS')
        gt_od = gt.get('OD')
        gt_os = gt.get('OS')

        od_evaluable = gt_od is not None
        os_evaluable = gt_os is not None
        if od_evaluable:
            stats['evaluated_samples_right_eye'] += 1
        if os_evaluable:
            stats['evaluated_samples_left_eye'] += 1
        if od_evaluable or os_evaluable:
            stats['evaluated_samples_any_eye'] += 1

        od_correct = bool(od_evaluable and pred_od == gt_od and pred_od is not None)
        os_correct = bool(os_evaluable and pred_os == gt_os and pred_os is not None)
        if od_correct:
            stats['correct_samples_right_eye'] += 1
        if os_correct:
            stats['correct_samples_left_eye'] += 1
        if od_correct or os_correct:
            stats['correct_samples_any_eye'] += 1

        is_correct = bool(od_correct and os_correct)
        
        if is_correct: 
            stats['correct_samples'] += 1
        else: 
            stats['incorrect_samples'] += 1
        
        # Token probability analysis.
        logprobs_data = result.get('output', {}).get('logprobs', [])
        if not logprobs_data: 
            continue
        
        seq_len = len(logprobs_data)
        stats['token_lengths'].append(seq_len)
        
        # Sequence perplexity.
        seq_logprobs = [_get_logprob(item) for item in logprobs_data]
        if seq_logprobs:
            ppl = np.exp(-np.mean(seq_logprobs))
            stats['ppl_scores'].append(min(ppl, 100))
            stats['ppl_by_correct'].append((is_correct, min(ppl, 100)))
            token_ppl = np.clip(np.exp(-np.array(seq_logprobs, dtype=float)), 0, 1000)
            denom = max(len(token_ppl) - 1, 1)
            for pos, token_p in enumerate(token_ppl):
                bin_idx = min(int((pos / denom) * COT_PPL_BINS), COT_PPL_BINS - 1)
                stats['cot_ppl_bins'][bin_idx].append(float(token_p))

        # Key-step confidence: token immediately after logical connector.
        next_token_entries = _collect_connector_next_token_probabilities(logprobs_data, lang_code)
        if next_token_entries:
            pid = result.get('patient_id')
            for entry in next_token_entries:
                stats['connector_next_token_confidences'].append({
                    'patient_id': pid,
                    **entry,
                })

        # Conclusion-number confidence with low-probability warning.
        conclusion_entries = _extract_conclusion_code_token_confidence(logprobs_data)
        if conclusion_entries:
            pid = result.get('patient_id')
            for entry in conclusion_entries:
                item = {'patient_id': pid, **entry}
                stats['conclusion_code_confidences'].append(item)
                if item['low_probability_warning']:
                    stats['low_probability_warnings'].append(item)

        # CoT logic-break detection from abrupt low-confidence valleys.
        logic_breaks = _detect_logic_breaks(logprobs_data)
        if logic_breaks:
            stats['logic_break_total'] += len(logic_breaks)
            stats['logic_break_samples'] += 1
            if len(stats['logic_break_examples']) < LOGIC_BREAK_MAX_RECORDS:
                stats['logic_break_examples'].append({
                    'patient_id': result.get('patient_id'),
                    'sample_idx': sample_idx,
                    'break_count': len(logic_breaks),
                    'events': logic_breaks[:6],
                })
        
        # Token-level and cross-lingual analysis.
        output = result.get('output', {})
        response_text = output.get('raw_response') or output.get('model_response_raw', '') or ''
        token_level.analyze_sample(logprobs_data, lang_code)
        cross_lingual.analyze_sample(logprobs_data, response_text, lang_code, language)
        
        # Per-token statistics.
        sample_entropies = []
        sample_avg_prob = []
        
        for pos, token_info in enumerate(logprobs_data):
            token = token_info.get('token', '')
            prob = _get_probability(token_info)
            top_logprobs = token_info.get('top_logprobs', [])
            
            stats['all_probs'].append(prob)
            sample_avg_prob.append(prob)
            stats['token_position_probs'][pos].append(prob)
            
            if is_correct:
                stats['correct_sample_probs'].append(prob)
            else:
                stats['incorrect_sample_probs'].append(prob)
            
            if top_logprobs:
                # Competitive confusion detection.
                confusion_detector.detect_from_logprobs(top_logprobs, token)
                
                # Entropy.
                e = calculate_entropy(top_logprobs)
                sample_entropies.append(e)
                stats['entropy_all'].append(e)
                
                # Top-1 dominance.
                sorted_tops = sorted(top_logprobs, key=_get_probability, reverse=True)
                if len(sorted_tops) >= 2:
                    p1 = _get_probability(sorted_tops[0])
                    p2 = _get_probability(sorted_tops[1])
                    stats['top1_dominance'].append(p1 - p2)
        
        # Diagnosis-stratified update.
        diag_code = gt.get('OD', '')
        category = diagnosis_stratifier.categorize(diag_code)
        avg_sample_prob = np.mean(sample_avg_prob) if sample_avg_prob else 0
        avg_sample_entropy = np.mean(sample_entropies) if sample_entropies else 0
        diagnosis_stratifier.update(
            category, is_correct, avg_sample_prob, avg_sample_entropy, 
            stats['ppl_scores'][-1] if stats['ppl_scores'] else 0
        )
        
        # Cognitive stability (mutation detection).
        traj = stability_analyzer.analyze_sample(logprobs_data, lang_code, is_correct, sample_idx)
        if traj:
            mc = traj.get('mutation_count', 0)
            if high_mutation_trajectory is None and mc >= 5 and not is_correct:
                high_mutation_trajectory = {k: v for k, v in traj.items() if k in ('logprobs', 'tokens', 'mutations_detail', 'n_tokens', 'mutation_count', 'stability_score')}
            if low_mutation_trajectory is None and mc <= 1 and is_correct and len(logprobs_data) >= 20:
                low_mutation_trajectory = {k: v for k, v in traj.items() if k in ('logprobs', 'tokens', 'mutations_detail', 'n_tokens', 'mutation_count', 'stability_score')}
    
    # Finalize stats.
    stats['cognitive_stability'] = stability_analyzer.get_summary()
    stats['stability_trajectory_high_mutation'] = high_mutation_trajectory
    stats['stability_trajectory_low_mutation'] = low_mutation_trajectory
    stats['auto_detected_confusions'] = confusion_detector.get_top_confusions(20)
    stats['diagnosis_stratified'] = diagnosis_stratifier.get_summary()
    
    # Confidence summary.
    ca = ConfidenceAnalyzer()
    stats['confidence_analysis'] = ca.get_summary(stats)
    
    # Token-level and cross-lingual summaries.
    stats['token_level_analysis'] = token_level.get_summary()
    stats['cross_lingual_analysis'] = cross_lingual.get_summary()

    # CoT uncertainty summaries.
    cot_ppl_curve = [np.mean(stats['cot_ppl_bins'][i]) if stats['cot_ppl_bins'][i] else 0 for i in range(COT_PPL_BINS)]
    sample_with_logprobs = len(stats['token_lengths'])
    conclusion_n = len(stats['conclusion_code_confidences'])
    low_warn_n = len(stats['low_probability_warnings'])
    key_step_probs = [x['probability'] for x in stats['connector_next_token_confidences']]
    key_step_logprobs = [x['logprob'] for x in stats['connector_next_token_confidences']]
    conclusion_probs = [x['probability'] for x in stats['conclusion_code_confidences']]
    conclusion_logprobs = [x['logprob'] for x in stats['conclusion_code_confidences']]
    stats['key_step_confidence'] = {
        'avg_next_token_prob': float(np.mean(key_step_probs)) if key_step_probs else 0.0,
        'avg_next_token_logprob': float(np.mean(key_step_logprobs)) if key_step_logprobs else 0.0,
        'token_count': len(key_step_probs),
    }
    stats['conclusion_confidence'] = {
        'avg_prob': float(np.mean(conclusion_probs)) if conclusion_probs else 0.0,
        'avg_logprob': float(np.mean(conclusion_logprobs)) if conclusion_logprobs else 0.0,
        'token_count': conclusion_n,
        'low_probability_warning_count': low_warn_n,
        'low_probability_warning_rate': float(low_warn_n / conclusion_n) if conclusion_n else 0.0,
        'warning_logprob_threshold': CONCLUSION_LOW_LOGPROB_THRESHOLD,
    }
    stats['cot_uncertainty_analysis'] = {
        'ppl_curve_mean': cot_ppl_curve,
        'ppl_curve_bins': COT_PPL_BINS,
        'logic_break_total': int(stats['logic_break_total']),
        'logic_break_samples': int(stats['logic_break_samples']),
        'logic_break_sample_rate': float(stats['logic_break_samples'] / sample_with_logprobs) if sample_with_logprobs else 0.0,
        'logic_break_event_rate_per_sample': float(stats['logic_break_total'] / sample_with_logprobs) if sample_with_logprobs else 0.0,
        'logic_break_examples': stats['logic_break_examples'],
        'sample_with_logprobs': sample_with_logprobs,
        'conclusion_token_count': conclusion_n,
        'low_probability_warning_count': low_warn_n,
        'low_probability_warning_rate': float(low_warn_n / conclusion_n) if conclusion_n else 0.0,
    }
    
    # Global summary.
    stats['accuracy'] = stats['correct_samples'] / stats['total_samples'] if stats['total_samples'] else 0
    stats['accuracy_both_eyes'] = stats['accuracy']
    stats['accuracy_right_eye'] = (
        stats['correct_samples_right_eye'] / stats['evaluated_samples_right_eye']
        if stats['evaluated_samples_right_eye'] else 0
    )
    stats['accuracy_left_eye'] = (
        stats['correct_samples_left_eye'] / stats['evaluated_samples_left_eye']
        if stats['evaluated_samples_left_eye'] else 0
    )
    stats['accuracy_at_least_one_eye'] = (
        stats['correct_samples_any_eye'] / stats['evaluated_samples_any_eye']
        if stats['evaluated_samples_any_eye'] else 0
    )
    stats['avg_prob_all'] = np.mean(stats['all_probs']) if stats['all_probs'] else 0
    
    return stats

# ==========================================
# 11. Utility functions
# ==========================================

def _json_serializable(obj):
    """Convert numpy types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    return obj

def get_language_code(language: str) -> str:
    mapping = {'Chinese': 'zh', 'English': 'en', 'Malay': 'ms', 'Thai': 'th'}
    return mapping.get(language, 'en')


def _ordered_languages(languages: List[str]) -> List[str]:
    """Keep preferred language order for plots/tables."""
    preferred = ['Chinese', 'English', 'Malay', 'Thai']
    existing = set(languages)
    return [l for l in preferred if l in existing] + sorted([l for l in existing if l not in preferred])


def calculate_entropy(top_logprobs: List[Dict]) -> float:
    """Shannon entropy over normalized top candidate probabilities."""
    if not top_logprobs:
        return 0.0
    probs = np.array([_get_probability(item) for item in top_logprobs], dtype=float)
    total_p = probs.sum()
    if total_p <= 0:
        return 0.0
    norm_probs = probs / total_p
    norm_probs = np.clip(norm_probs, 1e-12, 1.0)
    norm_probs = norm_probs / norm_probs.sum()
    return float(-np.sum(norm_probs * np.log(norm_probs)))

def analyze_error_attribution(direct_results: List[Dict], reasoning_results: List[Dict], 
                               language: str) -> Dict:
    """Error attribution; direct and reasoning inputs must be patient_id-aligned."""
    direct_by_pid = {r.get('patient_id'): r for r in direct_results if r.get('patient_id')}
    reasoning_by_pid = {r.get('patient_id'): r for r in reasoning_results if r.get('patient_id')}
    analyzer = ErrorAttributionAnalyzer(low_prob_threshold=0.5, high_prob_threshold=0.85)
    return analyzer.analyze_errors(direct_results, language, direct_by_pid, reasoning_by_pid)


def _is_patient_prediction_correct(result: Dict) -> bool:
    """Return whether both OD and OS diagnosis codes are correctly predicted."""
    pred, gt = extract_diagnosis_codes(result)
    return bool(
        pred.get('OD') == gt.get('OD') and
        pred.get('OS') == gt.get('OS') and
        pred.get('OD') is not None
    )


def _get_eye_prediction_status(result: Dict) -> Dict[str, bool]:
    """Return eye-level evaluable/correct flags for one prediction result."""
    pred, gt = extract_diagnosis_codes(result)
    gt_od = gt.get('OD')
    gt_os = gt.get('OS')
    pred_od = pred.get('OD')
    pred_os = pred.get('OS')
    right_evaluable = gt_od is not None
    left_evaluable = gt_os is not None
    right_correct = bool(right_evaluable and pred_od is not None and pred_od == gt_od)
    left_correct = bool(left_evaluable and pred_os is not None and pred_os == gt_os)
    return {
        'right_eye_evaluable': right_evaluable,
        'left_eye_evaluable': left_evaluable,
        'right_eye_correct': right_correct,
        'left_eye_correct': left_correct,
    }


def _update_eye_accuracy_counter(counter: Dict[str, int], eye_status: Dict[str, bool]) -> None:
    """Update per-eye numerator/denominator counters."""
    if eye_status.get('right_eye_evaluable'):
        counter['evaluated_right_eye'] += 1
        if eye_status.get('right_eye_correct'):
            counter['correct_right_eye'] += 1
    if eye_status.get('left_eye_evaluable'):
        counter['evaluated_left_eye'] += 1
        if eye_status.get('left_eye_correct'):
            counter['correct_left_eye'] += 1


def _compute_eye_accuracy(counter: Dict[str, int], eye: str) -> float:
    """Compute eye-level accuracy from accumulated counters."""
    if eye == 'right':
        denom = counter.get('evaluated_right_eye', 0)
        num = counter.get('correct_right_eye', 0)
    elif eye == 'left':
        denom = counter.get('evaluated_left_eye', 0)
        num = counter.get('correct_left_eye', 0)
    else:
        raise ValueError(f'Unsupported eye key: {eye}')
    return float(num / denom) if denom else 0.0


def _estimate_sample_metrics(result: Dict, language: str) -> Dict:
    """Estimate per-sample quality signals used for source attribution."""
    logprobs_data = result.get('output', {}).get('logprobs', []) or []
    probs = [_get_probability(item) for item in logprobs_data]
    avg_prob = float(np.mean(probs)) if probs else 0.0

    entropies = []
    for item in logprobs_data:
        top = item.get('top_logprobs', [])
        if top:
            entropies.append(calculate_entropy(top))
    avg_entropy = float(np.mean(entropies)) if entropies else 0.0

    mutation_count = 0
    if len(logprobs_data) >= 2:
        token_logprobs = np.array([_get_logprob(item) for item in logprobs_data], dtype=float)
        deltas = np.diff(token_logprobs)
        mutation_count = int(sum(1 for d in deltas if abs(float(d)) >= MUTATION_ABSOLUTE_THRESHOLDS[0]))

    code_switch_ratio = 0.0
    lang_code = get_language_code(language)
    if lang_code != 'en' and logprobs_data:
        en_pattern = re.compile(r'[a-zA-Z]{3,}')
        tokens = [item.get('token', '') or '' for item in logprobs_data]
        total = len([t for t in tokens if t.strip()])
        en_tokens = len([t for t in tokens if en_pattern.search(t)])
        code_switch_ratio = (en_tokens / total) if total > 0 else 0.0

    return {
        'avg_prob': avg_prob,
        'avg_entropy': avg_entropy,
        'mutation_count': mutation_count,
        'code_switch_ratio': code_switch_ratio,
    }


def _categorize_gain_source(direct_metrics: Dict, reasoning_metrics: Dict, language: str) -> str:
    """Heuristic attribution for cases fixed by reasoning-in-English."""
    if direct_metrics['avg_prob'] < GAIN_DIRECT_LOW_PROB and reasoning_metrics['avg_prob'] >= GAIN_REASONING_HIGH_PROB:
        return 'language_barrier_reduced'
    if reasoning_metrics['mutation_count'] + GAIN_MUTATION_DELTA_MIN <= direct_metrics['mutation_count']:
        return 'stability_improved'
    if reasoning_metrics['avg_entropy'] + GAIN_ENTROPY_DELTA_MIN <= direct_metrics['avg_entropy']:
        return 'uncertainty_reduced'
    if reasoning_metrics['avg_prob'] >= direct_metrics['avg_prob'] + GAIN_PROB_DELTA_MIN:
        return 'confidence_improved'
    return 'mixed_gain'


def _categorize_loss_source(direct_metrics: Dict, reasoning_metrics: Dict, language: str) -> str:
    """Heuristic attribution for cases degraded by reasoning-in-English."""
    if reasoning_metrics['avg_prob'] >= LOSS_OVERCONFIDENT_PROB and reasoning_metrics['avg_prob'] >= direct_metrics['avg_prob'] + LOSS_PROB_DELTA_MIN:
        return 'overconfident_wrong'
    if reasoning_metrics['mutation_count'] >= direct_metrics['mutation_count'] + LOSS_MUTATION_DELTA_MIN:
        return 'stability_degraded'
    if reasoning_metrics['avg_entropy'] >= direct_metrics['avg_entropy'] + LOSS_ENTROPY_INCREASE_MIN:
        return 'uncertainty_increased'
    if reasoning_metrics['avg_prob'] + LOSS_CONFIDENCE_DROP_MIN <= direct_metrics['avg_prob']:
        return 'confidence_dropped'
    if get_language_code(language) != 'en' and reasoning_metrics['code_switch_ratio'] >= direct_metrics['code_switch_ratio'] + LOSS_CODE_SWITCH_DELTA_MIN:
        return 'code_switch_interference'
    return 'mixed_loss'


def analyze_pairwise_gap_sources(
    baseline_results: List[Dict],
    compare_results: List[Dict],
    language: str,
    baseline_key: str = 'baseline',
    compare_key: str = 'compare',
) -> Dict:
    """
    Pair-wise comparison by patient_id:
    - outcome transition counts (both_correct / both_wrong / baseline_only_correct / compare_only_correct)
    - source attribution for improved and degraded transitions
    """
    baseline_by_pid = {r.get('patient_id'): r for r in baseline_results if r.get('patient_id') is not None}
    compare_by_pid = {r.get('patient_id'): r for r in compare_results if r.get('patient_id') is not None}
    shared_pids = sorted(set(baseline_by_pid.keys()) & set(compare_by_pid.keys()))

    outcome_counts = Counter({
        'both_correct': 0,
        'both_wrong': 0,
        'baseline_only_correct': 0,
        'compare_only_correct': 0,
    })
    gain_sources = Counter()
    loss_sources = Counter()

    for pid in shared_pids:
        baseline_item = baseline_by_pid[pid]
        compare_item = compare_by_pid[pid]
        baseline_ok = _is_patient_prediction_correct(baseline_item)
        compare_ok = _is_patient_prediction_correct(compare_item)

        if baseline_ok and compare_ok:
            outcome_counts['both_correct'] += 1
            continue
        if (not baseline_ok) and (not compare_ok):
            outcome_counts['both_wrong'] += 1
            continue

        baseline_metrics = _estimate_sample_metrics(baseline_item, language)
        compare_metrics = _estimate_sample_metrics(compare_item, language)
        if baseline_ok and (not compare_ok):
            outcome_counts['baseline_only_correct'] += 1
            loss_sources[_categorize_loss_source(baseline_metrics, compare_metrics, language)] += 1
        elif (not baseline_ok) and compare_ok:
            outcome_counts['compare_only_correct'] += 1
            gain_sources[_categorize_gain_source(baseline_metrics, compare_metrics, language)] += 1

    total_pairs = len(shared_pids)
    gain_count = outcome_counts['compare_only_correct']
    loss_count = outcome_counts['baseline_only_correct']
    outcome_rate = {k: (v / total_pairs if total_pairs else 0.0) for k, v in outcome_counts.items()}
    gain_rate = {k: (v / gain_count if gain_count else 0.0) for k, v in gain_sources.items()}
    loss_rate = {k: (v / loss_count if loss_count else 0.0) for k, v in loss_sources.items()}

    return {
        'language': language,
        'baseline_key': baseline_key,
        'compare_key': compare_key,
        'pair_count': total_pairs,
        'outcome_counts': dict(outcome_counts),
        'outcome_rates': outcome_rate,
        'gain_sources_counts': dict(gain_sources),
        'gain_sources_rates': gain_rate,
        'loss_sources_counts': dict(loss_sources),
        'loss_sources_rates': loss_rate,
        'net_gain_count': int(gain_count - loss_count),
        'net_gain_rate': float((gain_count - loss_count) / total_pairs) if total_pairs else 0.0,
    }


def analyze_gap_sources_across_languages(
    direct_results_by_lang: Dict[str, List[Dict]],
    reasoning_results_by_lang: Dict[str, List[Dict]],
    languages: List[str],
    baseline_key: str = 'baseline',
    compare_key: str = 'compare',
) -> Dict:
    """Run pairwise gap-source analysis for all available languages."""
    per_language = {}
    overall_outcomes = Counter({
        'both_correct': 0,
        'both_wrong': 0,
        'baseline_only_correct': 0,
        'compare_only_correct': 0,
    })
    overall_gain_sources = Counter()
    overall_loss_sources = Counter()
    total_pairs = 0

    for lang in languages:
        if lang == 'English':
            continue
        if lang not in direct_results_by_lang or lang not in reasoning_results_by_lang:
            continue
        analysis = analyze_pairwise_gap_sources(
            baseline_results=direct_results_by_lang[lang],
            compare_results=reasoning_results_by_lang[lang],
            language=lang,
            baseline_key=baseline_key,
            compare_key=compare_key,
        )
        per_language[lang] = analysis
        total_pairs += analysis['pair_count']
        overall_outcomes.update(analysis['outcome_counts'])
        overall_gain_sources.update(analysis['gain_sources_counts'])
        overall_loss_sources.update(analysis['loss_sources_counts'])

    gain_total = overall_outcomes['compare_only_correct']
    loss_total = overall_outcomes['baseline_only_correct']
    overall = {
        'baseline_key': baseline_key,
        'compare_key': compare_key,
        'pair_count': total_pairs,
        'outcome_counts': dict(overall_outcomes),
        'outcome_rates': {k: (v / total_pairs if total_pairs else 0.0) for k, v in overall_outcomes.items()},
        'gain_sources_counts': dict(overall_gain_sources),
        'gain_sources_rates': {k: (v / gain_total if gain_total else 0.0) for k, v in overall_gain_sources.items()},
        'loss_sources_counts': dict(overall_loss_sources),
        'loss_sources_rates': {k: (v / loss_total if loss_total else 0.0) for k, v in overall_loss_sources.items()},
        'net_gain_count': int(gain_total - loss_total),
        'net_gain_rate': float((gain_total - loss_total) / total_pairs) if total_pairs else 0.0,
    }
    return {'per_language': per_language, 'overall': overall}


def _prediction_signature(result: Dict) -> Tuple[Optional[int], Optional[int]]:
    """Prediction signature as (OD_code, OS_code)."""
    pred, _ = extract_diagnosis_codes(result)
    return pred.get('OD'), pred.get('OS')


def _is_semantically_consistent(result_a: Dict, result_b: Dict) -> bool:
    """
    Semantic consistency proxy for A2≈A3:
    both eyes' diagnosis codes are equal and non-empty.
    """
    sa = _prediction_signature(result_a)
    sb = _prediction_signature(result_b)
    return bool(sa[0] is not None and sb[0] is not None and sa[0] == sb[0] and sa[1] == sb[1])


def analyze_tri_perspective_verification(
    direct_results_by_lang: Dict[str, List[Dict]],
    reasoning_results_by_lang: Dict[str, List[Dict]],
    pivot_results_by_lang: Dict[str, List[Dict]],
    languages: List[str],
) -> Dict:
    """
    Tri-perspective verification:
    1) Run A2(reasoning) and A3(translation-pivot)
    2) If A2≈A3 => choose A2
    3) If conflict => use A1(direct) as majority-support resolver
       - A1 supports A2 => choose A2
       - A1 supports A3 => choose A3
       - unresolved => confidence tie-break between A2/A3
    """
    per_language = {}
    overall_resolution_counts = Counter()
    overall_source_counts = Counter()
    overall_total = 0
    overall_correct = 0
    overall_final_eye_counter = Counter({
        'evaluated_right_eye': 0,
        'evaluated_left_eye': 0,
        'correct_right_eye': 0,
        'correct_left_eye': 0,
    })

    for lang in languages:
        if lang == 'English':
            continue
        if lang not in reasoning_results_by_lang or lang not in pivot_results_by_lang:
            continue

        a1_by_pid = {r.get('patient_id'): r for r in direct_results_by_lang.get(lang, []) if r.get('patient_id') is not None}
        a2_by_pid = {r.get('patient_id'): r for r in reasoning_results_by_lang.get(lang, []) if r.get('patient_id') is not None}
        a3_by_pid = {r.get('patient_id'): r for r in pivot_results_by_lang.get(lang, []) if r.get('patient_id') is not None}
        shared_pids = sorted(set(a2_by_pid.keys()) & set(a3_by_pid.keys()))
        if not shared_pids:
            continue

        resolution_counts = Counter()
        source_counts = Counter()
        decisions = []
        final_correct = 0
        a2_correct = 0
        a3_correct = 0
        a1_correct = 0
        a1_available = 0
        a1_eye_counter = Counter({
            'evaluated_right_eye': 0,
            'evaluated_left_eye': 0,
            'correct_right_eye': 0,
            'correct_left_eye': 0,
        })
        a2_eye_counter = Counter({
            'evaluated_right_eye': 0,
            'evaluated_left_eye': 0,
            'correct_right_eye': 0,
            'correct_left_eye': 0,
        })
        a3_eye_counter = Counter({
            'evaluated_right_eye': 0,
            'evaluated_left_eye': 0,
            'correct_right_eye': 0,
            'correct_left_eye': 0,
        })
        final_eye_counter = Counter({
            'evaluated_right_eye': 0,
            'evaluated_left_eye': 0,
            'correct_right_eye': 0,
            'correct_left_eye': 0,
        })

        for pid in shared_pids:
            a2_item = a2_by_pid[pid]
            a3_item = a3_by_pid[pid]
            a1_item = a1_by_pid.get(pid)

            a2_eye_status = _get_eye_prediction_status(a2_item)
            _update_eye_accuracy_counter(a2_eye_counter, a2_eye_status)
            a3_eye_status = _get_eye_prediction_status(a3_item)
            _update_eye_accuracy_counter(a3_eye_counter, a3_eye_status)

            if _is_patient_prediction_correct(a2_item):
                a2_correct += 1
            if _is_patient_prediction_correct(a3_item):
                a3_correct += 1

            if a1_item is not None:
                a1_available += 1
                if _is_patient_prediction_correct(a1_item):
                    a1_correct += 1
                a1_eye_status = _get_eye_prediction_status(a1_item)
                _update_eye_accuracy_counter(a1_eye_counter, a1_eye_status)

            if _is_semantically_consistent(a2_item, a3_item):
                chosen_key = 'reasoning'
                chosen_item = a2_item
                resolution = 'a2_a3_consistent_choose_a2'
            else:
                a1_supports_a2 = bool(a1_item is not None and _is_semantically_consistent(a1_item, a2_item))
                a1_supports_a3 = bool(a1_item is not None and _is_semantically_consistent(a1_item, a3_item))
                if a1_supports_a2 and not a1_supports_a3:
                    chosen_key = 'reasoning'
                    chosen_item = a2_item
                    resolution = 'a2_a3_conflict_a1_supports_a2'
                elif a1_supports_a3 and not a1_supports_a2:
                    chosen_key = 'translate_pivot'
                    chosen_item = a3_item
                    resolution = 'a2_a3_conflict_a1_supports_a3'
                else:
                    a2_prob = _estimate_sample_metrics(a2_item, lang).get('avg_prob', 0.0)
                    a3_prob = _estimate_sample_metrics(a3_item, lang).get('avg_prob', 0.0)
                    if a2_prob >= a3_prob:
                        chosen_key = 'reasoning'
                        chosen_item = a2_item
                    else:
                        chosen_key = 'translate_pivot'
                        chosen_item = a3_item
                    resolution = 'a2_a3_conflict_tiebreak_confidence' if a1_item is not None else 'a2_a3_conflict_no_a1_tiebreak_confidence'

            source_counts[chosen_key] += 1
            resolution_counts[resolution] += 1

            chosen_correct = _is_patient_prediction_correct(chosen_item)
            if chosen_correct:
                final_correct += 1
            chosen_eye_status = _get_eye_prediction_status(chosen_item)
            _update_eye_accuracy_counter(final_eye_counter, chosen_eye_status)

            decisions.append({
                'patient_id': pid,
                'a1_pred': _prediction_signature(a1_item) if a1_item is not None else None,
                'a2_pred': _prediction_signature(a2_item),
                'a3_pred': _prediction_signature(a3_item),
                'resolution': resolution,
                'chosen_source': chosen_key,
                'chosen_pred': _prediction_signature(chosen_item),
                'is_correct': bool(chosen_correct),
                'is_correct_right_eye': bool(chosen_eye_status.get('right_eye_correct')),
                'is_correct_left_eye': bool(chosen_eye_status.get('left_eye_correct')),
            })

        total = len(shared_pids)
        lang_summary = {
            'shared_case_count': total,
            'a2_accuracy_on_shared': float(a2_correct / total) if total else 0.0,
            'a3_accuracy_on_shared': float(a3_correct / total) if total else 0.0,
            'a2_accuracy_right_eye_on_shared': _compute_eye_accuracy(a2_eye_counter, 'right'),
            'a2_accuracy_left_eye_on_shared': _compute_eye_accuracy(a2_eye_counter, 'left'),
            'a3_accuracy_right_eye_on_shared': _compute_eye_accuracy(a3_eye_counter, 'right'),
            'a3_accuracy_left_eye_on_shared': _compute_eye_accuracy(a3_eye_counter, 'left'),
            'a1_available_count': int(a1_available),
            'a1_accuracy_on_available': float(a1_correct / a1_available) if a1_available else 0.0,
            'a1_accuracy_right_eye_on_available': _compute_eye_accuracy(a1_eye_counter, 'right'),
            'a1_accuracy_left_eye_on_available': _compute_eye_accuracy(a1_eye_counter, 'left'),
            'final_accuracy': float(final_correct / total) if total else 0.0,
            'final_accuracy_right_eye': _compute_eye_accuracy(final_eye_counter, 'right'),
            'final_accuracy_left_eye': _compute_eye_accuracy(final_eye_counter, 'left'),
            'final_evaluated_count_right_eye': int(final_eye_counter.get('evaluated_right_eye', 0)),
            'final_evaluated_count_left_eye': int(final_eye_counter.get('evaluated_left_eye', 0)),
            'resolution_counts': dict(resolution_counts),
            'resolution_rates': {k: (v / total if total else 0.0) for k, v in resolution_counts.items()},
            'source_usage_counts': dict(source_counts),
            'source_usage_rates': {k: (v / total if total else 0.0) for k, v in source_counts.items()},
            'decisions': decisions,
        }
        per_language[lang] = lang_summary
        overall_total += total
        overall_correct += final_correct
        overall_resolution_counts.update(resolution_counts)
        overall_source_counts.update(source_counts)
        overall_final_eye_counter.update(final_eye_counter)

    overall = {
        'shared_case_count': overall_total,
        'final_accuracy': float(overall_correct / overall_total) if overall_total else 0.0,
        'final_accuracy_right_eye': _compute_eye_accuracy(overall_final_eye_counter, 'right'),
        'final_accuracy_left_eye': _compute_eye_accuracy(overall_final_eye_counter, 'left'),
        'final_evaluated_count_right_eye': int(overall_final_eye_counter.get('evaluated_right_eye', 0)),
        'final_evaluated_count_left_eye': int(overall_final_eye_counter.get('evaluated_left_eye', 0)),
        'resolution_counts': dict(overall_resolution_counts),
        'resolution_rates': {k: (v / overall_total if overall_total else 0.0) for k, v in overall_resolution_counts.items()},
        'source_usage_counts': dict(overall_source_counts),
        'source_usage_rates': {k: (v / overall_total if overall_total else 0.0) for k, v in overall_source_counts.items()},
    }
    return {'per_language': per_language, 'overall': overall}


def _normalize_diagnosis_code(value):
    """Normalize diagnosis code to int or uppercase string."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
        return str(value).strip().upper()
    value_str = str(value).strip()
    if value_str == '' or value_str.lower() in {'none', 'null', 'nan'}:
        return None
    if re.fullmatch(r'[-+]?\d+(\.0+)?', value_str):
        return int(float(value_str))
    return value_str.upper()


def extract_diagnosis_codes(result: Dict) -> Tuple[Dict, Dict]:
    """Extract and normalize predicted/ground-truth diagnosis codes."""
    predicted, ground_truth = {}, {}
    parsed_response = result.get('output', {}).get('parsed_response', {})
    
    od_keys = ['\u53f3\u773c', 'Right Eye', 'right eye', 'right_eye', 'OD', 'Right', 'mata_kanan', '\u0e15\u0e32\u0e02\u0e27\u0e32']
    os_keys = ['\u5de6\u773c', 'Left Eye', 'left eye', 'left_eye', 'OS', 'Left', 'mata_kiri', '\u0e15\u0e32\u0e0b\u0e49\u0e32\u0e22']
    code_keys = ['\u8bca\u65ad\u4ee3\u7801', 'diagnosis_code', 'kod_diagnosis', '\u0e23\u0e2b\u0e31\u0e2a\u0e01\u0e32\u0e23\u0e27\u0e34\u0e19\u0e34\u0e08\u0e09\u0e31\u0e22']

    def find_code(eye_keys):
        for ek in eye_keys:
            if ek in parsed_response:
                data = parsed_response[ek]
                if isinstance(data, dict):
                    for ck in code_keys:
                        if ck in data:
                            normalized = _normalize_diagnosis_code(data.get(ck))
                            if normalized is not None:
                                return normalized
        return None

    predicted['OD'] = find_code(od_keys)
    predicted['OS'] = find_code(os_keys)
    gt_data = result.get('ground_truth', {})
    ground_truth['OD'] = _normalize_diagnosis_code(gt_data.get('OD', {}).get('diagnosis_code'))
    ground_truth['OS'] = _normalize_diagnosis_code(gt_data.get('OS', {}).get('diagnosis_code'))
    return predicted, ground_truth

# ==========================================
# 12. Visualization: diagnosis-stratified performance
# ==========================================

def _add_bar_labels(ax, label_items: List[Tuple], value_fmt: str, is_int: bool = False):
    """Add value labels above bars with y-range-aware spacing."""
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * _BAR_LABEL_GAP_FRAC
    for bar, val, err in label_items:
        if val is None or not np.isfinite(val):
            continue
        err_val = err if (err is not None and np.isfinite(err)) else 0
        y_top = bar.get_height() + err_val
        txt = f'{int(val)}' if is_int else value_fmt.format(val)
        ax.text(bar.get_x() + bar.get_width()/2, y_top + gap, txt, ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL, rotation=0)


def _add_reference_line_with_value(ax, value: float, value_fmt: str = '{:.2f}'):
    """Draw red reference line and annotate its numeric value near the right edge."""
    ax.axhline(value, color='red', linestyle='-', linewidth=1.5, label=RED_LINE_LABEL)
    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-12)
    y_text = value + y_span * 0.01
    va = 'bottom'
    if y_text > ymax:
        y_text = value - y_span * 0.01
        va = 'top'
    ax.text(
        0.99,
        y_text,
        value_fmt.format(value),
        transform=ax.get_yaxis_transform(),
        ha='right',
        va=va,
        color='red',
        fontsize=_PLOT_FONTSIZE_LABEL,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 1},
        clip_on=False
    )


def _draw_diagnosis_row(axes_row, all_data: Dict, method_name: str, categories: List[str],
                        languages: List[str], colors: List, x: np.ndarray, width: float, y_max_acc: float):
    """Draw Acc/PPL/Count/Entropy panels for one method."""
    display_languages = list(languages)
    category_labels = [DIAGNOSIS_CATEGORY_DISPLAY.get(cat, cat) for cat in categories]
    ax1, ax2, ax3, ax4 = axes_row

    def _metric_with_count_guard(strat_item: Dict, metric_key: str, err_key: Optional[str] = None,
                                 scale: float = 1.0) -> Tuple[float, float]:
        n = int(strat_item.get('sample_count', 0) or 0)
        if n <= 0:
            return np.nan, 0.0
        val = _safe_float(strat_item.get(metric_key), 0.0) * scale
        err = _safe_float(strat_item.get(err_key), 0.0) * scale if err_key else 0.0
        return val, err

    # --- Accuracy ---
    label_items_1 = []
    for i, lang in enumerate(display_languages):
        strat = all_data[lang].get('diagnosis_stratified', {})
        accs, errs = [], []
        for cat in categories:
            acc, err = _metric_with_count_guard(strat.get(cat, {}), 'accuracy', 'accuracy_std', scale=100.0)
            accs.append(acc)
            errs.append(err)
        bars = ax1.bar(x + i*width, accs, width, label=lang, color=colors[i], alpha=0.8, yerr=errs, capsize=2, error_kw={'elinewidth': 1})
        for j, (bar, val) in enumerate(zip(bars, accs)):
            label_items_1.append((bar, val, errs[j] if j < len(errs) else 0))
    ax1.set_xticks(x + width*(len(display_languages)-1)/2)
    ax1.set_xticklabels(category_labels, rotation=20, ha='right', fontsize=_PLOT_FONTSIZE_LABEL)
    ax1.set_ylabel('Accuracy (%)', fontsize=_PLOT_FONTSIZE_LABEL)
    ax1.set_title(f'Accuracy ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
    ax1.set_ylim(0, y_max_acc)
    ax1.tick_params(axis='y', labelsize=_PLOT_FONTSIZE_LABEL)
    ax1.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    _add_bar_labels(ax1, label_items_1, '{:.1f}')

    # --- PPL ---
    label_items_2 = []
    for i, lang in enumerate(display_languages):
        strat = all_data[lang].get('diagnosis_stratified', {})
        ppls, errs = [], []
        for cat in categories:
            ppl, err = _metric_with_count_guard(strat.get(cat, {}), 'avg_ppl', 'avg_ppl_std')
            ppls.append(ppl)
            errs.append(err)
        bars = ax2.bar(x + i*width, ppls, width, label=lang, color=colors[i], alpha=0.8, yerr=errs, capsize=2, error_kw={'elinewidth': 1})
        for j, (bar, val) in enumerate(zip(bars, ppls)):
            label_items_2.append((bar, val, errs[j] if j < len(errs) else 0))
    ax2.set_xticks(x + width*(len(display_languages)-1)/2)
    ax2.set_xticklabels(category_labels, rotation=20, ha='right', fontsize=_PLOT_FONTSIZE_LABEL)
    ax2.set_ylabel('Mean PPL', fontsize=_PLOT_FONTSIZE_LABEL)
    ax2.set_title(f'Perplexity ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
    ax2.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    ax2.tick_params(axis='y', labelsize=_PLOT_FONTSIZE_LABEL)
    _add_bar_labels(ax2, label_items_2, '{:.1f}')

    # --- Sample Count ---
    label_items_3 = []
    for i, lang in enumerate(display_languages):
        strat = all_data[lang].get('diagnosis_stratified', {})
        counts = [strat.get(cat, {}).get('sample_count', 0) for cat in categories]
        bars = ax3.bar(x + i*width, counts, width, label=lang, color=colors[i], alpha=0.8)
        for bar, val in zip(bars, counts):
            label_items_3.append((bar, val, None))
    ax3.set_xticks(x + width*(len(display_languages)-1)/2)
    ax3.set_xticklabels(category_labels, rotation=20, ha='right', fontsize=_PLOT_FONTSIZE_LABEL)
    ax3.set_ylabel('Sample Count', fontsize=_PLOT_FONTSIZE_LABEL)
    ax3.set_title(f'Distribution ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
    ax3.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    ax3.tick_params(axis='y', labelsize=_PLOT_FONTSIZE_LABEL)
    _add_bar_labels(ax3, label_items_3, '{}', is_int=True)

    # --- Entropy ---
    label_items_4 = []
    for i, lang in enumerate(display_languages):
        strat = all_data[lang].get('diagnosis_stratified', {})
        ents, errs = [], []
        for cat in categories:
            ent, err = _metric_with_count_guard(strat.get(cat, {}), 'avg_entropy', 'avg_entropy_std')
            ents.append(ent)
            errs.append(err)
        bars = ax4.bar(x + i*width, ents, width, label=lang, color=colors[i], alpha=0.8, yerr=errs, capsize=2, error_kw={'elinewidth': 1})
        for j, (bar, val) in enumerate(zip(bars, ents)):
            label_items_4.append((bar, val, errs[j] if j < len(errs) else 0))
    ax4.set_xticks(x + width*(len(display_languages)-1)/2)
    ax4.set_xticklabels(category_labels, rotation=20, ha='right', fontsize=_PLOT_FONTSIZE_LABEL)
    ax4.set_ylabel('Mean Entropy', fontsize=_PLOT_FONTSIZE_LABEL)
    ax4.set_title(f'Uncertainty ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
    ax4.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    ax4.tick_params(axis='y', labelsize=_PLOT_FONTSIZE_LABEL)
    _add_bar_labels(ax4, label_items_4, '{:.2f}')


def plot_diagnosis_stratified_performance(all_data: Dict[str, Dict], output_path: Path, method_name: str, english_ref: Dict = None):
    """Diagnosis-stratified performance for one method."""
    languages = sorted(all_data.keys())
    display_languages = list(languages)
    all_categories = set()
    for lang_data in all_data.values():
        if lang_data.get('diagnosis_stratified'):
            all_categories.update(lang_data['diagnosis_stratified'].keys())
    if not all_categories:
        return
    extra_categories = sorted([c for c in all_categories if c not in DIAGNOSIS_CATEGORY_ORDER])
    categories = DIAGNOSIS_CATEGORY_ORDER + extra_categories
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Disease-Stratified Performance: {method_name}', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    colors = ['#9B59B6', '#27AE60', '#3498DB', '#F39C12'][:len(languages)]
    x = np.arange(len(categories))
    width = 0.8 / max(len(display_languages), 1)
    all_accs_flat = []
    for lang in display_languages:
        strat = all_data[lang].get('diagnosis_stratified', {})
        for cat in categories:
            item = strat.get(cat, {})
            if (item.get('sample_count') or 0) > 0:
                all_accs_flat.append(_safe_float(item.get('accuracy'), 0.0) * 100.0)
    y_max_acc = min(115, max(all_accs_flat) * 1.2 + 10) if all_accs_flat else 100
    _draw_diagnosis_row([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], all_data, method_name, categories, languages, colors, x, width, y_max_acc)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_diagnosis_stratified_combined(all_data_by_method: Dict[str, Dict], output_path: Path,
                                       method_names: Dict[str, str], english_ref: Dict = None):
    """Diagnosis-stratified comparison across all available methods."""
    method_keys = sorted(all_data_by_method.keys())
    if len(method_keys) < 1:
        return
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = list(languages)
    all_categories = set()
    for key in method_keys:
        for lang_data in list(all_data_by_method[key].values()):
            if lang_data.get('diagnosis_stratified'):
                all_categories.update(lang_data['diagnosis_stratified'].keys())
    if not all_categories:
        return
    extra_categories = sorted([c for c in all_categories if c not in DIAGNOSIS_CATEGORY_ORDER])
    categories = DIAGNOSIS_CATEGORY_ORDER + extra_categories
    colors = ['#9B59B6', '#27AE60', '#3498DB', '#F39C12'][:len(languages)]
    x = np.arange(len(categories))
    width = 0.8 / max(len(display_languages), 1)
    all_accs_flat = []
    for key in method_keys:
        d = all_data_by_method[key]
        for lang in display_languages:
            strat = d.get(lang, {}).get('diagnosis_stratified', {})
            for cat in categories:
                item = strat.get(cat, {})
                if (item.get('sample_count') or 0) > 0:
                    all_accs_flat.append(_safe_float(item.get('accuracy'), 0.0) * 100.0)
    y_max_acc = min(115, max(all_accs_flat) * 1.2 + 10) if all_accs_flat else 100

    n_rows = len(method_keys)
    fig = plt.figure(figsize=(20, max(7, 6.5 * n_rows)))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, height_ratios=[1] * n_rows, hspace=0.35, wspace=0.25)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(n_rows)])
    if n_rows == 1:
        fig.suptitle(
            f"Disease-Stratified Performance: {method_names.get(method_keys[0], method_keys[0])}",
            fontsize=_PLOT_FONTSIZE_SUPTITLE,
            fontweight='bold'
        )
    else:
        fig.suptitle(
            'Disease-Stratified Performance: Multi-Method Comparison',
            fontsize=_PLOT_FONTSIZE_SUPTITLE,
            fontweight='bold'
        )
    for row, key in enumerate(method_keys):
        _draw_diagnosis_row(
            axes[row],
            all_data_by_method[key],
            method_names.get(key, key),
            categories,
            languages,
            colors,
            x,
            width,
            y_max_acc
        )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.08, hspace=0.35, wspace=0.25)
    plt.savefig(output_path, dpi=300)
    plt.close()

def _draw_one_confusion_panel(ax, confusions: List, lang: str, method_name: str):
    """Draw top confusion pairs for one language/method panel."""
    if not confusions:
        ax.text(0.5, 0.5, 'No confusions detected', ha='center', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_title(f'{lang} ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
        return
    top_confusions = confusions[:15]
    labels = [f'{t1[:8]}<->{t2[:8]}' for t1, t2, _ in top_confusions]
    counts = [cnt for _, _, cnt in top_confusions]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color='#E74C3C', alpha=0.7)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{int(cnt)}', ha='left', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_PLOT_FONTSIZE_LABEL)
    ax.invert_yaxis()
    ax.set_xlabel('Confusion Count', fontsize=_PLOT_FONTSIZE_LABEL)
    ax.set_title(f'{lang} ({method_name})', fontsize=_PLOT_FONTSIZE_TITLE)
    ax.tick_params(axis='both', labelsize=_PLOT_FONTSIZE_LABEL)
    ax.grid(axis='x', alpha=0.3)


def plot_auto_detected_confusions(all_data: Dict[str, Dict], output_path: Path, method_name: str):
    """Visualize auto-detected confusion pairs for a single method."""
    languages = [l for l in sorted(all_data.keys()) if l not in DISPLAY_LANGUAGES_EXCLUDE]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Auto-Detected Semantic Confusions: {method_name}', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    axes_flat = axes.flatten()
    for i, lang in enumerate(languages):
        if i >= len(axes_flat):
            break
        _draw_one_confusion_panel(axes_flat[i], all_data[lang].get('auto_detected_confusions', []), lang, method_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_auto_confusions_combined(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str]):
    """Visualize confusion pairs across methods (rows=methods, cols=languages)."""
    method_keys = sorted(all_data_by_method.keys())
    if len(method_keys) < 1:
        return
    languages = [l for l in sorted(all_data_by_method[method_keys[0]].keys()) if l not in DISPLAY_LANGUAGES_EXCLUDE]
    if not languages:
        return
    n_rows = len(method_keys)
    n_cols = max(3, len(languages))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, max(7, 4.5 * n_rows)))
    if n_rows == 1:
        axes = np.array([axes])
    title = 'Auto-Detected Semantic Confusions: Multi-Method Comparison' if n_rows > 1 else 'Auto-Detected Semantic Confusions'
    fig.suptitle(title, fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(languages):
                lang = languages[col]
                _draw_one_confusion_panel(ax, data.get(lang, {}).get('auto_detected_confusions', []), lang, name)
            else:
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================
# 13. Visualization: confidence/error/token-level/cross-lingual
# ==========================================

def plot_confidence_analysis(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], english_ref: Dict = None):
    """Plot confidence for correct vs incorrect, cross-lingual confidence, and entropy."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    colors = ['#9B59B6', '#27AE60', '#3498DB', '#F39C12'][:len(display_languages)]
    
    n_methods = len(method_keys)
    fig, axes = plt.subplots(n_methods, 3, figsize=(14, 5 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Confidence Analysis: Correct vs Incorrect, Cross-lingual, Entropy', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    
    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        ax1, ax2, ax3 = axes[row]
        # Correct vs Incorrect logprob
        correct_probs = [data.get(l, {}).get('confidence_analysis', {}).get('avg_prob_correct', 0) for l in display_languages]
        incorrect_probs = [data.get(l, {}).get('confidence_analysis', {}).get('avg_prob_incorrect', 0) for l in display_languages]
        x = np.arange(len(display_languages))
        w = 0.35
        b1 = ax1.bar(x - w/2, correct_probs, w, label='Correct', color='#27AE60', alpha=0.8)
        b2 = ax1.bar(x + w/2, incorrect_probs, w, label='Incorrect', color='#E74C3C', alpha=0.8)
        for bar, val in zip(b1, correct_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
        for bar, val in zip(b2, incorrect_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_languages)
        ax1.set_ylabel('Avg Probability')
        ax1.set_title(f'Correct vs Incorrect ({name})')
        ax1.set_ylim(0, 1.05)
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            ca = english_ref.get('confidence_analysis') or {}
            _add_reference_line_with_value(ax1, ca.get('avg_prob_correct', 0), '{:.2f}')
        ax1.legend()
        # Cross-lingual avg logprob
        avg_probs = [np.mean(data.get(l, {}).get('all_probs', []) or [0]) for l in display_languages]
        bars2 = ax2.bar(x, avg_probs, color=colors, alpha=0.8)
        for bar, val in zip(bars2, avg_probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_languages)
        ax2.set_ylabel('Avg Token Probability')
        ax2.set_title(f'Cross-lingual Confidence ({name})')
        ax2.set_ylim(0, 1.05)
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            _add_reference_line_with_value(ax2, np.mean(english_ref.get('all_probs') or [0]), '{:.2f}')
            ax2.legend(fontsize=_PLOT_FONTSIZE_LABEL)
        # Entropy
        avg_ents = [data.get(l, {}).get('confidence_analysis', {}).get('avg_entropy', 0) for l in display_languages]
        bars3 = ax3.bar(x, avg_ents, color=colors, alpha=0.8)
        for bar, val in zip(bars3, avg_ents):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
        ax3.set_xticks(x)
        ax3.set_xticklabels(display_languages)
        ax3.set_ylabel('Avg Entropy')
        ax3.set_title(f'Uncertainty ({name})')
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            _add_reference_line_with_value(ax3, (english_ref.get('confidence_analysis') or {}).get('avg_entropy', 0), '{:.2f}')
            ax3.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_accuracy_comparison(
    all_data_by_method: Dict[str, Dict],
    output_path: Path,
    method_names: Dict[str, str],
    english_ref: Dict = None,
    accuracy_key: str = 'accuracy',
    title: str = 'Overall Accuracy Comparison Across Methods',
):
    """Accuracy comparison across methods for one metric key."""
    if not all_data_by_method:
        return
    method_keys = list(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    if not display_languages:
        return

    # Keep method order consistent with config when available.
    preferred_methods = ['direct', 'reasoning', 'translate_pivot']
    ordered = [k for k in preferred_methods if k in method_keys]
    ordered += [k for k in method_keys if k not in ordered]
    method_keys = ordered

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(display_languages))
    width = 0.8 / max(len(method_keys), 1)
    colors = ['#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD'][:len(method_keys)]

    all_vals = []
    for i, key in enumerate(method_keys):
        data = all_data_by_method[key]
        vals = [data.get(lang, {}).get(accuracy_key, 0.0) * 100 for lang in display_languages]
        bars = ax.bar(
            x + (i - (len(method_keys) - 1) / 2) * width,
            vals,
            width,
            label=method_names.get(key, key),
            color=colors[i],
            alpha=0.85
        )
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f'{v:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=_PLOT_FONTSIZE_LABEL
                )
        all_vals.extend(vals)

    # English reference baseline from direct English (legacy behavior).
    if english_ref and ENGLISH_AS_REFERENCE_LINE:
        eng_acc = _safe_float(english_ref.get(accuracy_key, english_ref.get('accuracy', 0.0)), 0.0) * 100.0
        ax.axhline(eng_acc, color='red', linestyle='-', linewidth=1.8, label='English (reference)')
        ax.text(
            0.99,
            eng_acc,
            f'{eng_acc:.1f}',
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='bottom',
            color='red',
            fontsize=_PLOT_FONTSIZE_LABEL,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 1},
            clip_on=False
        )
        all_vals.append(eng_acc)

    y_max = min(115, max(all_vals) * 1.2 + 8) if all_vals else 100
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(display_languages, fontsize=_PLOT_FONTSIZE_LABEL)
    for tick in ax.get_xticklabels():
        tick.set_color(LANGUAGE_LABEL_COLORS.get(tick.get_text(), 'black'))
    ax.set_ylabel('Accuracy (%)', fontsize=_PLOT_FONTSIZE_LABEL)
    ax.set_title(title, fontsize=_PLOT_FONTSIZE_TITLE)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=_PLOT_FONTSIZE_LABEL)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cot_uncertainty_analysis(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], english_ref: Dict = None):
    """
    Plot CoT uncertainty signals:
    - Perplexity curve across normalized token positions
    - Logic-break rate and low-probability conclusion warning rate
    """
    method_keys = sorted(all_data_by_method.keys())
    if not method_keys:
        return
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    if not display_languages:
        return

    n_methods = len(method_keys)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 5 * n_methods))
    if n_methods == 1:
        axes = np.array([axes])
    fig.suptitle('CoT Uncertainty Analysis: Perplexity Curve and Logic-Break Signals', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')

    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        ax1, ax2 = axes[row]

        # Left: mean PPL curve per language.
        for lang in display_languages:
            cua = data.get(lang, {}).get('cot_uncertainty_analysis', {})
            curve = cua.get('ppl_curve_mean', [])
            if not curve:
                continue
            xs = np.linspace(0, 1, len(curve))
            ax1.plot(xs, curve, marker='o', linewidth=1.5, markersize=3, label=lang)
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            curve_en = (english_ref.get('cot_uncertainty_analysis') or {}).get('ppl_curve_mean', [])
            if curve_en:
                xs = np.linspace(0, 1, len(curve_en))
                ax1.plot(xs, curve_en, linestyle='--', linewidth=1.8, color='red', label='English (reference)')
        ax1.set_xlabel('Normalized CoT Position')
        ax1.set_ylabel('Mean Token PPL')
        ax1.set_title(f'Perplexity Curve ({name})')
        ax1.grid(alpha=0.3)
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(fontsize=_PLOT_FONTSIZE_LABEL)

        # Right: logic-break sample rate vs low-probability conclusion warning rate.
        x = np.arange(len(display_languages))
        w = 0.38
        logic_rates = []
        warning_rates = []
        for lang in display_languages:
            cua = data.get(lang, {}).get('cot_uncertainty_analysis', {})
            logic_rates.append(cua.get('logic_break_sample_rate', 0.0) * 100)
            warning_rates.append(cua.get('low_probability_warning_rate', 0.0) * 100)
        bars1 = ax2.bar(x - w / 2, logic_rates, w, label='Logic-break sample rate', color='#E67E22', alpha=0.85)
        bars2 = ax2.bar(x + w / 2, warning_rates, w, label='Low-prob conclusion warning rate', color='#C0392B', alpha=0.85)
        for bar, val in zip(bars1, logic_rates):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + _BAR_LABEL_GAP_SMALL, f'{val:.1f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL - 1)
        for bar, val in zip(bars2, warning_rates):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + _BAR_LABEL_GAP_SMALL, f'{val:.1f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL - 1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_languages)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Rate (%)')
        ax2.set_title(f'Logic Break / Low-Prob Warning ({name})')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=_PLOT_FONTSIZE_LABEL)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_attribution(error_attribution_by_lang: Dict[str, Dict], output_path: Path):
    """Plot error attribution distribution by language."""
    languages = _ordered_languages(list(error_attribution_by_lang.keys()))
    if not languages:
        return
    dists = [error_attribution_by_lang.get(l, {}).get('distribution', {}) for l in languages]
    err_types = ['knowledge_gap', 'reasoning_failure', 'language_barrier', 'mixed']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(languages))
    w = 0.2
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#95A5A6']
    for i, et in enumerate(err_types):
        vals = [d.get(et, 0) * 100 for d in dists]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=et.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel('Error Type (%)')
    ax.set_title('Error Attribution: Knowledge Gap / Reasoning Failure / Language Barrier')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_attribution_combined(all_error_attribution_by_pair: Dict[str, Dict[str, Dict]],
                                    output_path: Path,
                                    method_names: Dict[str, str] = None):
    """Plot all pairwise error-attribution results in one figure."""
    if not all_error_attribution_by_pair:
        return
    pair_tags = sorted(all_error_attribution_by_pair.keys())
    err_types = ['knowledge_gap', 'reasoning_failure', 'language_barrier', 'mixed']
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#95A5A6']
    labels = [et.replace('_', ' ').title() for et in err_types]

    n_rows = len(pair_tags)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(4.5, 3.8 * n_rows)), squeeze=False)
    fig.suptitle('Error Attribution by Language (All Method Pairs)', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')

    for row, pair_tag in enumerate(pair_tags):
        ax = axes[row, 0]
        error_attribution_by_lang = all_error_attribution_by_pair[pair_tag]
        languages = _ordered_languages(list(error_attribution_by_lang.keys()))
        if not languages:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
            ax.set_axis_off()
            continue
        dists = [error_attribution_by_lang.get(l, {}).get('distribution', {}) for l in languages]
        x = np.arange(len(languages))
        w = 0.2
        for i, et in enumerate(err_types):
            vals = [d.get(et, 0) * 100 for d in dists]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, color=colors[i], alpha=0.85, label=labels[i] if row == 0 else None)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.0f}',
                        ha='center',
                        va='bottom',
                        fontsize=_PLOT_FONTSIZE_LABEL - 1
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Error Type (%)')
        ax.grid(axis='y', alpha=0.25)
        if '_vs_' in pair_tag:
            left, right = pair_tag.split('_vs_', 1)
            left_name = method_names.get(left, left) if method_names else left
            right_name = method_names.get(right, right) if method_names else right
            title = f'{left_name} vs {right_name}'
        else:
            title = pair_tag
        ax.set_title(title, fontsize=_PLOT_FONTSIZE_TITLE)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc='upper right', fontsize=_PLOT_FONTSIZE_LABEL)
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()


def _draw_gap_source_panels(axes_row, gap_analysis: Dict, baseline_name: str = 'Baseline', compare_name: str = 'Compare', show_legends: bool = True):
    """Draw one row of gap-source panels (outcome/gain/loss)."""
    per_lang = gap_analysis.get('per_language', {})
    languages = _ordered_languages(list(per_lang.keys()))
    if not languages:
        for ax in axes_row:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
            ax.set_axis_off()
        return
    ax_outcome, ax_gain, ax_loss = axes_row

    # Panel 1: outcome transition rates (stacked)
    outcome_order = ['both_correct', 'compare_only_correct', 'baseline_only_correct', 'both_wrong']
    outcome_colors = {
        'both_correct': '#27AE60',
        'compare_only_correct': '#3498DB',
        'baseline_only_correct': '#E74C3C',
        'both_wrong': '#95A5A6',
    }
    outcome_labels = {
        'both_correct': 'Both Correct',
        'compare_only_correct': f'{compare_name} Only Correct',
        'baseline_only_correct': f'{baseline_name} Only Correct',
        'both_wrong': 'Both Wrong',
    }
    x = np.arange(len(languages))
    bottom = np.zeros(len(languages), dtype=float)
    for outcome in outcome_order:
        vals = np.array([per_lang[l].get('outcome_rates', {}).get(outcome, 0.0) * 100 for l in languages], dtype=float)
        bars = ax_outcome.bar(x, vals, bottom=bottom, color=outcome_colors[outcome], label=outcome_labels[outcome], alpha=0.85)
        for bar, v, b in zip(bars, vals, bottom):
            if v >= 8:
                ax_outcome.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f'{v:.0f}',
                    ha='center',
                    va='center',
                    fontsize=_PLOT_FONTSIZE_LABEL - 1,
                    color='white'
                )
        bottom += vals
    ax_outcome.set_xticks(x)
    ax_outcome.set_xticklabels(languages)
    ax_outcome.set_ylim(0, 100)
    ax_outcome.set_ylabel('Rate (%)')
    ax_outcome.set_title('Outcome Transition')
    if show_legends:
        ax_outcome.legend(fontsize=_PLOT_FONTSIZE_LABEL - 1)
    ax_outcome.grid(axis='y', alpha=0.3)

    # Panel 2: source attribution for gains (compare_only_correct)
    gain_order = ['language_barrier_reduced', 'stability_improved', 'uncertainty_reduced', 'confidence_improved', 'mixed_gain']
    gain_colors = {
        'language_barrier_reduced': '#8E44AD',
        'stability_improved': '#2980B9',
        'uncertainty_reduced': '#16A085',
        'confidence_improved': '#27AE60',
        'mixed_gain': '#BDC3C7',
    }
    bottom = np.zeros(len(languages), dtype=float)
    for source in gain_order:
        vals = np.array([per_lang[l].get('gain_sources_rates', {}).get(source, 0.0) * 100 for l in languages], dtype=float)
        ax_gain.bar(x, vals, bottom=bottom, color=gain_colors[source], label=source, alpha=0.85)
        bottom += vals
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(languages)
    ax_gain.set_ylim(0, 100)
    ax_gain.set_ylabel(f'Rate in {compare_name} Only-Correct Cases (%)')
    ax_gain.set_title(f'Improvement Source ({compare_name})')
    if show_legends:
        ax_gain.legend(fontsize=_PLOT_FONTSIZE_LABEL - 1)
    ax_gain.grid(axis='y', alpha=0.3)

    # Panel 3: source attribution for losses (baseline_only_correct)
    loss_order = ['overconfident_wrong', 'stability_degraded', 'uncertainty_increased', 'confidence_dropped', 'code_switch_interference', 'mixed_loss']
    loss_colors = {
        'overconfident_wrong': '#C0392B',
        'stability_degraded': '#D35400',
        'uncertainty_increased': '#F39C12',
        'confidence_dropped': '#E67E22',
        'code_switch_interference': '#7F8C8D',
        'mixed_loss': '#BDC3C7',
    }
    bottom = np.zeros(len(languages), dtype=float)
    for source in loss_order:
        vals = np.array([per_lang[l].get('loss_sources_rates', {}).get(source, 0.0) * 100 for l in languages], dtype=float)
        ax_loss.bar(x, vals, bottom=bottom, color=loss_colors[source], label=source, alpha=0.85)
        bottom += vals
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(languages)
    ax_loss.set_ylim(0, 100)
    ax_loss.set_ylabel(f'Rate in {baseline_name} Only-Correct Cases (%)')
    ax_loss.set_title(f'Degradation Source ({baseline_name})')
    if show_legends:
        ax_loss.legend(fontsize=_PLOT_FONTSIZE_LABEL - 1)
    ax_loss.grid(axis='y', alpha=0.3)


def plot_gap_source_analysis(gap_analysis: Dict, output_path: Path, baseline_name: str = 'Baseline', compare_name: str = 'Compare'):
    """Plot paired outcome transitions and source attribution by language."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f'Paired Accuracy Gap and Source Attribution by Language ({baseline_name} vs {compare_name})',
        fontsize=_PLOT_FONTSIZE_SUPTITLE,
        fontweight='bold'
    )
    _draw_gap_source_panels(axes, gap_analysis, baseline_name, compare_name, show_legends=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gap_source_analysis_combined(all_gap_source_by_pair: Dict[str, Dict], output_path: Path, method_names: Dict[str, str] = None):
    """Plot all pairwise gap-source results in one figure."""
    if not all_gap_source_by_pair:
        return
    pair_tags = sorted(all_gap_source_by_pair.keys())
    n_rows = len(pair_tags)
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, max(6.0, 5.2 * n_rows)), squeeze=False)
    fig.suptitle('Paired Accuracy Gap and Source Attribution by Language (All Method Pairs)', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')

    for row, pair_tag in enumerate(pair_tags):
        row_info = all_gap_source_by_pair[pair_tag]
        gap_analysis = row_info.get('analysis', {})
        baseline_name = row_info.get('baseline_name', 'Baseline')
        compare_name = row_info.get('compare_name', 'Compare')
        _draw_gap_source_panels(axes[row], gap_analysis, baseline_name, compare_name, show_legends=(row == 0))

        if '_vs_' in pair_tag:
            left, right = pair_tag.split('_vs_', 1)
            left_name = method_names.get(left, left) if method_names else left
            right_name = method_names.get(right, right) if method_names else right
            pair_title = f'{left_name} vs {right_name}'
        else:
            pair_title = pair_tag
        axes[row, 0].text(
            0.0,
            1.18,
            pair_title,
            transform=axes[row, 0].transAxes,
            ha='left',
            va='bottom',
            fontsize=_PLOT_FONTSIZE_TITLE,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_tri_perspective_verification(verification: Dict, output_path: Path):
    """Plot tri-perspective verification outcomes by language."""
    per_lang = verification.get('per_language', {})
    languages = [l for l in sorted(per_lang.keys()) if l not in DISPLAY_LANGUAGES_EXCLUDE]
    if not languages:
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('Tri-Perspective Verification (Eye-wise Accuracy + Majority Resolution)', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')

    x = np.arange(len(languages))
    w = 0.2

    def eye_metric(metric_key: str, fallback_key: str) -> List[float]:
        values = []
        for l in languages:
            lang_data = per_lang.get(l, {})
            values.append(_safe_float(lang_data.get(metric_key, lang_data.get(fallback_key, 0.0)), 0.0) * 100.0)
        return values

    def draw_accuracy_panel(ax, title: str, a1_vals: List[float], a2_vals: List[float], a3_vals: List[float], af_vals: List[float]):
        b1 = ax.bar(x - 1.5 * w, a1_vals, w, label='A1 Direct', color='#95A5A6', alpha=0.85)
        b2 = ax.bar(x - 0.5 * w, a2_vals, w, label='A2 En-CoT', color='#3498DB', alpha=0.85)
        b3 = ax.bar(x + 0.5 * w, a3_vals, w, label='A3 Pivot', color='#9B59B6', alpha=0.85)
        b4 = ax.bar(x + 1.5 * w, af_vals, w, label='Final', color='#27AE60', alpha=0.9)
        for bars in [b1, b2, b3, b4]:
            for bar in bars:
                val = bar.get_height()
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + _BAR_LABEL_GAP_SMALL,
                        f'{val:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=_PLOT_FONTSIZE_LABEL - 1
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(fontsize=_PLOT_FONTSIZE_LABEL - 1)
        ax.grid(axis='y', alpha=0.3)

    # Panel 1: right-eye (OD) accuracy comparison.
    draw_accuracy_panel(
        axes[0],
        'Right Eye Accuracy (OD)',
        eye_metric('a1_accuracy_right_eye_on_available', 'a1_accuracy_on_available'),
        eye_metric('a2_accuracy_right_eye_on_shared', 'a2_accuracy_on_shared'),
        eye_metric('a3_accuracy_right_eye_on_shared', 'a3_accuracy_on_shared'),
        eye_metric('final_accuracy_right_eye', 'final_accuracy'),
    )

    # Panel 2: left-eye (OS) accuracy comparison.
    draw_accuracy_panel(
        axes[1],
        'Left Eye Accuracy (OS)',
        eye_metric('a1_accuracy_left_eye_on_available', 'a1_accuracy_on_available'),
        eye_metric('a2_accuracy_left_eye_on_shared', 'a2_accuracy_on_shared'),
        eye_metric('a3_accuracy_left_eye_on_shared', 'a3_accuracy_on_shared'),
        eye_metric('final_accuracy_left_eye', 'final_accuracy'),
    )

    # Panel 3: resolution composition.
    resolution_order = [
        'a2_a3_consistent_choose_a2',
        'a2_a3_conflict_a1_supports_a2',
        'a2_a3_conflict_a1_supports_a3',
        'a2_a3_conflict_tiebreak_confidence',
        'a2_a3_conflict_no_a1_tiebreak_confidence',
    ]
    colors = {
        'a2_a3_consistent_choose_a2': '#2ECC71',
        'a2_a3_conflict_a1_supports_a2': '#3498DB',
        'a2_a3_conflict_a1_supports_a3': '#9B59B6',
        'a2_a3_conflict_tiebreak_confidence': '#E67E22',
        'a2_a3_conflict_no_a1_tiebreak_confidence': '#E74C3C',
    }
    labels = {
        'a2_a3_consistent_choose_a2': 'A2≈A3 choose A2',
        'a2_a3_conflict_a1_supports_a2': 'Conflict: A1->A2',
        'a2_a3_conflict_a1_supports_a3': 'Conflict: A1->A3',
        'a2_a3_conflict_tiebreak_confidence': 'Conflict: confidence tie-break',
        'a2_a3_conflict_no_a1_tiebreak_confidence': 'Conflict: no A1, confidence tie-break',
    }
    bottom = np.zeros(len(languages), dtype=float)
    for key in resolution_order:
        vals = np.array([per_lang[l].get('resolution_rates', {}).get(key, 0.0) * 100 for l in languages], dtype=float)
        bars = axes[2].bar(x, vals, bottom=bottom, color=colors[key], alpha=0.88, label=labels[key])
        for bar, v, b in zip(bars, vals, bottom):
            if v >= 9:
                axes[2].text(
                    bar.get_x() + bar.get_width()/2,
                    b + v/2,
                    f'{v:.0f}',
                    ha='center',
                    va='center',
                    fontsize=_PLOT_FONTSIZE_LABEL - 1,
                    color='white'
                )
        bottom += vals
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(languages)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_title('Resolution Composition')
    axes[2].legend(fontsize=_PLOT_FONTSIZE_LABEL - 2)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_token_level_analysis(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], english_ref: Dict = None):
    """Plot token-level confidence decay and medical-term activation."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = [l for l in sorted(data0.keys()) if l not in DISPLAY_LANGUAGES_EXCLUDE]
    
    n_methods = len(method_keys)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 5 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Token-level Reasoning: Confidence Decay & Medical Term Activation', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    
    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        ax1, ax2 = axes[row]
        # Confidence decay aggregated across languages.
        decay_curves = []
        for l in languages:
            tla = data.get(l, {}).get('token_level_analysis', {})
            curve = tla.get('confidence_decay_curve', [])
            if curve:
                decay_curves.append(curve)
        if decay_curves:
            mean_curve = np.mean(decay_curves, axis=0)
            bins = np.linspace(0, 1, len(mean_curve))
            ax1.plot(bins, mean_curve, 'o-', color='#3498DB', linewidth=2)
            for bx, by in zip(bins, mean_curve):
                ax1.text(bx, by + 0.02, f'{by:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL - 1)
        ax1.set_xlabel('Position (0=start, 1=end)')
        ax1.set_ylabel('Avg Token Probability')
        ax1.set_title(f'Confidence Decay ({name})')
        ax1.set_ylim(0, 1.05)
        ax1.grid(alpha=0.3)
        # Medical-term activation from the first language with available term stats.
        tla = None
        for l in languages:
            tla = data.get(l, {}).get('token_level_analysis', {})
            mt = tla.get('medical_term_avg_prob', {})
            if mt:
                terms = list(mt.keys())[:10]
                vals = [mt[t] for t in terms]
                bars_mt = ax2.barh(terms, vals, color='#27AE60', alpha=0.7)
                for bar, val in zip(bars_mt, vals):
                    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', ha='left', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
                break
        if tla and tla.get('medical_term_avg_prob'):
            ax2.set_xlabel('Avg Probability')
            ax2.set_title(f'Medical Term Activation ({name})')
        else:
            ax2.text(0.5, 0.5, 'No medical terms detected', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cross_lingual_analysis(all_data_by_method: Dict[str, Dict], output_path: Path, english_ref: Dict = None):
    """Plot cross-lingual semantic alignment and code-switching."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Cross-lingual: Semantic Alignment & Code-switching', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    # Semantic alignment by concept and language.
    alignment_data = {}
    for l in display_languages:
        cla = data0.get(l, {}).get('cross_lingual_analysis', {})
        sal = cla.get('semantic_alignment', {})
        for concept, lang_probs in sal.items():
            if concept not in alignment_data:
                alignment_data[concept] = {}
            alignment_data[concept][l] = lang_probs.get(l, 0)
    if alignment_data:
        concepts = list(alignment_data.keys())
        x = np.arange(len(display_languages))
        w = 0.8 / len(concepts)
        colors = ['#9B59B6', '#27AE60', '#3498DB', '#F39C12'][:len(concepts)]
        for i, c in enumerate(concepts):
            vals = [alignment_data[c].get(l, 0) for l in display_languages]
            bars_al = axes[0].bar(x + (i - len(concepts)/2 + 0.5) * w, vals, w, label=c, color=colors[i % len(colors)], alpha=0.8)
            for bar, val in zip(bars_al, vals):
                if val > 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL - 1)
        eng_align = None
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            cla_en = english_ref.get('cross_lingual_analysis', {}).get('semantic_alignment', {})
            eng_align = np.mean([cla_en.get(c, {}).get('English', 0) for c in concepts]) if concepts else 0
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(display_languages)
        axes[0].set_ylabel('Avg Probability')
        axes[0].set_title('Semantic Alignment (concept logprob by language)')
        axes[0].set_ylim(0, 1.05)
        if eng_align is not None:
            _add_reference_line_with_value(axes[0], eng_align, '{:.2f}')
        axes[0].legend()
    # Code-switching for non-English targets.
    cs_ratios = {l: data0.get(l, {}).get('cross_lingual_analysis', {}).get('code_switch_ratio', {}).get(l, 0) for l in display_languages}
    if cs_ratios:
        x = np.arange(len(display_languages))
        cs_vals = [cs_ratios.get(l, 0) for l in display_languages]
        bars_cs = axes[1].bar(x, cs_vals, color=['#9B59B6', '#27AE60', '#3498DB'][:len(display_languages)], alpha=0.8)
        for bar, val in zip(bars_cs, cs_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=_PLOT_FONTSIZE_LABEL)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(display_languages)
        axes[1].set_ylabel('English Token Ratio')
        axes[1].set_title('Code-switching (non-English targets)')
        axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================
# 14. Visualization: cognitive stability
# ==========================================

def plot_cognitive_stability_distribution(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], english_ref: Dict = None):
    """Plot mutation-count distributions by language and by correctness."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    
    n_methods = len(method_keys)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 5 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Cognitive Stability: Mutation Count Distribution (Correct vs Incorrect)', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    
    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        # Left: violin plot grouped by language.
        ax1 = axes[row, 0]
        cs_data = [data.get(lang, {}).get('cognitive_stability', {}) for lang in display_languages]
        per_sample = [cs.get('per_sample_records', []) for cs in cs_data]
        counts_by_lang = [[r['mutation_count'] for r in recs] for recs in per_sample]
        parts = ax1.violinplot(counts_by_lang, positions=range(len(display_languages)), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498DB')
            pc.set_alpha(0.7)
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            cs_en = english_ref.get('cognitive_stability', {})
            recs_en = cs_en.get('per_sample_records', [])
            eng_mean = np.mean([r['mutation_count'] for r in recs_en]) if recs_en else 0
            _add_reference_line_with_value(ax1, eng_mean, '{:.2f}')
        ax1.set_xticks(range(len(display_languages)))
        ax1.set_xticklabels(display_languages, fontsize=_PLOT_FONTSIZE_LABEL)
        ax1.set_ylabel('Mutation Count', fontsize=_PLOT_FONTSIZE_LABEL)
        ax1.set_title(f'By Language ({name})', fontsize=_PLOT_FONTSIZE_TITLE)
        ax1.grid(axis='y', alpha=0.3)
        if english_ref and ENGLISH_AS_REFERENCE_LINE:
            ax1.legend(fontsize=_PLOT_FONTSIZE_LABEL)
        # Right: violin plot for correct vs incorrect samples.
        ax2 = axes[row, 1]
        correct_counts = []
        incorrect_counts = []
        for lang in display_languages:
            recs = data.get(lang, {}).get('cognitive_stability', {}).get('per_sample_records', [])
            correct_counts.extend([r['mutation_count'] for r in recs if r['is_correct']])
            incorrect_counts.extend([r['mutation_count'] for r in recs if not r['is_correct']])
        if correct_counts or incorrect_counts:
            parts2 = ax2.violinplot([correct_counts or [0], incorrect_counts or [0]], positions=[0, 1], showmeans=True, showmedians=True)
            for i, pc in enumerate(parts2['bodies']):
                pc.set_facecolor('#27AE60' if i == 0 else '#E74C3C')
                pc.set_alpha(0.7)
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['Correct', 'Incorrect'], fontsize=_PLOT_FONTSIZE_LABEL)
        ax2.set_ylabel('Mutation Count', fontsize=_PLOT_FONTSIZE_LABEL)
        ax2.set_title(f'Correct vs Incorrect ({name})', fontsize=_PLOT_FONTSIZE_TITLE)
        ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stability_heatmap(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str] = None):
    """Cross-lingual stability heatmap by method (including English)."""
    method_keys = sorted(all_data_by_method.keys())
    if not method_keys:
        return

    preferred_languages = ['Chinese', 'English', 'Malay', 'Thai']
    language_set = set()
    for key in method_keys:
        language_set.update(all_data_by_method.get(key, {}).keys())
    languages = [l for l in preferred_languages if l in language_set] + sorted([l for l in language_set if l not in preferred_languages])
    if not languages:
        return

    matrices = []
    all_vals = []
    for key in method_keys:
        data = all_data_by_method.get(key, {})
        matrix = []
        for lang in languages:
            cs = data.get(lang, {}).get('cognitive_stability', {})
            row = cs.get('binned_mean_logprob', [])
            if not row:
                matrix.append([np.nan] * 10)
                continue
            row_vals = [_safe_float(v, np.nan) for v in row[:10]]
            if len(row_vals) < 10:
                row_vals += [np.nan] * (10 - len(row_vals))
            matrix.append(row_vals)
        matrix_np = np.array(matrix, dtype=float)
        matrices.append((key, matrix_np))
        finite_vals = matrix_np[np.isfinite(matrix_np)]
        if finite_vals.size > 0:
            all_vals.extend(finite_vals.tolist())

    if not all_vals:
        return

    vmin = min(all_vals)
    vmax = max(all_vals)
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    n_rows = len(matrices)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, max(4.5, 3.0 * n_rows)), squeeze=False)
    fig.suptitle('Cross-lingual Reasoning Stability: Mean Logprob by Position (Per Method)', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad(color='#E6E6E6')
    last_im = None

    for row_idx, (method_key, matrix) in enumerate(matrices):
        ax = axes[row_idx, 0]
        last_im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(languages)))
        ax.set_yticklabels(languages, fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(10)], fontsize=_PLOT_FONTSIZE_LABEL - 1)
        if row_idx == n_rows - 1:
            ax.set_xlabel('Normalized Position (0=start, 1=end)', fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_ylabel('Language', fontsize=_PLOT_FONTSIZE_LABEL)
        title = method_names.get(method_key, method_key) if method_names else method_key
        ax.set_title(f'{title}', fontsize=_PLOT_FONTSIZE_TITLE)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=[a for a in axes[:, 0]], fraction=0.025, pad=0.02)
        cbar.set_label('Mean Logprob')

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_single_case_trajectory(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], language: str = None):
    """Single-case trajectories: high-mutation incorrect vs low-mutation correct sample."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    lang = language or (display_languages[0] if display_languages else languages[0])
    data = data0.get(lang, {})
    high_traj = data.get('stability_trajectory_high_mutation')
    low_traj = data.get('stability_trajectory_low_mutation')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Token-level Reasoning Trajectory: High vs Low Mutation ({lang})', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    
    for ax, traj, label in zip(axes, [high_traj, low_traj], ['High Mutation (Incorrect)', 'Low Mutation (Correct)']):
        if not traj:
            ax.text(0.5, 0.5, f'No {label} sample available', ha='center', va='center', fontsize=_PLOT_FONTSIZE_LABEL)
            ax.set_title(label, fontsize=_PLOT_FONTSIZE_TITLE)
            continue
        logprobs = traj.get('logprobs', [])
        n_tokens = traj.get('n_tokens', len(logprobs))
        mutations_detail = traj.get('mutations_detail', [])
        mut_x = [m[0] for m in mutations_detail if m[0] < len(logprobs)]
        mut_y = [logprobs[m[0]] for m in mutations_detail if m[0] < len(logprobs)]
        x = np.arange(len(logprobs))
        ax.plot(x, logprobs, color='#3498DB', linewidth=1, alpha=0.8)
        if mut_x:
            ax.scatter(mut_x, mut_y, color='#E74C3C', s=40, zorder=3, label='Mutation')
        # Phase separators: problem 0-20%, retrieval 20-60%, answer 60-100%.
        for frac in [0.2, 0.6]:
            ax.axvline(x=frac * len(logprobs), color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Token Position', fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_ylabel('Log Probability', fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_title(f'{label} (n={n_tokens}, mutations={traj.get("mutation_count", 0)})', fontsize=_PLOT_FONTSIZE_TITLE)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=_PLOT_FONTSIZE_LABEL)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stability_metrics_summary(all_data_by_method: Dict[str, Dict], output_path: Path, method_names: Dict[str, str], english_ref: Dict = None):
    """Summary of stability metrics by language with optional English reference lines."""
    method_keys = sorted(all_data_by_method.keys())
    data0 = all_data_by_method[method_keys[0]]
    languages = sorted(data0.keys())
    display_languages = [l for l in languages if l not in DISPLAY_LANGUAGES_EXCLUDE]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cognitive Stability Metrics Summary', fontsize=_PLOT_FONTSIZE_SUPTITLE, fontweight='bold')
    x = np.arange(len(display_languages))
    w = 0.8 / max(len(method_keys), 1)
    colors = ['#9B59B6', '#27AE60', '#3498DB', '#F39C12', '#E74C3C'][:max(len(method_keys), 1)]
    
    for row, key in enumerate(method_keys):
        data = all_data_by_method[key]
        name = method_names.get(key, key)
        cs_list = [data.get(lang, {}).get('cognitive_stability', {}) for lang in display_languages]
        mutation_rate = [cs.get('mutation_rate_mean', 0) for cs in cs_list]
        stability_score = [cs.get('stability_score_mean', 0) for cs in cs_list]
        mutation_at_medical = [cs.get('mutation_at_medical_ratio_mean', 0) for cs in cs_list]
        mutation_at_answer = [cs.get('mutation_at_answer_ratio_mean', 0) for cs in cs_list]
        off = (row - len(method_keys)/2 + 0.5) * w
        color = colors[row % len(colors)]
        axes[0, 0].bar(x + off, mutation_rate, w, label=name, color=color, alpha=0.8)
        axes[0, 1].bar(x + off, stability_score, w, label=name, color=color, alpha=0.8)
        axes[1, 0].bar(x + off, mutation_at_medical, w, label=name, color=color, alpha=0.8)
        axes[1, 1].bar(x + off, mutation_at_answer, w, label=name, color=color, alpha=0.8)
    
    if english_ref and ENGLISH_AS_REFERENCE_LINE:
        cs_en = english_ref.get('cognitive_stability', {})
        line_specs = [
            (axes[0, 0], 'mutation_rate_mean', '{:.4f}'),
            (axes[0, 1], 'stability_score_mean', '{:.3f}'),
            (axes[1, 0], 'mutation_at_medical_ratio_mean', '{:.3f}'),
            (axes[1, 1], 'mutation_at_answer_ratio_mean', '{:.3f}')
        ]
        for ax, key, fmt in line_specs:
            _add_reference_line_with_value(ax, cs_en.get(key, 0), fmt)
    
    for ax, ylabel, title in [(axes[0,0], 'Mutation Rate', 'Mutation Rate (per token)'),
                              (axes[0,1], 'Stability Score', 'Stability Score (1=stable)'),
                              (axes[1,0], 'Ratio', 'Mutation at Medical Terms'),
                              (axes[1,1], 'Ratio', 'Mutation at Answer Phase')]:
        ax.set_xticks(x)
        ax.set_xticklabels(display_languages, fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=_PLOT_FONTSIZE_LABEL)
        ax.set_title(title, fontsize=_PLOT_FONTSIZE_TITLE)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=_PLOT_FONTSIZE_LABEL)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================
# 15. Main
# ==========================================

DATASET_METHODS = {
    '2type': [
        ('direct', 'response_2type', 'Reasoning in Target Language'),
        ('reasoning', 'response_2type_reasoning_english', 'Cross-Lingual En-CoT'),
        ('translate_pivot', 'response_2type_translate_back_in_english', 'Translation Pivot')
    ],
    '3type': [
        ('direct', 'response_3type', 'Reasoning in Target Language'),
        ('reasoning', 'response_3type_reasoning_english', 'Cross-Lingual En-CoT'),
        ('translate_pivot', 'response_3type_back_in_english', 'Translation Pivot')
    ],
}


def _export_method_summary_json(method_stats: Dict[str, Dict], out_path: Path):
    """Write compact summary JSON for one method."""
    out_json = {}
    for lang, stats in method_stats.items():
        cs = stats.get('cognitive_stability') or {}
        cs_export = {k: v for k, v in cs.items() if k != 'per_sample_records'}
        if cs.get('per_sample_records'):
            cs_export['per_sample_count'] = len(cs['per_sample_records'])
            cs_export['per_sample_mutation_count_sample'] = [r['mutation_count'] for r in cs['per_sample_records'][:20]]
        out_json[lang] = {
            'accuracy': stats['accuracy'],
            'accuracy_both_eyes': stats.get('accuracy_both_eyes', stats.get('accuracy', 0.0)),
            'accuracy_right_eye': stats.get('accuracy_right_eye', 0.0),
            'accuracy_left_eye': stats.get('accuracy_left_eye', 0.0),
            'accuracy_at_least_one_eye': stats.get('accuracy_at_least_one_eye', 0.0),
            'diagnosis_stratified': stats['diagnosis_stratified'],
            'confidence_analysis': stats.get('confidence_analysis'),
            'key_step_confidence': stats.get('key_step_confidence'),
            'conclusion_confidence': stats.get('conclusion_confidence'),
            'cot_uncertainty_analysis': stats.get('cot_uncertainty_analysis'),
            'token_level_analysis': stats.get('token_level_analysis'),
            'cross_lingual_analysis': stats.get('cross_lingual_analysis'),
            'cognitive_stability': cs_export,
            'top_confusions': stats['auto_detected_confusions'][:10]
        }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_json_serializable(out_json), f, indent=2, ensure_ascii=False)


def run_analysis_for_dataset(
    dataset_key: str,
    methods_config: List[Tuple[str, str, str]],
    base_dir: Path,
    out_dir: Path,
    model: str,
    languages: List[str],
):
    """Run full analysis for one dataset family (2type or 3type)."""
    model_safe = model.replace("/", "_")
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"{'='*60}")

    all_method_stats = {}
    method_names = {}
    method_folders = {key: folder for key, folder, _ in methods_config}
    raw_results_by_method: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    direct_folder = method_folders.get('direct')
    english_shared_results: Optional[List[Dict]] = None
    english_shared_stats: Optional[Dict] = None

    for key, folder, name in methods_config:
        print(f"\n>> Processing Method: {name}")
        method_stats = {}

        for lang in languages:
            if lang == 'English' and key != 'direct':
                if english_shared_results is None or english_shared_stats is None:
                    if not direct_folder:
                        print("   [!] Missing direct folder: cannot share English data")
                        continue
                    shared_fpath = base_dir / direct_folder / model / lang / 'round1.json'
                    if not shared_fpath.exists():
                        print(f"   [!] Missing shared English data: {shared_fpath}")
                        continue
                    print("   Loading English (shared direct data)...")
                    with open(shared_fpath, 'r', encoding='utf-8') as f:
                        shared_raw_data = json.load(f)
                    english_shared_results = shared_raw_data.get('results', [])
                    english_shared_stats = analyze_dataset_v2(english_shared_results, lang)
                raw_results_by_method[key][lang] = english_shared_results
                method_stats[lang] = copy.deepcopy(english_shared_stats)
                print("   [Reuse] English: shared direct data")
                print(
                    "     Acc (Both/OD/OS): "
                    f"{method_stats[lang]['accuracy']:.1%} / "
                    f"{method_stats[lang].get('accuracy_right_eye', 0):.1%} / "
                    f"{method_stats[lang].get('accuracy_left_eye', 0):.1%}"
                )
                continue

            fpath = base_dir / folder / model / lang / 'round1.json'
            if not fpath.exists():
                print(f"   [!] Missing: {lang}")
                continue

            print(f"   Loading {lang}...")
            with open(fpath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            results = raw_data.get('results', [])
            raw_results_by_method[key][lang] = results
            stats = analyze_dataset_v2(results, lang)
            method_stats[lang] = stats
            if lang == 'English' and key == 'direct':
                english_shared_results = results
                english_shared_stats = copy.deepcopy(stats)
            print(
                "     Acc (Both/OD/OS): "
                f"{stats['accuracy']:.1%} / "
                f"{stats.get('accuracy_right_eye', 0):.1%} / "
                f"{stats.get('accuracy_left_eye', 0):.1%}"
            )

        if method_stats:
            all_method_stats[key] = method_stats
            method_names[key] = name
            _export_method_summary_json(
                method_stats,
                out_dir / f"{model_safe}_{key}_improved_summary.json"
            )

    # English is reference-only for line overlays.
    english_ref = all_method_stats.get('direct', {}).get('English')

    # Pairwise attribution across available methods.
    available_method_keys = [key for key, _, _ in methods_config if key in all_method_stats]
    method_pairs: List[Tuple[str, str]] = []
    if 'direct' in available_method_keys:
        method_pairs.extend([('direct', k) for k in available_method_keys if k != 'direct'])
    for left, right in combinations(available_method_keys, 2):
        if (left, right) in method_pairs or (right, left) in method_pairs:
            continue
        method_pairs.append((left, right))

    all_error_attribution_by_pair: Dict[str, Dict[str, Dict]] = {}
    all_gap_source_by_pair: Dict[str, Dict] = {}
    english_direct_results = raw_results_by_method.get('direct', {}).get('English')

    for baseline_key, compare_key in method_pairs:
        baseline_folder = method_folders.get(baseline_key)
        compare_folder = method_folders.get(compare_key)
        if not baseline_folder or not compare_folder:
            continue

        pair_tag = f"{baseline_key}_vs_{compare_key}"
        print(f"\n>> Pairwise attribution: {method_names.get(baseline_key, baseline_key)} vs {method_names.get(compare_key, compare_key)}")

        error_attribution_by_lang = {}
        baseline_results_by_lang = {}
        compare_results_by_lang = {}

        for lang in languages:
            if lang == 'English':
                if english_direct_results is None:
                    continue
                # For all pairs, English uses the direct-method round1 results.
                baseline_results = english_direct_results
                compare_results = english_direct_results
            else:
                baseline_results = raw_results_by_method.get(baseline_key, {}).get(lang)
                compare_results = raw_results_by_method.get(compare_key, {}).get(lang)
            if baseline_results is None or compare_results is None:
                continue
            if lang != 'English':
                baseline_results_by_lang[lang] = baseline_results
                compare_results_by_lang[lang] = compare_results
            error_attribution_by_lang[lang] = analyze_error_attribution(
                baseline_results,
                compare_results,
                lang
            )

        if error_attribution_by_lang:
            err_json_path = out_dir / f"{model_safe}_error_attribution_{pair_tag}.json"
            with open(err_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_attribution_by_lang, f, indent=2, ensure_ascii=False)
            all_error_attribution_by_pair[pair_tag] = error_attribution_by_lang
            print(f"   Error attribution saved: {pair_tag}")

            # Keep legacy filenames for direct vs reasoning to preserve backward compatibility.
            if baseline_key == 'direct' and compare_key == 'reasoning':
                with open(out_dir / f"{model_safe}_error_attribution.json", 'w', encoding='utf-8') as f:
                    json.dump(error_attribution_by_lang, f, indent=2, ensure_ascii=False)

        gap_source_analysis = analyze_gap_sources_across_languages(
            direct_results_by_lang=baseline_results_by_lang,
            reasoning_results_by_lang=compare_results_by_lang,
            languages=languages,
            baseline_key=baseline_key,
            compare_key=compare_key,
        )
        if gap_source_analysis.get('per_language'):
            gap_json_path = out_dir / f"{model_safe}_gap_source_analysis_{pair_tag}.json"
            with open(gap_json_path, 'w', encoding='utf-8') as f:
                json.dump(_json_serializable(gap_source_analysis), f, indent=2, ensure_ascii=False)
            all_gap_source_by_pair[pair_tag] = {
                'analysis': gap_source_analysis,
                'baseline_name': method_names.get(baseline_key, baseline_key),
                'compare_name': method_names.get(compare_key, compare_key),
            }
            print(f"   Gap-source analysis saved: {pair_tag}")

            # Keep legacy filenames for direct vs reasoning to preserve backward compatibility.
            if baseline_key == 'direct' and compare_key == 'reasoning':
                with open(out_dir / f"{model_safe}_gap_source_analysis.json", 'w', encoding='utf-8') as f:
                    json.dump(_json_serializable(gap_source_analysis), f, indent=2, ensure_ascii=False)

    # All pairwise error-attribution panels in one figure.
    if all_error_attribution_by_pair:
        combined_err_png = out_dir / f"{model_safe}_error_attribution_all_pairs.png"
        plot_error_attribution_combined(all_error_attribution_by_pair, combined_err_png, method_names)
        # Keep legacy figure name as alias to the combined plot.
        plot_error_attribution_combined(all_error_attribution_by_pair, out_dir / f"{model_safe}_error_attribution.png", method_names)

    # All pairwise gap-source panels in one figure.
    if all_gap_source_by_pair:
        combined_gap_png = out_dir / f"{model_safe}_gap_source_analysis_all_pairs.png"
        plot_gap_source_analysis_combined(all_gap_source_by_pair, combined_gap_png, method_names)
        # Keep legacy figure name as alias to the combined plot.
        plot_gap_source_analysis_combined(all_gap_source_by_pair, out_dir / f"{model_safe}_gap_source_analysis.png", method_names)

    # Tri-perspective verification (A1/A2/A3 with conflict resolution).
    if 'reasoning' in raw_results_by_method and 'translate_pivot' in raw_results_by_method:
        tri_verification = analyze_tri_perspective_verification(
            direct_results_by_lang=raw_results_by_method.get('direct', {}),
            reasoning_results_by_lang=raw_results_by_method.get('reasoning', {}),
            pivot_results_by_lang=raw_results_by_method.get('translate_pivot', {}),
            languages=languages,
        )
        if tri_verification.get('per_language'):
            tri_json_path = out_dir / f"{model_safe}_tri_perspective_verification.json"
            tri_png_path = out_dir / f"{model_safe}_tri_perspective_verification.png"
            with open(tri_json_path, 'w', encoding='utf-8') as f:
                json.dump(_json_serializable(tri_verification), f, indent=2, ensure_ascii=False)
            plot_tri_perspective_verification(tri_verification, tri_png_path)
            print("   Tri-perspective verification saved.")

    # Combined figures for this dataset.
    if len(all_method_stats) >= 1:
        print("\n>> Generating combined figures...")
        plot_accuracy_comparison(all_method_stats, out_dir / f"{model_safe}_accuracy_comparison.png", method_names, english_ref)
        plot_accuracy_comparison(
            all_method_stats,
            out_dir / f"{model_safe}_accuracy_comparison_right_eye.png",
            method_names,
            english_ref,
            accuracy_key='accuracy_right_eye',
            title='Right Eye Accuracy (OD) Across Methods'
        )
        plot_accuracy_comparison(
            all_method_stats,
            out_dir / f"{model_safe}_accuracy_comparison_left_eye.png",
            method_names,
            english_ref,
            accuracy_key='accuracy_left_eye',
            title='Left Eye Accuracy (OS) Across Methods'
        )
        plot_diagnosis_stratified_combined(all_method_stats, out_dir / f"{model_safe}_disease_stratified.png", method_names, english_ref)
        plot_confidence_analysis(all_method_stats, out_dir / f"{model_safe}_confidence_analysis.png", method_names, english_ref)
        plot_cot_uncertainty_analysis(all_method_stats, out_dir / f"{model_safe}_cot_uncertainty.png", method_names, english_ref)
        plot_token_level_analysis(all_method_stats, out_dir / f"{model_safe}_token_level.png", method_names, english_ref)
        plot_cross_lingual_analysis(all_method_stats, out_dir / f"{model_safe}_cross_lingual.png", english_ref)
        plot_cognitive_stability_distribution(all_method_stats, out_dir / f"{model_safe}_stability_distribution.png", method_names, english_ref)
        plot_stability_heatmap(all_method_stats, out_dir / f"{model_safe}_stability_heatmap.png", method_names)
        plot_single_case_trajectory(all_method_stats, out_dir / f"{model_safe}_stability_single_case.png", method_names)
        plot_stability_metrics_summary(all_method_stats, out_dir / f"{model_safe}_stability_metrics.png", method_names, english_ref)
        if len(all_method_stats) >= 2:
            plot_auto_confusions_combined(all_method_stats, out_dir / f"{model_safe}_auto_confusions.png", method_names)

    print(f"\nDataset {dataset_key} completed. Results saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/data3/yuqian/OphthalmologyEHRglaucoma/result')
    parser.add_argument('--output_dir', type=str, default='/mnt/data3/yuqian/OphthalmologyEHRglaucoma/result/improved_analysis')
    parser.add_argument('--model', type=str, default='openai_gpt-4o')
    parser.add_argument('--languages', nargs='+', default=['Chinese', 'English', 'Malay', 'Thai'])
    parser.add_argument('--dataset', choices=['2type', '3type', 'both'], default='both',
                        help='Which dataset family to analyze. "both" runs two independent analyses.')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'both':
        dataset_keys = ['2type', '3type']
    else:
        dataset_keys = [args.dataset]

    print(f"Starting Improved Analysis for model: {args.model}")
    print(f"Selected dataset scope: {', '.join(dataset_keys)}")

    for dataset_key in dataset_keys:
        methods_config = DATASET_METHODS[dataset_key]
        dataset_out_dir = output_root / dataset_key
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        run_analysis_for_dataset(
            dataset_key=dataset_key,
            methods_config=methods_config,
            base_dir=base_dir,
            out_dir=dataset_out_dir,
            model=args.model,
            languages=args.languages,
        )

    print(f"\nAll tasks completed. Results saved under: {output_root}")

if __name__ == '__main__':
    main()
