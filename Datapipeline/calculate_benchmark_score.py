#!/usr/bin/env python3
"""
Calculate WER/CER for ASR transcriptions
Core functions used by test scripts
"""

import re
from typing import List, Tuple, Dict

# Buckwalter transliteration mapping
BUCKWALTER_TO_ARABIC = {
    "'": "ء", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ",
    "A": "ا", "b": "ب", "p": "ة", "t": "ت", "v": "ث",
    "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ",
    "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص",
    "D": "ض", "T": "ط", "Z": "ظ", "E": "ع", "g": "غ",
    "_": "ـ", "f": "ف", "q": "ق", "k": "ك", "l": "ل",
    "m": "م", "n": "ن", "h": "ه", "w": "و", "Y": "ى",
    "y": "ي", "F": "ً", "N": "ٌ", "K": "ٍ", "a": "َ",
    "u": "ُ", "i": "ِ", "~": "ّ", "o": "ْ", "`": "ٰ",
    "{": "ٱ", "^": "ٓ", "#": "٪", "@": "ۚ", "[": "ۖ",
    ";": "ۗ", ",": "ۘ", ".": "۠", "!": "ۡ", "-": "-",
    " ": " "
}


def buckwalter_to_arabic(text: str) -> str:
    """Convert Buckwalter transliteration to Arabic script"""
    result = []
    for char in text:
        result.append(BUCKWALTER_TO_ARABIC.get(char, char))
    return ''.join(result)


def normalize_text(text: str, language: str = 'ar') -> str:
    """Normalize text for comparison
    
    Args:
        text: Input text to normalize
        language: Language code ('ar' for Arabic, other for generic)
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # For Arabic text, remove diacritics (tashkeel)
    if language == 'ar':
        # Remove Arabic diacritics (َ ُ ِ ّ ْ ً ٌ ٍ etc.)
        arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
        text = arabic_diacritics.sub('', text)
        
        # Normalize Alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # Normalize other common variants
        text = re.sub(r'ى', 'ي', text)  # Alef Maqsura to Ya
        text = re.sub(r'ة', 'ه', text)  # Ta Marbuta to Ha
    
    # Convert to lowercase (for non-Arabic characters)
    text = text.lower()
    
    # Remove punctuation and special characters (but keep Arabic letters)
    if language == 'ar':
        # Keep Arabic letters and spaces only
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    else:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Calculate edit distance and error counts
    
    Args:
        ref: Reference tokens (words or characters)
        hyp: Hypothesis tokens (words or characters)
    
    Returns:
        Tuple of (distance, substitutions, deletions, insertions)
    """
    m, n = len(ref), len(hyp)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Track operation types
    ops = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = [('del', i)]
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = [('ins', j)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = ops[i-1][j-1]
            else:
                # Substitution
                sub_cost = dp[i-1][j-1] + 1
                # Deletion
                del_cost = dp[i-1][j] + 1
                # Insertion
                ins_cost = dp[i][j-1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                
                if min_cost == sub_cost:
                    ops[i][j] = ops[i-1][j-1] + [('sub', 1)]
                elif min_cost == del_cost:
                    ops[i][j] = ops[i-1][j] + [('del', 1)]
                else:
                    ops[i][j] = ops[i][j-1] + [('ins', 1)]
    
    # Count error types
    operations = ops[m][n]
    subs = sum(1 for op, _ in operations if op == 'sub')
    dels = sum(1 for op, _ in operations if op == 'del')
    ins = sum(1 for op, _ in operations if op == 'ins')
    
    return dp[m][n], subs, dels, ins


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """Calculate Word Error Rate and detailed stats
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        Tuple of (WER as float, detailed statistics dict)
    """
    if not reference:
        return 1.0, {'distance': 0, 'ref_words': 0, 'hyp_words': 0}
    
    if not hypothesis:
        ref_words = reference.split()
        return 1.0, {'distance': len(ref_words), 'ref_words': len(ref_words), 'hyp_words': 0}
    
    # Tokenize
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate edit distance
    distance, subs, dels, ins = calculate_edit_distance(ref_words, hyp_words)
    
    # WER = edit_distance / reference_length
    wer = distance / len(ref_words) if ref_words else 1.0
    
    stats = {
        'distance': distance,
        'ref_words': len(ref_words),
        'hyp_words': len(hyp_words),
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'wer': wer
    }
    
    return wer, stats


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """Calculate Character Error Rate and detailed stats
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        Tuple of (CER as float, detailed statistics dict)
    """
    if not reference:
        return 1.0, {'distance': 0, 'ref_chars': 0, 'hyp_chars': 0}
    
    if not hypothesis:
        return 1.0, {'distance': len(reference), 'ref_chars': len(reference), 'hyp_chars': 0}
    
    # Convert to character lists (excluding spaces)
    ref_chars = list(reference.replace(' ', ''))
    hyp_chars = list(hypothesis.replace(' ', ''))
    
    # Calculate edit distance
    distance, subs, dels, ins = calculate_edit_distance(ref_chars, hyp_chars)
    
    # CER = edit_distance / reference_length
    cer = distance / len(ref_chars) if ref_chars else 1.0
    
    stats = {
        'distance': distance,
        'ref_chars': len(ref_chars),
        'hyp_chars': len(hyp_chars),
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'cer': cer
    }
    
    return cer, stats
