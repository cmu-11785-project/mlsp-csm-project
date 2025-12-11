#!/usr/bin/env python3
"""
English ASR evaluation (100 files, LibriSpeech test-clean)
Fully adapted from Arabic advanced evaluation script,
but using English ground truth and English normalization.
"""

import sys
import json
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

# -------------------------------
# Logging
# -------------------------------
log_dir = Path(__file__).parent / "run_result"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "advance_english_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Repo root
# -------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.audio_labels_trans_pipeline import AudioLabeler
from benchmarks.calculate_benchmark_score import normalize_text, calculate_wer, calculate_cer

# -------------------------------
# LibriSpeech test-clean directory
# -------------------------------
LIBRI_ROOT = Path("/ocean/projects/cis220031p/xbai2/librispeech/LibriSpeech/test-clean")

if not LIBRI_ROOT.exists():
    raise RuntimeError(f"LibriSpeech directory not found: {LIBRI_ROOT}")

# -------------------------------
# Initialize pipeline
# -------------------------------
logger.info("Initializing AudioLabeler for English...")
try:
    labeler = AudioLabeler(language="en")
    logger.info("AudioLabeler initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AudioLabeler: {e}", exc_info=True)
    sys.exit(1)

# -------------------------------
# Collect flac + transcription pairs
# -------------------------------
wav_list = []

for speaker_dir in sorted(LIBRI_ROOT.glob("*")):
    if not speaker_dir.is_dir():
        continue

    for chapter_dir in sorted(speaker_dir.glob("*")):
        trans_files = list(chapter_dir.glob("*.trans.txt"))
        if len(trans_files) != 1:
            continue

        trans_file = trans_files[0]

        # Load ground truth transcription lines
        lines = trans_file.read_text(encoding="utf-8").strip().split("\n")
        text_dict = {}
        for line in lines:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                utt_id, text = parts
                text_dict[utt_id] = text.strip()

        # Match .flac files with their ground truth
        for flac_file in sorted(chapter_dir.glob("*.flac")):
            utt_id = flac_file.stem  # e.g. "1089-134686-0000"
            if utt_id in text_dict:
                wav_list.append((flac_file, text_dict[utt_id]))

logger.info(f"Found {len(wav_list)} utterances in LibriSpeech test-clean.")

# Only keep first 100
wav_list = wav_list[:100]
logger.info(f"Evaluating first {len(wav_list)} utterances.")

# -------------------------------
# Storage
# -------------------------------
transcriptions_list = []
stats = {
    "whisper": [],
    "gemini": [],
    "judge": [],
    "nemo": []
}

# -------------------------------
# Main loop
# -------------------------------
for idx, (wav_path, ref_text) in enumerate(tqdm(wav_list, desc="Processing files"), 1):
    try:
        logger.info(f"[{idx}/{len(wav_list)}] Processing {wav_path.name}")

        # Normalize reference (English)
        ref_norm = normalize_text(ref_text, "en")

        # Run pipeline
        results = labeler.process_audio(str(wav_path))
        trans = results["transcriptions"]

        whisper_text = trans.get("whisper", "")
        gemini_text = trans.get("gemini", "")
        judge_text  = trans.get("final", "")
        nemo_text   = trans.get("nemo", "")

        transcriptions_list.append({
            "file": wav_path.name,
            "ref": ref_text,
            "whisper_improved": whisper_text,
            "gemini_clean": gemini_text,
            "gemini_clean_judge": judge_text,
            "nemo_canary": nemo_text
        })

        # Metrics helper
        def append_stats(model_name, hypothesis):
            hyp_norm = normalize_text(hypothesis, "en")
            wer, wer_stats = calculate_wer(ref_norm, hyp_norm)
            cer, _ = calculate_cer(ref_norm, hyp_norm)
            ref_words = wer_stats.get("ref_words", 0)

            stats[model_name].append({
                "wer": wer,
                "cer": cer,
                "sub_rate": wer_stats.get("substitutions", 0) / ref_words if ref_words else 0,
                "del_rate": wer_stats.get("deletions", 0) / ref_words if ref_words else 0,
                "ins_rate": wer_stats.get("insertions", 0) / ref_words if ref_words else 0
            })

        append_stats("whisper", whisper_text)
        append_stats("gemini", gemini_text)
        append_stats("judge", judge_text)
        append_stats("nemo", nemo_text)

    except Exception as e:
        logger.error(f"Error processing {wav_path.name}: {e}", exc_info=True)
        continue

# -------------------------------
# Average metrics
# -------------------------------
final_results = {}
for model_name, mstats in stats.items():
    if len(mstats) > 0:
        n = len(mstats)
        final_results[model_name] = {
            "wer": sum(s["wer"] for s in mstats) / n * 100,
            "cer": sum(s["cer"] for s in mstats) / n * 100,
            "sub_rate": sum(s["sub_rate"] for s in mstats) / n * 100,
            "del_rate": sum(s["del_rate"] for s in mstats) / n * 100,
            "ins_rate": sum(s["ins_rate"] for s in mstats) / n * 100,
        }
    else:
        final_results[model_name] = None

# -------------------------------
# Write JSON
# -------------------------------
output = {
    "summary_metrics": final_results,
    "transcriptions": transcriptions_list,
    "num_files": len(transcriptions_list)
}

out_path = Path(__file__).parent / "result_advance_100_english.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info("English ASR evaluation completed.")
print(f"Processed {len(transcriptions_list)} files. Saved to: {out_path}")
