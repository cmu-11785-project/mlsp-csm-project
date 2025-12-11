#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        return iterable

# Set up logging
log_dir = Path(__file__).parent / "run_result"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'advance_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.audio_labels_trans_pipeline import AudioLabeler
from benchmarks.calculate_benchmark_score import buckwalter_to_arabic, normalize_text, calculate_wer, calculate_cer

# -------------------------------
# File system configuration
# -------------------------------
ROOT_DIR = Path("/ocean/projects/cis220031p/xbai2/arabic_speech_corpus/arabic-speech-corpus")
WAV_DIR = ROOT_DIR / "wav"
LAB_DIR = ROOT_DIR / "lab"

# -------------------------------
# Initialize pipeline
# -------------------------------
logger.info("Initializing AudioLabeler...")
try:
    labeler = AudioLabeler(language="ar")
    logger.info("AudioLabeler initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AudioLabeler: {e}", exc_info=True)
    sys.exit(1)

# -------------------------------
# Collect WAV files (main corpus only)
# -------------------------------
logger.info(f"Scanning WAV files in {WAV_DIR}...")
wav_files = sorted(WAV_DIR.glob("*.wav"))   # Sort by filename (dictionary order)
logger.info(f"Found {len(wav_files)} WAV files under main corpus.")

# Limit to first 100
MAX_FILES = 100
if len(wav_files) > MAX_FILES:
    wav_files = wav_files[:MAX_FILES]
logger.info(f"Processing first {len(wav_files)} WAV files (sorted by filename).")

# -------------------------------
# Storage for results
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
for idx, wav_file in enumerate(tqdm(wav_files, desc="Processing files"), 1):
    base_name = wav_file.stem
    lab_file = LAB_DIR / f"{base_name}.lab"

    if not lab_file.exists():
        logger.error(f"Missing label file for {wav_file} → expected {lab_file}")
        raise FileNotFoundError(f"Label file not found for {wav_file}")

    # Read ground truth
    ref_text = lab_file.read_text(encoding="utf-8").strip()
    ref_arabic = buckwalter_to_arabic(ref_text)
    ref_norm = normalize_text(ref_arabic, "ar")

    try:
        logger.info(f"[{idx}/{len(wav_files)}] Processing {wav_file.name}")

        # Run pipeline
        results = labeler.process_audio(str(wav_file))
        trans = results['transcriptions']

        whisper_text = trans.get('whisper', '')
        gemini_text = trans.get('gemini', '')
        judge_text = trans.get('final', '')
        nemo_text = trans.get('nemo', '')

        # Store transcription record
        transcriptions_list.append({
            "file": wav_file.name,
            "ref": ref_text,
            "whisper_improved": whisper_text,
            "gemini_clean": gemini_text,
            "gemini_clean_judge": judge_text,
            "nemo_canary": nemo_text
        })

        # Metric helper
        def append_stats(model_name, hypothesis):
            hyp_norm = normalize_text(hypothesis, "ar")
            wer, wer_stats = calculate_wer(ref_norm, hyp_norm)
            cer, _ = calculate_cer(ref_norm, hyp_norm)
            ref_words = wer_stats.get('ref_words', 0)

            file_stats = {
                "wer": wer,
                "cer": cer,
                "sub_rate": wer_stats.get("substitutions", 0) / ref_words if ref_words > 0 else 0,
                "del_rate": wer_stats.get("deletions", 0) / ref_words if ref_words > 0 else 0,
                "ins_rate": wer_stats.get("insertions", 0) / ref_words if ref_words > 0 else 0
            }
            stats[model_name].append(file_stats)
            return file_stats

        append_stats("whisper", whisper_text)
        append_stats("gemini", gemini_text)
        append_stats("judge", judge_text)
        append_stats("nemo", nemo_text)

        logger.info(f"Completed {wav_file.name}")

    except Exception as e:
        logger.error(f"Error processing {wav_file.name}: {e}", exc_info=True)
        continue

# -------------------------------
# Compute average metrics
# -------------------------------
final_results = {}
for model_name, model_stats in stats.items():
    if len(model_stats) > 0:
        n = len(model_stats)
        final_results[model_name] = {
            "wer": sum(s["wer"] for s in model_stats) / n * 100,
            "cer": sum(s["cer"] for s in model_stats) / n * 100,
            "sub_rate": sum(s["sub_rate"] for s in model_stats) / n * 100,
            "del_rate": sum(s["del_rate"] for s in model_stats) / n * 100,
            "ins_rate": sum(s["ins_rate"] for s in model_stats) / n * 100,
        }
    else:
        final_results[model_name] = None

# -------------------------------
# Save final JSON
# -------------------------------
output_data = {
    "summary_metrics": final_results,
    "transcriptions": transcriptions_list,
    "num_files": len(transcriptions_list)
}

output_file = Path(__file__).parent / "result_advance_100.json"
logger.info(f"Saving results to {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

logger.info("Advanced evaluation (100-file version) completed successfully!")
print(f"Processed {len(transcriptions_list)} files. Results saved to {output_file}")

