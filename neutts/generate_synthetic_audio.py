import os
from pathlib import Path
import argparse
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from neutts.basic_streaming_example import main

def process_folder(text_folder, codes_folder, backbone, output_folder='neutts_outputs', start_value=0):
    """
    Process all text and code files in the given folders.

    Args:
        text_folder: Path to folder containing .txt files
        codes_folder: Path to folder containing .pt code files
        backbone: The backbone model to use
        output_folder: Optional folder to save outputs (not used in streaming mode)
    """
    text_folder = Path(text_folder)
    codes_folder = Path(codes_folder)

    # Get all text files
    text_files = sorted(text_folder.glob("*.txt"))

    if not text_files:
        print(f"No text files found in {text_folder}")
        return

    print(f"Found {len(text_files)} text files to process")

    for i, text_file in enumerate(text_files):

        if (i < start_value):
            if (i % 20 == 0):
                print(f"skipping file. index: {i}")
            continue

        # Construct the corresponding codes file path
        # Assumes matching filenames (e.g., file1.txt -> file1.pt)
        codes_file = codes_folder / f"{text_file.stem}.pt"

        if not codes_file.exists():
            print(f"Warning: Codes file not found for {text_file.name}, skipping...")
            continue

        # Read the input text
        with open(text_file, "r") as f:
            input_text = f.read().strip()

        # Skip if wav file already exists in output folder
        wav_file = f"{output_folder}/{text_file.stem}.wav"
        print(f"Output wav file will be saved to: {wav_file}")
        if os.path.isfile(wav_file):
                print(f"Output wav file already exists for {text_file.name}, skipping...")
                continue

        print(f"\n{'='*60}")
        print(f"Processing: {text_file.name}")
        print(f"Text file: {text_file}")
        print(f"Codes file: {codes_file}")
        print(f"{'='*60}\n")

        try:
            # Call the main function with the text content, codes path, and ref_text path
            # Note: Using the text file itself as ref_text (you may want to adjust this)
            main(
                input_text=input_text,
                ref_codes_path=str(codes_file),
                ref_text=str(text_file),  # Using same text as reference
                backbone=backbone,
                output_path=output_folder
            )
            print(f"✓ Successfully processed {text_file.name}\n")
        except Exception as e:
            print(f"✗ Error processing {text_file.name}: {str(e)}\n")
            continue


if __name__ == "__main__":
    data_folder = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/mythicinfinity/libritts/clean"
    split = "dev.clean"
    parser = argparse.ArgumentParser(
        description="Batch process TTS for multiple files"
    )
    parser.add_argument(
        "--text_folder",
        type=str,
        default=f"{data_folder}/{split}/text",
        help="Folder containing text files"
    )
    parser.add_argument(
        "--codes_folder",
        type=str,
        default=f"{data_folder}/{split}/codes",
        help="Folder containing code files (.pt)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-air-q8-gguf",
        help="Huggingface repo containing the backbone checkpoint"
    )
    output_folder = f"/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/neutts/output/libritts/clean/{split}"
    os.makedirs(output_folder, exist_ok=True)
    parser.add_argument(
        "--output_folder",
        type=str,
        default=output_folder,
        help="Optional output folder (not used in streaming mode)"
    )

    args = parser.parse_args()

    process_folder(
        text_folder=args.text_folder,
        codes_folder=args.codes_folder,
        backbone=args.backbone,
        output_folder=args.output_folder,
        start_value=0
    )