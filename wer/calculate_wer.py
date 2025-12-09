from pathlib import Path
from jiwer import wer

class WERCalculator:
    def __init__(self, reference_folder, hypothesis_folder):
        """
        Initialize WER calculator with reference and hypothesis folders.

        Args:
            reference_folder: Path to folder containing reference txt files
            hypothesis_folder: Path to folder containing hypothesis txt files
        """
        self.reference_folder = Path(reference_folder)
        self.hypothesis_folder = Path(hypothesis_folder)

    def calculate_wer_for_files(self, ref_file, hyp_file):
        # Read texts
        with open(ref_file, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()

        with open(hyp_file, 'r', encoding='utf-8') as f:
            hypothesis_text = f.read().strip()

        # Calculate WER
        file_wer = wer(reference_text, hypothesis_text)

        return file_wer

    def calculate_average_wer(self):
        """
        Calculate average WER between reference and hypothesis texts.

        Returns:
            float: Average WER across all file pairs
        """
        total_wer = 0
        file_count = 0
        non_matching_file_count = 0

        # Get sorted list of reference files
        ref_files = sorted(self.reference_folder.glob("*.txt"))

        for ref_file in ref_files:
            hyp_file = self.hypothesis_folder / ref_file.name

            if not hyp_file.exists():
                print(f"Warning: No matching hypothesis file for {ref_file.name}")
                non_matching_file_count += 1
                continue

            file_wer = self.calculate_wer_for_files(ref_file, hyp_file)

            total_wer += file_wer
            file_count += 1

        if file_count == 0:
            raise ValueError("No matching file pairs found")

        average_wer = total_wer / file_count
        non_matching_percentage = (non_matching_file_count / len(ref_files)) * 100
        print(f"Non-matching files: {non_matching_file_count} out of {len(ref_files)} ({non_matching_percentage:.2f}%)")
        return average_wer


if __name__ == "__main__":
    calculator = WERCalculator(
        "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/whisper/output/libritts/clean/dev.clean/inference/normalized",
        "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/encoding/output/libritts/clean/dev.clean/original/normalized")
    avg_wer = calculator.calculate_average_wer()
    print(f"Average WER: {avg_wer:.4f}")

    """
    wer_sc = calculator.calculate_wer_for_files(
        Path("/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/speechcolab/gigaspeech/xs/test/normalized_text/POD1000000005_S0000237.txt"),
        Path("/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/whisper/output/xs/test/inference/text_normalized/POD1000000005_S0000237.txt"))
    print(wer_sc)
    """