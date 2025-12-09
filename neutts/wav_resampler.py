import os
import librosa
import soundfile as sf
from pathlib import Path
import warnings

class AudioResampler:
    def __init__(self, input_dir, output_dir, target_sr=16000):
        """
        Initialize the AudioResampler.

        Args:
            input_dir (str): Path to the folder containing source wav files.
            output_dir (str): Path to save the resampled files.
            target_sr (int): Target sample rate (default: 16000).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_batch(self):
        """
        Iterates through the input directory and processes all .wav files.
        """
        wav_files = list(self.input_dir.glob("*.wav"))

        if not wav_files:
            print(f"No .wav files found in {self.input_dir}")
            return

        print(f"Found {len(wav_files)} files. Starting resampling to {self.target_sr}Hz...")

        for file_path in wav_files:
            try:
                self.resample_and_save(file_path)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        print("Batch processing complete.")

    def resample_and_save(self, file_path):
        """
        Helper method to load, resample, and save a single file.
        """
        output_path = self.output_dir / file_path.name

        if os.path.exists(output_path.as_posix()):
                print(f"Skipping: {output_path.as_posix()} (already exists)")
                return

        # librosa.load automatically resamples if 'sr' is provided
        # We catch warnings to suppress PySoundFile warnings on some metadata tags
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(file_path, sr=self.target_sr)

        # Write the file using soundfile (subtype='PCM_16' is standard for wav)
        sf.write(output_path, y, self.target_sr, subtype='PCM_16')
        print(f"Processed: {file_path.name}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your folders
    input_folder = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/neutts/output/xs/test/raw"
    output_folder = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/neutts/output/xs/test/resampled"

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Initialize and run
    resampler = AudioResampler(input_folder, output_folder, target_sr=16000)
    resampler.process_batch()