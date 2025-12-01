from pathlib import Path
import whisper
import sys
import os

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))


class WhisperTranscriptor:
    def __init__(self, input_dir, out_dir, model_name="base", skip=0):
        # Load model
        self.model = whisper.load_model(model_name)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.input_dir = Path(input_dir)
        self.skip = skip

    def transcribe_all(self):
        for index, audio_file in enumerate(sorted(self.input_dir.glob("*.wav"))):
            if index < self.skip:
                if index % 100 == 0:
                    print(f"Skipping index {index}")
                continue
            self.transcribe(audio_file)

    def transcribe(self, audio_path):

        # Transcribe a WAV file
        result = self.model.transcribe(audio_path.as_posix())

        # Save to file
        out_file = f"{Path(self.out_dir) / audio_path.stem}.txt"
        print(out_file) #result["text"])
        with open(out_file, "w") as f:
            f.write(result["text"])

if __name__ == "__main__":
    input_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/speechcolab/gigaspeech/xs/test/wavs"
    out_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/whisper/output/xs/test/original/text"
    transcriptor = WhisperTranscriptor(model_name="base", input_dir=input_dir, out_dir=out_dir)
    transcriptor.transcribe_all()