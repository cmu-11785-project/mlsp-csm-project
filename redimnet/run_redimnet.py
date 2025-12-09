from pathlib import Path
import torch
import torchaudio
import os

class RedimnetInference():
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Load model pretrained and fine-tuned on vox2, voxblink2 and cn-celeb datasets
        self.model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name="M",
            train_type="ft_mix",
            dataset="vb2+vox2+cnc"
        )

        # Select device and setup inference precision (AMP for GPU)
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device_type}")
        self.device = torch.device(self.device_type)
        self.precision = torch.float16 if self.device_type == "cuda" else torch.float32

        # Setup model evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_embeddings_for_folder(self):
        for index, audio_file in enumerate(sorted(self.input_dir.glob("*.wav"))):
            out_file = f"{Path(self.output_dir) / audio_file.stem}.pt"

            if Path(out_file).is_file():
                print(f"Skipping already processed file {out_file}")
                continue

            self.get_embeddings_for_file(audio_file, out_file)

    def get_embeddings_for_file(self, audio_path, out_file):

        # Load audio samples
        samples, fs = torchaudio.load(audio_path)  # shape [1, T]
        assert fs == 16000, f"Audio sampling rate {fs} != 16000"
        assert samples.shape[0] == 1, f"Expected mono audio, but got {samples.shape[0]} channels"

        with torch.no_grad(), torch.autocast(device_type=self.device_type, dtype=self.precision):
            # Model input is [N, T], where N - batch size, T - samples length
            embedding = self.model(samples.to(self.device))

            print(f"Writing embeddings to file {out_file}")
            torch.save(embedding, out_file)

if __name__ == "__main__":
    #input_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/speechcolab/gigaspeech/xs/test/wavs" # original wavs
    input_dir = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/neutts/output/xs/test/resampled"
    #out_dir = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/redimnet/output/original"
    out_dir = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/redimnet/output/inference"
    redimnet = RedimnetInference(input_dir=input_dir, output_dir=out_dir)
    redimnet.get_embeddings_for_folder()