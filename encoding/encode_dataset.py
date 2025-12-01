import gc
import torch
import os
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
from neucodec import NeuCodec

from config_manager import ConfigManager

class DatasetEncoder():
    def __init__(self, output_base_path, config_path="datasets.json"):
        self.config_manager = ConfigManager(config_path)
        self.output_base_path = output_base_path

    def process_datasets(self):
        datasets_config = self.config_manager.load_config()

        # Initialize codec
        codec = NeuCodec.from_pretrained("neuphonic/neucodec")
        codec.eval()

        for config in datasets_config:
            dataset_path = config["path"]
            dataset_name = config["name"]
            dataset_skip = config["skip"]
            dataset_id_col = config["id_col"] if "id_col" in config else "id"
            dataset_audio_col = config["audio_col"] if "audio_col" in config else "audio"
            dataset_text_col = config["text_col"] if "text_col" in config else "text"

            if dataset_skip:
                print(f"Skipping dataset: {dataset_path}/{dataset_name}")
                continue

            dataset_out_path = f"{self.output_base_path}/{dataset_path}/{dataset_name}"
            dataset = load_dataset(dataset_path, dataset_name) # Optional: might need streaming=True if dataset is too large
            try:
                self.extract_audio_files(dataset, dataset_out_path, codec, dataset_text_col, dataset_audio_col, dataset_id_col)
            except KeyboardInterrupt or ValueError:
                print("Writing last index")
                self.config_manager.write_processed_index_files()
                raise

    def extract_audio_files(self, dataset, dataset_out_path, codec, text_column_name="text", audio_column_name="audio", id_column_name="id"):
        for split in dataset.keys():
            sorted_dataset = dataset.sort(id_column_name)
            audio_dataset = sorted_dataset[split].cast_column("audio", Audio(sampling_rate=16000))

            dataset_split_path = f"{dataset_out_path}/{split}"
            os.makedirs(f"{dataset_split_path}/codes", exist_ok=True)
            os.makedirs(f"{dataset_split_path}/text", exist_ok=True)
            os.makedirs(f"{dataset_split_path}/wavs", exist_ok=True)
            processed_index_file = f"{dataset_split_path}/processed_index.json"
            processed_index_config = self.config_manager.load_config(processed_index_file)
            last_processed_index = 0

            if processed_index_config != None and "last_processed_index" in processed_index_config:
                last_processed_index = processed_index_config["last_processed_index"]
                print(f"Creating codes at {dataset_split_path}/codes from index {last_processed_index}")

            for i, row in tqdm(enumerate(audio_dataset)):

                if i < last_processed_index:
                    if i % 1000 == 0:
                        print(f"Still skipping already processed files. Index: {i}")
                    continue

                with torch.no_grad():
                    file_name = row[id_column_name]

                    #Encode audio
                    audio_example = row[audio_column_name]
                    audio_array = audio_example["array"]

                    # Save the wav file
                    wav_file_name = os.path.join(f"{dataset_split_path}/wavs", f"{file_name}.wav")
                    if not os.path.exists(wav_file_name):
                        sampling_rate = audio_example["sampling_rate"]
                        sf.write(wav_file_name, audio_array, sampling_rate)
                        del sampling_rate

                    # Save the codes
                    codes_file_name = os.path.join(f"{dataset_split_path}/codes", f"{file_name}.pt")
                    if not os.path.exists(codes_file_name):
                        wav_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
                        ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
                        torch.save(ref_codes, codes_file_name)
                        del wav_tensor, ref_codes

                    # Save the text file
                    text_file_name = os.path.join(f"{dataset_split_path}/text", f"{file_name}.txt")
                    if not os.path.exists(text_file_name):
                        text = row[text_column_name]
                        with open(text_file_name, "w") as f:
                            f.write(text)
                        del text

                self.config_manager.increase_processing_index(processed_index_file)

                # Clean up to free memory
                del audio_example, audio_array, file_name, codes_file_name, text_file_name, wav_file_name
                torch.cuda.empty_cache()
                gc.collect()

        self.config_manager.write_processed_index_files()


if __name__ == "__main__":
    print(f"Is cuda available:{torch.cuda.is_available()}")
    output_base_path = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data"
    encoder = DatasetEncoder(output_base_path=output_base_path)
    encoder.process_datasets()