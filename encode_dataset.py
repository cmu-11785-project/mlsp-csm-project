import gc
import torch
import os
import json
from datasets import load_dataset, Audio
from tqdm import tqdm
from neucodec import NeuCodec

def process_datasets():
    datasets_config = load_config("datasets.json")

    output_base_path = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data"

    # Initialize codec
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")

    for config in datasets_config:
        dataset_path = config["path"]
        dataset_name = config["name"]
        dataset_skip = config["skip"]

        if dataset_skip:
            print(f"Skipping dataset: {dataset_path}/{dataset_name}")
            continue

        dataset_audio_column_name = config["audio_column_name"]
        output_dataset_codes_path = f"{output_base_path}/{dataset_path}/{dataset_name}/codes"
        os.makedirs(output_dataset_codes_path, exist_ok=True)

        dataset = load_dataset(dataset_path, dataset_name, streaming=True)
        extract_audio_files(dataset, output_dataset_codes_path, codec, dataset_audio_column_name)

def extract_audio_files(dataset, output_ds_codes_path, codec, audio_column_name="audio"):
    for split in dataset.keys():
        audio_dataset = dataset[split].cast_column("audio", Audio(sampling_rate=16000))
        for i, row in tqdm(enumerate(audio_dataset)):
            if i >= 100:
                break

            audio_example = row[audio_column_name]
            audio_array = audio_example["array"]
            wav_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
            ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)

            # Save the codes
            file_output_path = os.path.join(output_ds_codes_path, f"{row['id']}.pt")
            torch.save(ref_codes, file_output_path)

            # Clean up to free memory
            del wav_tensor, ref_codes, audio_example, audio_array
            torch.cuda.empty_cache()
            gc.collect()


def load_config(filename):
    """
    Loads a JSON configuration file into a Python dictionary.
    Args:
        filename (str): The path to the JSON configuration file.
    Returns:
        dict: A dictionary containing the loaded configuration data.
    """
    try:
        with open(filename, 'r') as f:
            config_data = json.load(f)
        return config_data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file '{filename}'. Check for malformed JSON.")
        return None


print(torch.cuda.is_available())
process_datasets()