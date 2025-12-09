from datasets import load_dataset
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_manager import ConfigManager

from tqdm import tqdm

class DatasetCleaner:
    def __init__(self, output_base_path, config_path="datasets.json"):
        self.config_manager = ConfigManager(config_path)
        self.output_base_path = output_base_path

    def process_dataset(self):
        datasets_config = self.config_manager.load_config()
        for config in datasets_config:
            dataset_path = config["path"]
            dataset_name = config["name"]
            dataset_skip = config["skip"]
            dataset_id_col = config["id_col"] if "id_col" in config else "id"
            dataset_text_col = config["text_column_name"] if "text_column_name" in config else "text"

            if dataset_skip:
                print(f"Skipping dataset: {dataset_path}/{dataset_name}")
                continue

            dataset_out_path = f"{self.output_base_path}/{dataset_path}/{dataset_name}"
            dataset = load_dataset(dataset_path, dataset_name) # Optional: might need streaming=True if dataset is too large

            for split in dataset.keys():
                sorted_dataset = dataset[split].sort(dataset_id_col)
                dataset_split_path = f"{dataset_out_path}/{split}"
                os.makedirs(f"{dataset_split_path}/clean_text", exist_ok=True)
                processed_index_file = f"{dataset_split_path}/processed_index.json"
                processed_index_config = self.config_manager.load_config(processed_index_file)
                last_clean_processed_index = 0

                if processed_index_config != None and "last_clean_processed_index" in processed_index_config:
                    last_clean_processed_index = processed_index_config["last_clean_processed_index"]
                    print(f"Creating clean text at {dataset_split_path}/clean_text from index {last_clean_processed_index}")

                for i, row in tqdm(enumerate(sorted_dataset)):

                    if i < last_clean_processed_index:
                        if i % 1000 == 0:
                            print(f"Still skipping already processed files. Index: {i}")
                        continue

                    clean_text = self.capitalize_remove_punctuation(row[dataset_text_col])

                    file_name = row[dataset_id_col]
                    text_file_name = os.path.join(f"{dataset_split_path}/clean_text", f"{file_name}.txt")
                    with open(text_file_name, "w") as f:
                        f.write(clean_text)

    def process_files_in_directory(self, input_dir, output_dir, action):
        os.makedirs(output_dir, exist_ok=True)
        for text_file in sorted(os.listdir(input_dir)):
            if text_file.endswith(".txt"):
                input_file_path = os.path.join(input_dir, text_file)
                output_file_path = os.path.join(output_dir, text_file)

                with open(input_file_path, "r") as f:
                    text = f.read()

                if action == "normalize_text":
                    processed_text = self.normalize_text(text)
                elif action == "capitalize_remove_punctuation":
                    processed_text = self.capitalize_remove_punctuation(text)
                elif action == "extreme":
                    processed_text = self.extreme_normalize(text)
                elif action == "both":
                    processed_text = self.capitalize_remove_punctuation(text)
                    processed_text = self.normalize_text(processed_text)
                else:
                    continue

                with open(output_file_path, "w") as f:
                    f.write(processed_text)

    def normalize_text(self, text):
        """
        Normalize transcription text by lowercasing and removing extra spaces.

        Args:
            text: Input transcription text
        """
        # Lowercase the text
        text = text.lower()
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def capitalize_remove_punctuation(self, text):
        """
        Clean transcription by replacing punctuation tags and removing garbage tags.

        Args:
            text: Input transcription text

        Returns:
            Cleaned transcription text
        """
        # Replace punctuation tags
        text = text.replace(' <COMMA>', ',')
        text = text.replace(' <PERIOD>', '.')
        text = text.replace(' <QUESTIONMARK>', '?')
        text = text.replace(' <EXCLAMATIONPOINT>', '!')

        # Remove garbage utterance tags
        garbage_tags = ['<SIL>', '<MUSIC>', '<NOISE>', '<OTHER>']
        for tag in garbage_tags:
            text = text.replace(tag, '')

        # Capitalize first letter of each sentence
        result = []
        capitalize_next = True
        for char in text:
            if char.isalpha():
                if capitalize_next:
                    result.append(char.upper())
                    capitalize_next = False
                else:
                    result.append(char.lower())
            else:
                result.append(char)
                if char in '.!?':
                    capitalize_next = True

        text = ''.join(result)

        # Replace standalone "i" and "i'" contractions with capital I
        text = text.replace(' i ', ' I ')
        text = text.replace(' i\'', ' I\'')
        if text.startswith('i '):
            text = 'I' + text[1:]
        if text.startswith('i\''):
            text = 'I' + text[1:]

        return text

    def extreme_normalize(self, text):
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Use Regex to remove anything that isn't a word character or whitespace
        # r'[^\w\s]' matches any character that is NOT alphanumeric or a space
        text = re.sub(r'[^\w\s]', '', text)
        
        return " ".join(text.split())

# Example usage
if __name__ == "__main__":
    cleaner = DatasetCleaner(output_base_path="/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/encoding/output/libritts/clean/dev.clean/original/normalized")
    #cleaner.process_dataset()
    #normalized_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/speechcolab/gigaspeech/xs/test/normalized_text"
    #clean_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/speechcolab/gigaspeech/xs/test/clean_text"

    input_dir = "/ocean/projects/cis220031p/shared/11785-project/neutts-air/data/mythicinfinity/libritts/clean/dev.clean/text"
    output_dir = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/encoding/output/libritts/clean/dev.clean/original/normalized"

    cleaner.process_files_in_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        action="extreme")
