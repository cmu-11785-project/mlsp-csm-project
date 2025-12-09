import json

class ConfigManager:
    def __init__(self, config_path):
        self.PROCESSING_INDEXES = {}
        self.config_path = config_path

    def increase_processing_index(self, path):
        if path in self.PROCESSING_INDEXES:
            self.PROCESSING_INDEXES[path] += 1
        else:
            self.PROCESSING_INDEXES[path] = 0

    def write_processed_index_files(self):
        print("Writing last index files")
        for path, index in self.PROCESSING_INDEXES.items():
            config = self.load_config(path)

            if config != None and "last_processed_index" in config:
                config["last_processed_index"] = config["last_processed_index"] + index
            else:
                config = {"last_processed_index": index}

            with open(path, "w") as f:
                json.dump(config, f)

    def load_config(self, filename=None):
        """
        Loads a JSON configuration file into a Python dictionary.
        Args:
            filename (str): The path to the JSON configuration file.
        Returns:
            dict: A dictionary containing the loaded configuration data.
        """

        if filename is None:
            filename = self.config_path

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
