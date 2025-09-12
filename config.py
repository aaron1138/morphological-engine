# config.py

import json

class AppConfig:
    def __init__(self):
        self.input_mode = "folder"
        self.input_folder = ""
        self.output_folder = ""
        self.uvtools_path = ""
        self.uvtools_temp_folder = ""
        self.uvtools_input_file = ""
        self.thread_count = 4

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, filepath):
        config = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
            config.__dict__.update(data)
        return config

app_config = AppConfig()
