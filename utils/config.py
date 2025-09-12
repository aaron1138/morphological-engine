# utils/config.py
import json

class Config:
    def __init__(self):
        self.input_folder = ""
        self.output_folder = ""
        self.uvtools_path = ""
        self.uvtools_input_file = ""
        self.uvtools_temp_folder = ""
        self.thread_count = 4

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = Config()
        config.__dict__.update(data)
        return config
