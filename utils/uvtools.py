# utils/uvtools.py
import subprocess
import os

class UVTools:
    def __init__(self, uvtools_path):
        if not os.path.exists(uvtools_path):
            raise FileNotFoundError(f"UVToolsCmd.exe not found at: {uvtools_path}")
        self.uvtools_path = uvtools_path

    def extract(self, input_file, output_folder):
        """
        Executes UVToolsCmd.exe to extract layers from a slice file.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input slice file not found: {input_file}")
        os.makedirs(output_folder, exist_ok=True)

        command = [self.uvtools_path, "extract", input_file, output_folder, "--content", "Layers"]

        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            return process.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UVTools extraction failed with exit code {e.returncode}:\n{e.stderr}")

    def repack(self, input_file, uvtop_file, output_file):
        """
        Executes UVToolsCmd.exe to repack layers into a slice file.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input slice file not found: {input_file}")
        if not os.path.exists(uvtop_file):
            raise FileNotFoundError(f"UVTools operation file not found: {uvtop_file}")

        command = [self.uvtools_path, "run", input_file, uvtop_file, "--output", output_file]

        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            return process.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"UVTools repacking failed with exit code {e.returncode}:\n{e.stderr}")

    @staticmethod
    def generate_uvtop_file(image_folder, output_filepath):
        """
        Generates the .uvtop XML file for repacking.
        """
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith('.png')])
        if not image_files:
            raise RuntimeError("No processed image files found to generate .uvtop file.")

        xml_content = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\\n'
        xml_content += '<OperationLayerImport xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\\n'
        xml_content += '  <LayerRangeSelection>None</LayerRangeSelection>\\n'
        xml_content += '  <ImportType>Replace</ImportType>\\n'
        xml_content += '  <Files>\\n'
        for f_path in image_files:
            xml_content += '    <GenericFileRepresentation>\\n'
            xml_content += f'      <FilePath>{f_path}</FilePath>\\n'
            xml_content += '    </GenericFileRepresentation>\\n'
        xml_content += '  </Files>\\n'
        xml_content += '</OperationLayerImport>\\n'

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(xml_content)
