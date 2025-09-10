import os
import sys
from typing import List, Dict

# Add the src directory to the Python path to import the utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.rawgl_wrapper import RawGLWrapper
from src.pipeline_controller import PipelineController

# --- Placeholder for a Qt-like QWidget ---
# In a real application, this would be `from PySide6.QtWidgets import QWidget`
class QWidget:
    def __init__(self):
        print("Placeholder QWidget created.")
# --- End Placeholder ---


class RawGLPanel(QWidget):
    """
    A UI panel for configuring and running RawGL processing jobs.
    This class is a placeholder for the UI structure and logic.
    """

    def __init__(self, controller: PipelineController):
        super().__init__()
        print("Initializing RawGLPanel.")
        self.controller = controller

        # --- Mock UI State ---
        # These attributes represent the values that would be held by UI widgets
        # like QLineEdit, QSpinBox, QCheckBox, etc.
        self.ui_rawgl_executable_path = "path/to/rawgl"
        self.ui_shader_path = "shaders/my_effect.frag"
        self.ui_input_files = ["images/input1.png", "images/input2.png"]
        self.ui_output_dir = "output/"
        self.ui_width = 512
        self.ui_height = 512
        self.ui_is_greyscale_png = True
        # --- End Mock UI State ---

    def _gather_settings_from_ui(self) -> Dict:
        """
        In a real UI, this method would read the current values from the widgets.
        Here, it just returns the mock state.
        """
        print("Gathering settings from UI controls...")
        return {
            "executable": self.ui_rawgl_executable_path,
            "shader": self.ui_shader_path,
            "inputs": self.ui_input_files,
            "output_dir": self.ui_output_dir,
            "size": (self.ui_width, self.ui_height),
            "greyscale": self.ui_is_greyscale_png
        }

    def on_run_pipeline_clicked(self):
        """
        This method would be connected to a 'Run' button's clicked signal.

        It gathers settings, creates jobs, and submits them to the controller.
        """
        print("\n'Run Pipeline' button clicked.")
        settings = self._gather_settings_from_ui()

        if not os.path.exists(settings["output_dir"]):
            os.makedirs(settings["output_dir"])
            print(f"Created output directory: {settings['output_dir']}")

        print(f"Preparing to submit {len(settings['inputs'])} jobs to the pipeline.")
        for input_file in settings["inputs"]:
            # Create a unique output path for each input file
            base_name = os.path.basename(input_file)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(settings["output_dir"], f"{name}_processed.png")

            # Create and configure a RawGL job
            job = RawGLWrapper(settings["executable"])
            job.add_pass(
                shader_path=settings["shader"],
                output_size=settings["size"],
                output_path=output_path,
                inputs={'u_texture': input_file},  # Assuming one input texture uniform
                is_greyscale_png=settings["greyscale"]
            )

            # Submit the job to the controller
            self.controller.submit_job(job)

        print("\nAll jobs submitted. The pipeline is running in the background.")
        # In a real app, you might update a progress bar here.
        # For this example, we'll just wait for completion immediately.
        self.controller.wait_for_completion()
        self.controller.shutdown()


if __name__ == '__main__':
    # This example demonstrates how the UI panel would be instantiated and used.
    print("--- Running RawGLPanel UI Example ---")

    # 1. Create the backend controller
    pipeline_controller = PipelineController(max_workers=2)

    # 2. Create the UI panel, passing it the controller
    ui_panel = RawGLPanel(controller=pipeline_controller)

    # 3. Simulate the user clicking the 'Run' button
    ui_panel.on_run_pipeline_clicked()

    print("\n--- RawGLPanel UI Example Finished ---")
