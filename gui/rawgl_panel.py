import os
import sys

# Add the src directory to the Python path to import the utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.image_loader import load_png_stack_as_dask_array
from src.dask_pipeline import run_orthogonal_pipeline

# --- Placeholder for a Qt-like QWidget ---
class QWidget:
    def __init__(self):
        print("Placeholder QWidget created.")
# --- End Placeholder ---


class RawGLPanel(QWidget):
    """
    A UI panel for configuring and running the Dask-based RawGL pipeline.
    This class is a placeholder for the UI structure and logic.
    """

    def __init__(self):
        super().__init__()
        print("Initializing RawGLPanel for Dask workflow.")

        # --- Mock UI State ---
        # These attributes represent the values that would be held by UI widgets
        self.ui_input_dir = "path/to/image_stack/"
        self.ui_rawgl_executable_path = "path/to/rawgl"
        self.ui_shader_path = "shaders/my_effect.frag"
        self.ui_output_dir = "output/"
        self.ui_dask_workers = 4  # New UI control for thread/worker count
        # --- End Mock UI State ---

    def _gather_settings_from_ui(self) -> dict:
        """
        In a real UI, this method would read the current values from the widgets.
        Here, it just returns the mock state.
        """
        print("Gathering settings from UI controls...")
        return {
            "input_dir": self.ui_input_dir,
            "output_dir": self.ui_output_dir,
            "num_workers": self.ui_dask_workers,
            "rawgl_config": {
                "executable": self.ui_rawgl_executable_path,
                "shader_path": self.ui_shader_path,
            }
        }

    def on_run_pipeline_clicked(self):
        """
        This method would be connected to a 'Run' button's clicked signal.
        It now orchestrates the entire Dask-based workflow.
        """
        print("\n'Run Pipeline' button clicked.")
        settings = self._gather_settings_from_ui()

        # 1. Load the image stack as a Dask array
        print("\nStep 1: Loading image stack...")
        dask_volume = load_png_stack_as_dask_array(settings["input_dir"])

        if dask_volume is None:
            print("Pipeline run cancelled due to loading error.")
            return

        # 2. Run the orthogonal processing pipeline
        print("\nStep 2: Starting Dask processing pipeline...")
        # This function is blocking as it calls dask.compute() internally.
        # In a real GUI, this would be run in a separate QThread to avoid freezing the UI.
        processed_yz, processed_xz = run_orthogonal_pipeline(
            dask_volume=dask_volume,
            rawgl_config=settings["rawgl_config"],
            num_workers=settings["num_workers"]
        )

        # 3. Save the results
        # In a real app, you might do something more sophisticated. Here we just save one slice.
        print("\nStep 3: Saving sample results...")
        if not os.path.exists(settings["output_dir"]):
            os.makedirs(settings["output_dir"])

        yz_slice_0 = processed_yz[0].compute()
        xz_slice_0 = processed_xz[0].compute()

        Image.fromarray(yz_slice_0).save(os.path.join(settings["output_dir"], "result_yz_000.png"))
        Image.fromarray(xz_slice_0).save(os.path.join(settings["output_dir"], "result_xz_000.png"))

        print(f"Saved sample slices to '{settings['output_dir']}'.")
        print("\nPipeline finished successfully.")


if __name__ == '__main__':
    # This example demonstrates how the updated UI panel would be used.
    # It creates dummy data and runs the full Dask pipeline.
    from PIL import Image
    import numpy as np

    print("--- Running Updated RawGLPanel UI Example ---")

    # --- Create dummy data for the example ---
    dummy_input_dir = "temp_png_stack_for_ui"
    if not os.path.exists(dummy_input_dir):
        os.makedirs(dummy_input_dir)

    print(f"Creating dummy PNG files in '{dummy_input_dir}'...")
    for i in range(10):
        img_data = np.full((128, 128), i * 25, dtype=np.uint8)
        Image.fromarray(img_data, 'L').save(os.path.join(dummy_input_dir, f"slice_{i:03d}.png"))
    # --- End dummy data creation ---

    # 1. Create the UI panel
    ui_panel = RawGLPanel()

    # 2. Override mock UI settings to point to our dummy data
    ui_panel.ui_input_dir = dummy_input_dir
    ui_panel.ui_dask_workers = 2 # Use 2 workers for the example

    # 3. Simulate the user clicking the 'Run' button
    ui_panel.on_run_pipeline_clicked()

    # --- Clean up dummy data ---
    print("\nCleaning up dummy files...")
    for f in os.listdir(dummy_input_dir):
        os.remove(os.path.join(dummy_input_dir, f))
    os.rmdir(dummy_input_dir)
    if os.path.exists("result_yz_000.png"): os.remove("result_yz_000.png")
    if os.path.exists("result_xz_000.png"): os.remove("result_xz_000.png")
    if os.path.exists("output"):
        if len(os.listdir("output")) == 2:
            os.remove("output/result_yz_000.png")
            os.remove("output/result_xz_000.png")
        os.rmdir("output")


    print("\n--- Updated RawGLPanel UI Example Finished ---")
