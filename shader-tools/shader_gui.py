# Frontend GUI for the shader tool.

import os
import glob
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    Button, E, Entry, Frame, Label, N, S, W, X, Y,
    StringVar, Tk, filedialog, messagebox, scrolledtext
)
from tkinter import ttk

class ShaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Shader Tool")
        self.root.minsize(600, 500)

        # --- Member variables ---
        self.input_image_path = StringVar()
        self.input_folder_path = StringVar()
        self.shader_preset_path = StringVar()
        self.output_folder_path = StringVar()
        self.processing_thread = None
        self.log_queue = queue.Queue()

        # --- UI Setup ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(N, S, E, W))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # --- Input Configuration ---
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, columnspan=3, sticky=(E, W, N, S), pady=5)
        input_frame.grid_columnconfigure(1, weight=1)

        # Single Image
        ttk.Label(input_frame, text="Input Image:").grid(row=0, column=0, sticky=W, pady=2)
        ttk.Entry(input_frame, textvariable=self.input_image_path).grid(row=0, column=1, sticky=(E, W))
        ttk.Button(input_frame, text="Browse...", command=self.select_input_image).grid(row=0, column=2, padx=5)

        # OR Folder
        ttk.Label(input_frame, text="Input Folder:").grid(row=1, column=0, sticky=W, pady=2)
        ttk.Entry(input_frame, textvariable=self.input_folder_path).grid(row=1, column=1, sticky=(E, W))
        ttk.Button(input_frame, text="Browse...", command=self.select_input_folder).grid(row=1, column=2, padx=5)

        # --- Shader and Output Configuration ---
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(E, W, N, S), pady=5)
        config_frame.grid_columnconfigure(1, weight=1)

        # Shader Preset
        ttk.Label(config_frame, text="Shader Preset:").grid(row=0, column=0, sticky=W, pady=2)
        ttk.Entry(config_frame, textvariable=self.shader_preset_path).grid(row=0, column=1, sticky=(E, W))
        ttk.Button(config_frame, text="Browse...", command=self.select_shader_preset).grid(row=0, column=2, padx=5)

        # Output Folder
        ttk.Label(config_frame, text="Output Folder:").grid(row=1, column=0, sticky=W, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_folder_path).grid(row=1, column=1, sticky=(E, W))
        ttk.Button(config_frame, text="Browse...", command=self.select_output_folder).grid(row=1, column=2, padx=5)

        # --- Controls and Logging ---
        # Start Button
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=2, column=0, columnspan=3, pady=10)

        # Log Area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(N, S, E, W))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)

        self.log_widget = scrolledtext.ScrolledText(log_frame, state="disabled", wrap="word", height=10)
        self.log_widget.grid(row=0, column=0, sticky=(N, S, E, W))

        # --- Start periodic queue check ---
        self.root.after(100, self.check_log_queue)

    def _log(self, message):
        """Helper to add a message to the log widget."""
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", message + "\n")
        self.log_widget.configure(state="disabled")
        self.log_widget.see("end")

    def select_input_image(self):
        path = filedialog.askopenfilename(title="Select Input Image")
        if path:
            self.input_image_path.set(path)
            self.input_folder_path.set("") # Clear other input

    def select_input_folder(self):
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.input_folder_path.set(path)
            self.input_image_path.set("") # Clear other input

    def select_shader_preset(self):
        path = filedialog.askopenfilename(title="Select .slangp Preset", filetypes=[("Slang Presets", "*.slangp")])
        if path:
            self.shader_preset_path.set(path)

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder_path.set(path)

    def check_log_queue(self):
        """Periodically check the queue for messages from the worker thread."""
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                if message == "DONE":
                    self.start_button.config(state="normal")
                    self._log("\n>>> Processing complete. <<<")
                else:
                    self._log(message)
            except queue.Empty:
                pass
        self.root.after(100, self.check_log_queue)

    def start_processing(self):
        """Validate inputs and start the processing thread."""
        # --- Validation ---
        if not self.input_image_path.get() and not self.input_folder_path.get():
            messagebox.showerror("Error", "Please select an input image or an input folder.")
            return
        if not self.shader_preset_path.get():
            messagebox.showerror("Error", "Please select a shader preset file.")
            return
        if not self.output_folder_path.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self.log_widget.configure(state="normal")
        self.log_widget.delete(1.0, "end")
        self.log_widget.configure(state="disabled")

        self._log(">>> Starting processing... <<<")
        self.start_button.config(state="disabled")

        # --- Start Thread ---
        self.processing_thread = threading.Thread(
            target=self._processing_thread_worker,
            daemon=True
        )
        self.processing_thread.start()

    def _processing_thread_worker(self):
        """The actual workhorse function that runs in a separate thread."""
        try:
            # --- Gather inputs ---
            shader_preset = self.shader_preset_path.get()
            output_folder = Path(self.output_folder_path.get())

            image_files = []
            if self.input_image_path.get():
                image_files.append(Path(self.input_image_path.get()))
            else:
                folder = self.input_folder_path.get()
                self.log_queue.put(f"Scanning folder: {folder}")
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    image_files.extend(Path(folder).glob(ext))

            if not image_files:
                self.log_queue.put("No image files found to process.")
                return

            # --- Locate the backend script ---
            # Assume it's in the same directory as the GUI script
            backend_script = Path(__file__).parent / "shader_tool.py"
            if not backend_script.exists():
                self.log_queue.put(f"Error: Backend script not found at {backend_script}")
                return

            # --- Process each image ---
            for i, image_path in enumerate(image_files):
                self.log_queue.put(f"\n--- Processing file {i+1}/{len(image_files)}: {image_path.name} ---")

                output_path = output_folder / f"{image_path.stem}_processed.png"

                command = [
                    sys.executable, # Use the same python interpreter
                    str(backend_script),
                    str(image_path),
                    str(shader_preset),
                    str(output_path)
                ]

                # Run the subprocess
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )

                # Stream output to the queue in real-time
                for line in iter(process.stdout.readline, ''):
                    self.log_queue.put(line.strip())

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    self.log_queue.put(f"Successfully processed {image_path.name}")
                else:
                    self.log_queue.put(f"Error processing {image_path.name}. See log above.")

        except Exception as e:
            import traceback
            self.log_queue.put(f"An unexpected error occurred in the GUI thread: {e}")
            self.log_queue.put(traceback.format_exc())
        finally:
            # --- Signal completion ---
            self.log_queue.put("DONE")


def main():
    root = Tk()
    app = ShaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
