import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shader Image Processor")
        self.geometry("800x600")

        self.processing_queue = queue.Queue()
        self.is_processing = False

        # --- UI Elements ---
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input Path
        input_frame = tk.LabelFrame(main_frame, text="Input", padx=5, pady=5)
        input_frame.pack(fill=tk.X, pady=5)

        self.input_image_var = tk.StringVar()
        self.input_folder_var = tk.StringVar()

        tk.Label(input_frame, text="Image:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(input_frame, textvariable=self.input_image_var, width=80).grid(row=0, column=1, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self.browse_input_image).grid(row=0, column=2, padx=5)

        tk.Label(input_frame, text="Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(input_frame, textvariable=self.input_folder_var, width=80).grid(row=1, column=1, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self.browse_input_folder).grid(row=1, column=2, padx=5)

        input_frame.columnconfigure(1, weight=1)

        # Shader and Output Path
        config_frame = tk.LabelFrame(main_frame, text="Configuration", padx=5, pady=5)
        config_frame.pack(fill=tk.X, pady=5)

        self.shader_preset_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()

        tk.Label(config_frame, text="Shader Preset:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(config_frame, textvariable=self.shader_preset_var, width=80).grid(row=0, column=1, sticky="ew")
        tk.Button(config_frame, text="Browse...", command=self.browse_shader).grid(row=0, column=2, padx=5)

        tk.Label(config_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(config_frame, textvariable=self.output_folder_var, width=80).grid(row=1, column=1, sticky="ew")
        tk.Button(config_frame, text="Browse...", command=self.browse_output_folder).grid(row=1, column=2, padx=5)

        config_frame.columnconfigure(1, weight=1)

        # Controls
        self.start_button = tk.Button(main_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=10)

        # Log
        log_frame = tk.LabelFrame(main_frame, text="Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_widget = ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=10)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Append a message to the log widget."""
        self.log_widget.config(state='normal')
        self.log_widget.insert(tk.END, message + '\n')
        self.log_widget.see(tk.END)
        self.log_widget.config(state='disabled')
        self.update_idletasks()

    def browse_input_image(self):
        path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
        )
        if path:
            self.input_image_var.set(path)
            self.input_folder_var.set("") # Clear folder if image is selected

    def browse_input_folder(self):
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.input_folder_var.set(path)
            self.input_image_var.set("") # Clear image if folder is selected

    def browse_shader(self):
        path = filedialog.askopenfilename(
            title="Select Shader Preset",
            filetypes=[("Slangp Presets", "*.slangp"), ("All files", "*.*")]
        )
        if path:
            self.shader_preset_var.set(path)

    def browse_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder_var.set(path)

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # --- Validate Inputs ---
        input_image = self.input_image_var.get()
        input_folder = self.input_folder_var.get()
        shader_preset = self.shader_preset_var.get()
        output_folder = self.output_folder_var.get()

        if not (input_image or input_folder):
            messagebox.showerror("Error", "Please select an input image or an input folder.")
            return
        if not shader_preset:
            messagebox.showerror("Error", "Please select a shader preset file.")
            return
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        images_to_process = []
        if input_image:
            images_to_process.append(Path(input_image))
        else:
            folder_path = Path(input_folder)
            for file in folder_path.iterdir():
                if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    images_to_process.append(file)

        if not images_to_process:
            messagebox.showinfo("Info", "No images found to process in the selected folder.")
            return

        self.log_widget.config(state='normal')
        self.log_widget.delete(1.0, tk.END)
        self.log_widget.config(state='disabled')

        self.is_processing = True
        self.start_button.config(text="Processing...", state='disabled')

        # --- Start Worker Thread ---
        thread = threading.Thread(
            target=self.worker_thread,
            args=(images_to_process, shader_preset, output_folder),
            daemon=True
        )
        thread.start()
        self.process_queue()

    def worker_thread(self, images, shader, output_dir):
        """This function runs in a separate thread to avoid freezing the GUI."""
        try:
            for i, image_path in enumerate(images):
                self.processing_queue.put(f"--- Processing {i+1}/{len(images)}: {image_path.name} ---")

                output_path = Path(output_dir) / f"{image_path.stem}_processed.png"

                # Path to the backend script, assuming it's in the same directory
                backend_script = Path(__file__).parent / "shader_tool.py"

                command = [
                    sys.executable,
                    str(backend_script),
                    str(image_path),
                    str(shader),
                    str(output_path)
                ]

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8'
                )

                # Read output line by line in real-time
                for line in iter(process.stdout.readline, ''):
                    self.processing_queue.put(line.strip())

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    self.processing_queue.put(f"--- Finished processing {image_path.name} ---")
                else:
                    self.processing_queue.put(f"!!! ERROR processing {image_path.name} (exit code: {return_code}) !!!")

            self.processing_queue.put("=== ALL DONE ===")
        except Exception as e:
            self.processing_queue.put(f"!!! A critical error occurred in the worker thread: {e} !!!")
        finally:
            self.processing_queue.put(None) # Signal that processing is finished

    def process_queue(self):
        """Check the queue for messages from the worker thread and update the GUI."""
        try:
            while True:
                message = self.processing_queue.get_nowait()
                if message is None: # End signal
                    self.is_processing = False
                    self.start_button.config(text="Start Processing", state='normal')
                    return
                self.log(message)
        except queue.Empty:
            pass # No new messages

        self.after(100, self.process_queue) # Check again in 100ms

if __name__ == "__main__":
    app = App()
    app.mainloop()
