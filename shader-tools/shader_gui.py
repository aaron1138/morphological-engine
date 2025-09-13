import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
import queue
import os
import sys

class ShaderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shader Tool GUI")
        self.geometry("800x600")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.image_path = tk.StringVar()
        self.image_folder_path = tk.StringVar()
        self.shader_path = tk.StringVar()
        self.output_folder_path = tk.StringVar()

        self.create_widgets()
        self.log_queue = queue.Queue()
        self.after(100, self.process_log_queue)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Path Selection Frames ---
        path_frame = ttk.LabelFrame(main_frame, text="Paths", padding="10")
        path_frame.pack(fill=tk.X, pady=5)
        path_frame.grid_columnconfigure(1, weight=1)

        # Input Image
        ttk.Label(path_frame, text="Input Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.image_path).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_image).grid(row=0, column=2, padx=5)

        # Input Folder
        ttk.Label(path_frame, text="Input Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.image_folder_path).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_image_folder).grid(row=1, column=2, padx=5)

        # Shader Preset
        ttk.Label(path_frame, text="Shader Preset:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.shader_path).grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_shader).grid(row=2, column=2, padx=5)

        # Output Folder
        ttk.Label(path_frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(path_frame, textvariable=self.output_folder_path).grid(row=3, column=1, sticky=tk.EW, padx=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_output_folder).grid(row=3, column=2, padx=5)

        # --- Controls ---
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=10)

        # --- Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, bg="#2b2b2b", fg="white")
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def browse_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path.set(path)

    def browse_image_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.image_folder_path.set(path)

    def browse_shader(self):
        path = filedialog.askopenfilename(filetypes=[("Slangp Presets", "*.slangp")])
        if path:
            self.shader_path.set(path)

    def browse_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_folder_path.set(path)

    def log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        self.after(100, self.process_log_queue)

    def start_processing(self):
        shader = self.shader_path.get()
        output_folder = self.output_folder_path.get()

        # --- Validation ---
        if not shader or not output_folder:
            self.log("Error: Shader Preset and Output Folder must be selected.")
            return

        images_to_process = []
        single_image = self.image_path.get()
        batch_folder = self.image_folder_path.get()

        if single_image:
            images_to_process.append(single_image)
        elif batch_folder:
            for f in os.listdir(batch_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tga')):
                    images_to_process.append(os.path.join(batch_folder, f))

        if not images_to_process:
            self.log("Error: No input images selected. Choose an Input Image or Input Folder.")
            return

        self.start_button.config(state='disabled')
        self.log_area.config(state='normal')
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state='disabled')

        thread = threading.Thread(
            target=self._processing_thread,
            args=(images_to_process, shader, output_folder),
            daemon=True
        )
        thread.start()

    def _processing_thread(self, images, shader, output_folder):
        self.log("--- Starting Batch Process ---")

        # Get the absolute path to the backend script
        backend_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shader_tool.py")

        for image_in in images:
            try:
                self.log(f"\nProcessing: {os.path.basename(image_in)}")

                output_filename = os.path.basename(image_in)
                base, _ = os.path.splitext(output_filename)
                image_out = os.path.join(output_folder, base + ".png")

                # Environment setup for dependencies
                env = os.environ.copy()
                # This is a bit of a hack for the sandbox, to ensure our mock slangc is found
                mock_bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_bin")
                env["PATH"] = mock_bin_path + os.pathsep + env["PATH"]
                env["GLCONTEXT_BACKEND"] = "osmesa" # Attempt to force headless

                command = [sys.executable, backend_script, image_in, shader, image_out]

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )

                for line in iter(process.stdout.readline, ''):
                    self.log(line.strip())

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    self.log(f"Successfully processed and saved to {image_out}")
                else:
                    self.log(f"Error processing {os.path.basename(image_in)}. See log for details.")

            except Exception as e:
                self.log(f"An unexpected error occurred: {e}")

        self.log("\n--- Batch Process Finished ---")
        self.after(0, self.enable_start_button)

    def enable_start_button(self):
        self.start_button.config(state='normal')

if __name__ == "__main__":
    app = ShaderGUI()
    app.mainloop()
