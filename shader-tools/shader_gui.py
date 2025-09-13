#!/usr/bin/env python3

"""
RetroArch Shader Tool (GUI Frontend)

A user-friendly graphical interface for the shader_tool.py backend.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# --- Constants ---
BACKEND_SCRIPT_NAME = "shader_tool.py"

# --- GUI Class ---

class ShaderApp:
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("RetroArch Shader Tool")
        self.root.geometry("800x600")
        self.root.minsize(600, 450)

        # --- Data ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.backend_path = os.path.join(self.script_dir, BACKEND_SCRIPT_NAME)
        self.processing_thread = None
        self.log_queue = queue.Queue()

        # --- Widgets ---
        self.create_widgets()

        # --- Check for backend script ---
        if not os.path.exists(self.backend_path):
            self.log_message(f"FATAL: Backend script '{BACKEND_SCRIPT_NAME}' not found in the application directory.")
            self.log_message("Please ensure the GUI and backend script are in the same folder.")
            self.start_button.config(state=tk.DISABLED)

        # Start the queue checker
        self.root.after(100, self.process_log_queue)

    def create_widgets(self):
        """Create and lay out all the GUI widgets."""
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid
        main_frame.grid_columnconfigure(1, weight=1)

        # --- Input Configuration ---
        input_frame = tk.LabelFrame(main_frame, text="Input", padx=10, pady=10)
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        input_frame.grid_columnconfigure(1, weight=1)

        # Single Image Input
        self.single_image_path = tk.StringVar()
        tk.Label(input_frame, text="Input Image:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.single_image_path).grid(row=0, column=1, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self.browse_input_image).grid(row=0, column=2, padx=5)

        # Batch Folder Input
        self.batch_folder_path = tk.StringVar()
        tk.Label(input_frame, text="Input Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.batch_folder_path).grid(row=1, column=1, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self.browse_input_folder).grid(row=1, column=2, padx=5)

        # --- Shader and Output Configuration ---
        config_frame = tk.LabelFrame(main_frame, text="Configuration", padx=10, pady=10)
        config_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        config_frame.grid_columnconfigure(1, weight=1)

        # Shader Preset
        self.shader_preset_path = tk.StringVar()
        tk.Label(config_frame, text="Shader Preset (.slangp):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(config_frame, textvariable=self.shader_preset_path).grid(row=0, column=1, sticky="ew")
        tk.Button(config_frame, text="Browse...", command=self.browse_shader_preset).grid(row=0, column=2, padx=5)

        # Output Folder
        self.output_folder_path = tk.StringVar()
        tk.Label(config_frame, text="Output Folder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(config_frame, textvariable=self.output_folder_path).grid(row=1, column=1, sticky="ew")
        tk.Button(config_frame, text="Browse...", command=self.browse_output_folder).grid(row=1, column=2, padx=5)

        # --- Action Button ---
        self.start_button = tk.Button(main_frame, text="Start Processing", command=self.start_processing, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.start_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10, ipady=5)

        # --- Log Viewer ---
        log_frame = tk.LabelFrame(main_frame, text="Log", padx=10, pady=10)
        log_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        main_frame.grid_rowconfigure(3, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD, bg="#2E2E2E", fg="#F1F1F1")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    # --- Browse Dialog Functions ---

    def browse_input_image(self):
        path = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tga"), ("All files", "*.*")])
        if path:
            self.single_image_path.set(path)
            self.batch_folder_path.set("") # Clear batch folder if single image is selected

    def browse_input_folder(self):
        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.batch_folder_path.set(path)
            self.single_image_path.set("") # Clear single image if batch folder is selected

    def browse_shader_preset(self):
        path = filedialog.askopenfilename(title="Select Shader Preset", filetypes=[("Slang Presets", "*.slangp"), ("All files", "*.*")])
        if path:
            self.shader_preset_path.set(path)

    def browse_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder_path.set(path)

    # --- Logging ---

    def log_message(self, message):
        """Append a message to the log text area."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def process_log_queue(self):
        """Check the queue for messages from the worker thread and log them."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message is None: # Sentinel for "done"
                    self.processing_finished()
                else:
                    self.log_message(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    # --- Processing Logic ---

    def start_processing(self):
        """Validate inputs and start the processing thread."""
        # --- Validate Inputs ---
        if not self.single_image_path.get() and not self.batch_folder_path.get():
            messagebox.showerror("Error", "Please select an input image or an input folder.")
            return
        if not self.shader_preset_path.get():
            messagebox.showerror("Error", "Please select a shader preset file.")
            return
        if not self.output_folder_path.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        # --- Start Thread ---
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

    def processing_finished(self):
        """Called when the worker thread is done."""
        self.start_button.config(state=tk.NORMAL, text="Start Processing")
        messagebox.showinfo("Complete", "Processing has finished.")

    def processing_worker(self):
        """
        The worker function that runs in a separate thread.
        Iterates through images and calls the backend script.
        """
        try:
            # Gather paths from thread-safe StringVars
            single_image = self.single_image_path.get()
            batch_folder = self.batch_folder_path.get()
            shader_preset = self.shader_preset_path.get()
            output_folder = self.output_folder_path.get()

            # Determine list of images to process
            images_to_process = []
            if single_image:
                images_to_process.append(single_image)
            else:
                self.log_queue.put(f"Scanning folder: {batch_folder}")
                for filename in os.listdir(batch_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tga')):
                        images_to_process.append(os.path.join(batch_folder, filename))

            if not images_to_process:
                self.log_queue.put("No compatible images found in the selected folder.")
                self.log_queue.put(None) # Signal completion
                return

            # Process each image
            total = len(images_to_process)
            for i, image_path in enumerate(images_to_process):
                self.log_queue.put("-" * 50)
                self.log_queue.put(f"Processing image {i+1}/{total}: {os.path.basename(image_path)}")

                output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)

                command = [
                    sys.executable, # Use the same python interpreter that runs the GUI
                    self.backend_path,
                    image_path,
                    shader_preset,
                    output_path
                ]

                # Run the backend script
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )

                # Read output in real-time
                for line in iter(process.stdout.readline, ''):
                    self.log_queue.put(line.strip())

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    self.log_queue.put(f"Successfully processed: {os.path.basename(image_path)}")
                else:
                    self.log_queue.put(f"ERROR: Backend script failed for {os.path.basename(image_path)} with exit code {return_code}.")

            self.log_queue.put("-" * 50)
            self.log_queue.put("All tasks complete.")

        except Exception as e:
            self.log_queue.put(f"An unexpected error occurred in the GUI worker thread: {e}")
        finally:
            self.log_queue.put(None) # Signal completion


# --- Main Execution ---

def main():
    """Main function to create and run the application."""
    root = tk.Tk()
    app = ShaderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
