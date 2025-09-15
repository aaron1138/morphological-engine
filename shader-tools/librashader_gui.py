import tkinter as tk
from tkinter import ttk, filedialog
import configparser
import os
import threading
import subprocess
import concurrent.futures
from pathlib import Path

class DispatcherThread(threading.Thread):
    def __init__(self, settings, stop_event, log_callback):
        super().__init__()
        self.settings = settings
        self.stop_event = stop_event
        self.log = log_callback

    def run(self):
        try:
            self.log("Dispatcher started.")

            cli_path = Path(self.settings['cli_path'])
            input_path = Path(self.settings['input_path'])
            shader_path = Path(self.settings['shader_preset_path'])
            output_path = Path(self.settings['output_path'])
            thread_count = self.settings['thread_count']

            if not cli_path.is_file():
                self.log(f"ERROR: librashader-cli.exe not found at '{cli_path}'")
                return
            if not input_path.exists():
                self.log(f"ERROR: Input path not found at '{input_path}'")
                return
            if not shader_path.is_file():
                self.log(f"ERROR: Shader preset not found at '{shader_path}'")
                return
            if not output_path.is_dir():
                self.log(f"ERROR: Output folder not found at '{output_path}'")
                return

            image_files = []
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            if input_path.is_dir():
                self.log(f"Scanning for images in '{input_path}'...")
                for ext in image_extensions:
                    image_files.extend(input_path.glob(f"*{ext}"))
            elif input_path.is_file() and input_path.suffix.lower() in image_extensions:
                image_files.append(input_path)

            if not image_files:
                self.log("No image files found to process.")
                return

            self.log(f"Found {len(image_files)} image(s) to process.")

            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = {executor.submit(self.process_image, image_file): image_file for image_file in image_files}

                for future in concurrent.futures.as_completed(futures):
                    if self.stop_event.is_set():
                        self.log("Stop signal received. Halting dispatch of new tasks.")
                        break
                    try:
                        future.result()
                    except Exception as e:
                        self.log(f"ERROR: A task failed with an exception: {e}")

        except Exception as e:
            self.log(f"FATAL: An unexpected error occurred in the dispatcher: {e}")
        finally:
            self.log("Dispatcher finished.")

    def process_image(self, image_path):
        if self.stop_event.is_set():
            return

        cli_path = self.settings['cli_path']
        shader_path = self.settings['shader_preset_path']
        output_path = Path(self.settings['output_path'])
        output_file = output_path / image_path.name

        self.log(f"Processing '{image_path.name}'...")

        command = [
            str(cli_path), "render",
            "--preset", str(shader_path),
            "--image", str(image_path),
            "--out", str(output_file),
            "--runtime", "d3d12"
        ]

        try:
            self.log(f"  -> EXECUTING: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.log(f"Successfully processed '{image_path.name}'.")
            if result.stdout: self.log(f"  -> stdout: {result.stdout.strip()}")
            if result.stderr: self.log(f"  -> stderr: {result.stderr.strip()}")
        except FileNotFoundError:
            self.log(f"ERROR: Command not found: '{cli_path}'. Make sure the path is correct.")
        except subprocess.CalledProcessError as e:
            self.log(f"ERROR processing '{image_path.name}':")
            self.log(f"  -> Return Code: {e.returncode}")
            if e.stdout: self.log(f"  -> stdout: {e.stdout.strip()}")
            if e.stderr: self.log(f"  -> stderr: {e.stderr.strip()}")
        except Exception as e:
            self.log(f"An unexpected error occurred while processing '{image_path.name}': {e}")


class LibraShaderGUI(tk.Tk):
    CONFIG_FILE = "librashader_gui.ini"

    def __init__(self):
        super().__init__()
        self.title("LibraShader GUI")
        self.geometry("800x680") # Increased height for new widgets
        self.dispatcher_thread = None
        self.stop_event = threading.Event()

        self.cli_path = tk.StringVar()
        self.input_path = tk.StringVar()
        self.shader_preset_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.thread_count = tk.IntVar(value=4)
        self.input_mode = tk.StringVar(value="Folder") # Default to Folder mode

        self.create_widgets()
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        settings_frame.columnconfigure(1, weight=1)

        def create_path_row(parent, label_text, string_var, row, command):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(parent, textvariable=string_var).grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
            ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=3, sticky=tk.E, padx=5, pady=2)

        create_path_row(settings_frame, "librashader-cli.exe:", self.cli_path, 0, self.browse_cli_path)

        # Input Path Row with Radio buttons
        ttk.Label(settings_frame, text="Input Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        input_entry = ttk.Entry(settings_frame, textvariable=self.input_path)
        input_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_input_path).grid(row=1, column=3, sticky=tk.E, padx=5, pady=2)

        input_mode_frame = ttk.Frame(settings_frame)
        input_mode_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Radiobutton(input_mode_frame, text="Folder", variable=self.input_mode, value="Folder").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_mode_frame, text="File", variable=self.input_mode, value="File").pack(side=tk.LEFT, padx=5)

        create_path_row(settings_frame, "Shader Preset:", self.shader_preset_path, 3, self.browse_shader_path)
        create_path_row(settings_frame, "Output Folder:", self.output_path, 4, self.browse_output_path)

        ttk.Label(settings_frame, text="Number of Threads:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=64, textvariable=self.thread_count, width=5).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

    def browse_cli_path(self):
        path = filedialog.askopenfilename(title="Select librashader-cli.exe", filetypes=[("Executable files", "*.exe"), ("All files", "*.*")])
        if path: self.cli_path.set(path)

    def browse_input_path(self):
        if self.input_mode.get() == "Folder":
            path = filedialog.askdirectory(title="Select Input Folder", mustexist=True)
        else:
            path = filedialog.askopenfilename(title="Select Input File", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path: self.input_path.set(path)

    def browse_shader_path(self):
        path = filedialog.askopenfilename(title="Select Shader Preset", filetypes=[("Slang Presets", "*.slangp"), ("All files", "*.*")])
        if path: self.shader_preset_path.set(path)

    def browse_output_path(self):
        path = filedialog.askdirectory(title="Select Output Folder", mustexist=True)
        if path: self.output_path.set(path)

    def save_settings(self):
        config = configparser.ConfigParser()
        config['Settings'] = {
            'cli_path': self.cli_path.get(),
            'input_path': self.input_path.get(),
            'input_mode': self.input_mode.get(),
            'shader_preset_path': self.shader_preset_path.get(),
            'output_path': self.output_path.get(),
            'thread_count': self.thread_count.get()
        }
        with open(self.CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        self.log("Settings saved.")

    def load_settings(self):
        if not os.path.exists(self.CONFIG_FILE):
            self.log("No config file found. Starting with default settings.")
            return
        config = configparser.ConfigParser()
        config.read(self.CONFIG_FILE)
        if 'Settings' in config:
            settings = config['Settings']
            self.cli_path.set(settings.get('cli_path', ''))
            self.input_path.set(settings.get('input_path', ''))
            self.input_mode.set(settings.get('input_mode', 'Folder'))
            self.shader_preset_path.set(settings.get('shader_preset_path', ''))
            self.output_path.set(settings.get('output_path', ''))
            self.thread_count.set(settings.getint('thread_count', 1))
            self.log("Settings loaded.")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_processing(self):
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.log("Dispatcher is already running.")
            return

        self.save_settings()
        self.stop_event.clear()

        settings = {
            'cli_path': self.cli_path.get(),
            'input_path': self.input_path.get(),
            'shader_preset_path': self.shader_preset_path.get(),
            'output_path': self.output_path.get(),
            'thread_count': self.thread_count.get()
        }

        self.dispatcher_thread = DispatcherThread(settings, self.stop_event, self.log)
        self.dispatcher_thread.start()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.check_dispatcher_thread()

    def stop_processing(self):
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.log("Sending stop signal to dispatcher...")
            self.stop_event.set()
        self.stop_button.config(state=tk.DISABLED)

    def check_dispatcher_thread(self):
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.after(100, self.check_dispatcher_thread)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.log("Processing finished or was stopped.")

    def on_close(self):
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.log("Stopping dispatcher before closing...")
            self.stop_event.set()
            self.dispatcher_thread.join()
        self.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = LibraShaderGUI()
    app.mainloop()
