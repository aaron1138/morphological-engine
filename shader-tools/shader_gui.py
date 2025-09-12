import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess
import threading
import os

class ShaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shader Tool GUI")

        # Frame for input widgets
        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(padx=10, pady=10)

        # Input Image
        self.input_image_label = tk.Label(frame, text="Input Image:")
        self.input_image_label.grid(row=0, column=0, sticky="w")
        self.input_image_path = tk.StringVar()
        self.input_image_entry = tk.Entry(frame, textvariable=self.input_image_path, width=50)
        self.input_image_entry.grid(row=0, column=1, padx=5)
        self.input_image_button = tk.Button(frame, text="Browse...", command=self.browse_input_image)
        self.input_image_button.grid(row=0, column=2)

        # Input Folder
        self.input_folder_label = tk.Label(frame, text="Input Folder:")
        self.input_folder_label.grid(row=1, column=0, sticky="w")
        self.input_folder_path = tk.StringVar()
        self.input_folder_entry = tk.Entry(frame, textvariable=self.input_folder_path, width=50)
        self.input_folder_entry.grid(row=1, column=1, padx=5)
        self.input_folder_button = tk.Button(frame, text="Browse...", command=self.browse_input_folder)
        self.input_folder_button.grid(row=1, column=2)

        # Shader Preset
        self.shader_preset_label = tk.Label(frame, text="Shader Preset:")
        self.shader_preset_label.grid(row=2, column=0, sticky="w")
        self.shader_preset_path = tk.StringVar()
        self.shader_preset_entry = tk.Entry(frame, textvariable=self.shader_preset_path, width=50)
        self.shader_preset_entry.grid(row=2, column=1, padx=5)
        self.shader_preset_button = tk.Button(frame, text="Browse...", command=self.browse_shader_preset)
        self.shader_preset_button.grid(row=2, column=2)

        # Output Folder
        self.output_folder_label = tk.Label(frame, text="Output Folder:")
        self.output_folder_label.grid(row=3, column=0, sticky="w")
        self.output_folder_path = tk.StringVar()
        self.output_folder_entry = tk.Entry(frame, textvariable=self.output_folder_path, width=50)
        self.output_folder_entry.grid(row=3, column=1, padx=5)
        self.output_folder_button = tk.Button(frame, text="Browse...", command=self.browse_output_folder)
        self.output_folder_button.grid(row=3, column=2)

        # Start Button
        self.start_button = tk.Button(root, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=10)

        # Status Log
        self.log_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, width=80)
        self.log_area.pack(padx=10, pady=10)
        self.log_area.config(state=tk.DISABLED)

    def browse_input_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.input_image_path.set(path)

    def browse_input_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.input_folder_path.set(path)

    def browse_shader_preset(self):
        path = filedialog.askopenfilename(filetypes=[("Slangp files", "*.slangp")])
        if path:
            self.shader_preset_path.set(path)

    def browse_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_folder_path.set(path)

    def start_processing(self):
        shader_preset = self.shader_preset_path.get()
        output_folder = self.output_folder_path.get()
        input_image = self.input_image_path.get()
        input_folder = self.input_folder_path.get()

        if not shader_preset or not output_folder:
            self.log_message("Error: Shader Preset and Output Folder are required.")
            return

        if not input_image and not input_folder:
            self.log_message("Error: Please select an Input Image or Input Folder.")
            return

        images_to_process = []
        if input_image:
            images_to_process.append(input_image)
        if input_folder:
            for f in os.listdir(input_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    images_to_process.append(os.path.join(input_folder, f))

        if not images_to_process:
            self.log_message("No images found to process.")
            return

        self.start_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._run_processing, args=(images_to_process, shader_preset, output_folder))
        thread.start()

    def _run_processing(self, images, shader_preset, output_folder):
        for image_path in images:
            self.log_message(f"Processing {os.path.basename(image_path)}...")
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)

            command = ["python", "shader_tool.py", image_path, shader_preset, output_path]

            try:
                process = subprocess.Popen(command, cwd="shader-tools", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

                for line in iter(process.stdout.readline, ''):
                    self.log_message(line.strip())
                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    self.log_message(f"Finished processing {os.path.basename(image_path)}.")
                else:
                    self.log_message(f"Error processing {os.path.basename(image_path)}. See log for details.")

            except Exception as e:
                self.log_message(f"An error occurred: {e}")

        self.root.after(0, self.processing_finished)

    def processing_finished(self):
        self.log_message("All processing finished.")
        self.start_button.config(state=tk.NORMAL)

    def log_message(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ShaderGUI(root)
    root.mainloop()
