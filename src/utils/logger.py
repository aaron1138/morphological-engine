import json
import os
import datetime
from typing import Dict, Any, List

class RunLogger:
    """
    Logs the parameters of processing jobs to a JSON log file.
    Each log is a JSON object appended to a list in the log file.
    """
    def __init__(self, log_path: str = 'run_log.json'):
        """
        Initializes the RunLogger.

        Args:
            log_path (str): The path to the log file.
        """
        self.log_path = log_path

    def log_run(self, run_data: Dict[str, Any]) -> None:
        """
        Appends a new run entry to the log file.

        The entry includes a timestamp and the data provided by the caller.

        Args:
            run_data (Dict[str, Any]): A dictionary containing the data for the run,
                                      such as settings, shaders, and input files.
        """
        log_entry = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_data': run_data
        }

        log_records: List[Dict[str, Any]] = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    content = f.read()
                    # Handle empty file case
                    if content:
                        log_records = json.loads(content)
                    if not isinstance(log_records, list):
                        print(f"Warning: Log file {self.log_path} is not a list. Starting a new log.")
                        log_records = []
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not decode {self.log_path}. Starting a new log.")
                log_records = []
            except IOError as e:
                print(f"Error reading log file: {e}")
                return # Abort if we can't read the file

        log_records.append(log_entry)

        try:
            with open(self.log_path, 'w') as f:
                json.dump(log_records, f, indent=4)
        except IOError as e:
            print(f"Error writing to log file: {e}")
