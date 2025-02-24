import os
import subprocess
import json
import logging
import sys
from pathlib import Path
from tkinter import filedialog, Tk
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    last_database: str
    last_folder: str
    folder_path: str

class ConfigManager:
    def __init__(self, config_file: Path = Path(__file__).parent / 'app_config.json'):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    def load(self) -> Dict:
        """Load configuration from JSON file with error handling."""
        try:
            if self.config_file.exists():
                content = self.config_file.read_text().strip()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error reading config file: {e}")
        return {}

    def save(self, config: Dict) -> None:
        """Save configuration to JSON file."""
        try:
            self.config_file.write_text(json.dumps(config, indent=4))
            self.logger.info(f"Config updated: {config}")
        except Exception as e:
            self.logger.error(f"Error saving config file: {e}")
            raise

class ScriptProcessor:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.logger = logging.getLogger(__name__)

    def run_script(self, script_name: str, folder_path: str) -> None:
        """Run a single script with error handling and logging."""
        script_path = self.base_dir / script_name
        self.logger.info(f"Running script: {script_path}")

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path), folder_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self._process_output_stream(process.stdout, "INFO")
            self._process_output_stream(process.stderr, "ERROR")

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, f"Script {script_name} failed"
                )

        except FileNotFoundError:
            self.logger.error(f"Script not found: {script_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error running {script_name}: {str(e)}")
            raise

    def _process_output_stream(self, stream, level: str) -> None:
        """Process output stream from subprocess."""
        if stream:
            for line in stream:
                line = line.strip()
                if level == "ERROR":
                    self.logger.error(line)
                    print(f"Error: {line}", file=sys.stderr)
                else:
                    self.logger.info(line)
                    print(line)

class MonthlyDataProcessor:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.script_processor = ScriptProcessor()
        self.logger = logging.getLogger(__name__)

        # Configure logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = Path(__file__).parent / 'monthly_processor.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def process_monthly_data(self, folder_path: str) -> None:
        """Process monthly data for the specified folder."""
        scripts = [
            'Monthly_Bilateral_Collate.py',
            # Uncomment to include more scripts
            'Monthly_LR.py',
            'Monthly_Settlement.py',
            'Monthly_Summary.py',
            'Monthly_Summary2.py'
        ]

        for script in scripts:
            try:
                self.script_processor.run_script(script, folder_path)
            except Exception as e:
                self.logger.error(f"Failed to process {script}: {e}")
                raise

    def select_folder(self) -> Optional[str]:
        """Show folder selection dialog and return selected path."""
        root = Tk()
        root.withdraw()

        config = self.config_manager.load()
        initial_dir = config.get('last_folder', str(Path.home()))

        folder_path = filedialog.askdirectory(
            title="Select the folder containing monthly data",
            initialdir=initial_dir
        )
        root.destroy()

        if folder_path:
            config['last_folder'] = folder_path
            self.config_manager.save(config)
            return folder_path
        return None


def main():
    os.environ["QT_LOGGING_RULES"] = "*.debug=false"
    processor = MonthlyDataProcessor()

    try:
        folder_path = processor.select_folder()
        if folder_path:
            processor.process_monthly_data(folder_path)
        else:
            print("No folder selected. Exiting.")
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
