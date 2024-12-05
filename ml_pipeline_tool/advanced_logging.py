import logging
import time
import json
from typing import Dict, Any


class MLPipelineLogger:
    """
    Advanced logging for machine learning pipelines with JSON output and performance tracking.
    """

    def __init__(
        self, log_file: str = "ml_pipeline.log", log_level: int = logging.INFO
    ):
        """
        Initialize the ML pipeline logger.

        Args:
            log_file (str): Path to the log file
            log_level (int): Logging level
        """
        # Configure file logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("MLPipelineLogger")

        # Performance tracking
        self.performance_logs = []

    def log_pipeline_start(self, config: Dict[str, Any]):
        """
        Log the start of a pipeline run.

        Args:
            config (dict): Configuration for the pipeline
        """
        self.logger.info("Pipeline Started: %s", json.dumps(config, indent=2))
        return time.time()

    def log_pipeline_end(self, start_time: float, results: Dict[str, Any]):
        """
        Log the end of a pipeline run and compute runtime.

        Args:
            start_time (float): Start time of the pipeline
            results (dict): Performance results
        """
        end_time = time.time()
        runtime = end_time - start_time

        log_entry = {
            "runtime": runtime,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.performance_logs.append(log_entry)

        self.logger.info("Pipeline Completed: %s", json.dumps(log_entry, indent=2))
        return log_entry

    def log_error(self, error_message: str, exception: Exception = None):
        """
        Log an error with optional exception details.

        Args:
            error_message (str): Error message
            exception (Exception, optional): Exception object
        """
        self.logger.error(error_message)
        if exception:
            self.logger.exception(exception)

    def save_performance_log(self, file_path: str = "performance_log.json"):
        """
        Save performance logs to a JSON file.

        Args:
            file_path (str): Path to save the performance log
        """
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(self.performance_logs, f, indent=2)