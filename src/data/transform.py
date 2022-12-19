"""Transform data."""

import os

import pandas as pd

from ..logger import AppLogger


class DataTransform:
    """Transform accepted data before loading it in Database."""

    def __init__(self, mode: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."

        if "train" == str(mode):
            self.accepted_dir = f"{path}/data/interim/train/accepted/"
            self.mode = "train"
            self.log_file = f"{path}/logs/train_transform.log"
        else:
            self.accepted_dir = f"{path}/data/interim/test/accepted/"
            self.mode = "test"
            self.log_file = f"{path}/logs/test_transform.log"

        self.logger = AppLogger().get_logger(self.log_file)

    def add_quote_to_string(self) -> None:
        """Convert "?" to "'?'"."""
        try:
            files = os.listdir(self.accepted_dir)
            for file in files:
                file_path = str(self.accepted_dir) + str(file)
                data = pd.read_csv(file_path)

                for column in data.columns:
                    data[column] = data[column].replace("?", "'?'")

                data.to_csv(file_path, index=None, header=True)
                self.logger.info(" %s: file transformed successfully", file)

        except Exception as exception:
            self.logger.error("%s transform failed", self.mode)
            self.logger.exception(exception)
            raise Exception from exception
