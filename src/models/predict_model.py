"""Predict using model."""
import os

import pandas as pd

from ..data import DataLoader, DataValidation, make_dataset
from ..features import Preprocessor
from ..logger import AppLogger
from .utils import Utils

MODE = "test"


class Prediction:
    """Class for Prediction."""

    def __init__(self, path: str) -> None:
        """Initialize required variables."""
        self.base = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        logger_path = f"{self.base}/logs/prediction.log"
        self.logger = AppLogger().get_logger(logger_path)
        if path is not None:
            self.pred_data_val = DataValidation(path, mode=MODE)
            make_dataset(path, MODE)

    def predict(self) -> pd.DataFrame:
        """Predict using saved model."""
        try:
            # deletes the existing prediction file from last run
            self.pred_data_val.delete_prediction_file()
            self.logger.info("start of prediction")
            data = DataLoader("test").get_data()

            # create a copy of actual data to save to prediction file
            data_copy = data.copy()

            preprocessor = Preprocessor(mode=MODE)
            data = preprocessor.remove_columns(data, ["veil-type"])
            data = preprocessor.replace_invalid_val_with_null(data)

            # handle null values if they exist
            is_null_present = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data)

            # convert categorical data to numerical
            encoded_data = preprocessor.encode_categorical_values(data)

            utils = Utils()
            kmeans = utils.load_model("KMeans")

            # predict clusters for data
            clusters = kmeans.predict(encoded_data)

            # append cluster data
            encoded_data["clusters"] = clusters
            data_copy["clusters"] = clusters
            clusters = encoded_data["clusters"].unique()
            path = f"{self.base}/data/processed/test/Predictions.csv"
            header_flag = True

            for i in clusters:
                # grab only corresponding cluster data
                cluster_data = pd.DataFrame(
                    encoded_data[encoded_data["clusters"] == i]
                )
                result = pd.DataFrame(data_copy[data_copy["clusters"] == i])
                result.drop(columns=["clusters"], inplace=True)
                cluster_data.drop(columns=["clusters"], inplace=True)

                # find and load model
                model_name = utils.find_model_file(i)
                self.logger.info("model_name %s", str(model_name))
                model = utils.load_model(model_name)

                # predict using loaded model
                result["Prediction"] = list(model.predict(cluster_data))
                result["Prediction"] = result["Prediction"].astype(int)

                # map predictions back to respective categories
                result["Prediction"] = result["Prediction"].map(
                    {0: "'p'", 1: "'e'"}
                )

                # appends result to prediction file
                result.to_csv(path, header=header_flag, mode="a+", index=False)
                # add headers only first time
                header_flag = False

            data = pd.read_csv(path)
            self.logger.info("End of Prediction")

        except Exception as exception:
            self.logger.error("error occurred while running the prediction")
            self.logger.exception(exception)
            raise Exception from exception

        return data
