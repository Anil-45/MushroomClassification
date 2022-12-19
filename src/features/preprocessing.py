"""Preprocessing."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer

from ..logger import AppLogger


class Preprocessor:
    """Class to clean and transform the data before training."""

    def __init__(self, mode: str) -> None:
        """Initialize required variables."""
        path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        self.logger = AppLogger().get_logger(f"{path}/logs/preprocessing.log")
        self.col_with_zero_std = []
        self.mode = mode

        if "train" == str(mode):
            self.processed_path = f"{path}/data/processed/train/"
        else:
            self.processed_path = f"{path}/data/processed/test/"

    def remove_columns(self, data: DataFrame, columns: list) -> DataFrame:
        """Remove the given columns from a pandas dataframe."""
        try:
            useful_data = data.drop(columns=columns)
            self.logger.info("columns removal success")
            return useful_data

        except Exception as exception:
            self.logger.error("remove columns failed")
            self.logger.exception(exception)
            raise Exception from exception

    def separate_label_feature(
        self, data: DataFrame, label_column_name: str
    ) -> Tuple[DataFrame, Series]:
        """Separate the features and label columns."""
        try:
            # drop the columns specified and separate the feature columns
            features = data.drop(columns=[label_column_name])
            # Filter the Label column
            labels = data[label_column_name]
            self.logger.info("label separation Successful")
            return features, labels

        except Exception as exception:
            self.logger.error("separate label features failed")
            self.logger.exception(exception)
            raise Exception from exception

    def is_null_present(self, data: DataFrame) -> bool:
        """Check the presence of null values."""
        try:
            null_present = False
            null_counts = data.isna().sum()
            for null_count in null_counts:
                if null_count > 0:
                    null_present = True
                    break

            return null_present

        except Exception as exception:
            self.logger.error("null check failed")
            self.logger.exception(exception)
            raise Exception from exception

    def replace_invalid_val_with_null(self, data: DataFrame) -> DataFrame:
        """Replace invalid values like '?' with null."""
        for column in data.columns:
            data[column] = data[column].replace("'?'", np.nan)
        return data

    def encode_categorical_values(self, data: DataFrame) -> DataFrame:
        """Encode all the categorical values."""
        columns = list(data.columns)
        # For training data
        if "class" in columns:
            data["class"] = data["class"].map({"'p'": 0, "'e'": 1})
            columns.remove("class")

        for column in columns:
            data = pd.get_dummies(data, columns=[column], drop_first=True)
        return data

    def impute_missing_values(self, data: DataFrame) -> DataFrame:
        """Replace all the missing values using KNN Imputer."""
        try:
            null_counts = data.isna().sum()
            cols_with_null = []
            for idx, col in enumerate(data.columns):
                if null_counts[idx] > 0:
                    cols_with_null.append(col)

            imputer = SimpleImputer(
                missing_values=np.nan, strategy="most_frequent"
            )
            for col in cols_with_null:
                # impute the missing values
                data[col] = imputer.fit_transform(
                    data[col].values.reshape(-1, 1)
                )[:, 0]

            self.logger.info("Imputing missing values successful")
            return data

        except Exception as exception:
            self.logger.error("impute missing values failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_cols_with_zero_std_dev(self, data: DataFrame) -> list:
        """Find out the columns which have a standard deviation of zero.

        Columns with zero standard deviation are found only during training,
        during testing this columns will directly be returned without any
        evaluation of test data.
        """
        data_description = data.describe()

        if "train" != str(self.mode):
            return self.col_with_zero_std

        try:
            for col in data.columns:
                # check if standard deviation is zero
                if 0 == data_description[col]["std"]:
                    self.col_with_zero_std.append(col)
            return self.col_with_zero_std

        except Exception as exception:
            self.logger.error("get columns with zero std failed")
            self.logger.exception(exception)
            raise Exception from exception
