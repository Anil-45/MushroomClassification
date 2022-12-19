"""Tuner."""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from ..logger import AppLogger
from ..visualization import plot_confusion_matrix, plot_roc_auc_curve


class ModelFinder:
    """Find the model with best accuracy and AUC score."""

    def __init__(self) -> None:
        """Initialize required variables."""
        self.path = str(os.path.abspath(os.path.dirname(__file__))) + "/../.."
        self.logger = AppLogger().get_logger(f"{self.path}/logs/tuner.log")

    def get_best_params_for_random_forest(
        self, train_x, train_y
    ) -> RandomForestClassifier:
        """Get the parameters for Random Forest which give best accuracy."""
        try:
            # initializing with different combination of parameters
            param_grid = {
                "n_estimators": [10, 50, 100, 130],
                "criterion": ["gini", "entropy"],
                "max_depth": range(2, 4, 1),
                "max_features": ["sqrt", "log2"],
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid=param_grid,
                cv=5,
                verbose=3,
            )

            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            criterion = grid.best_params_["criterion"]
            max_depth = grid.best_params_["max_depth"]
            max_features = grid.best_params_["max_features"]
            n_estimators = grid.best_params_["n_estimators"]

            # creating a new model with the best parameters
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features,
            )

            # training the new model
            clf.fit(train_x, train_y)
            self.logger.info("Random Forest best:%s", str(grid.best_params_))
            return clf

        except Exception as exception:
            self.logger.error("get best params for Random Forest failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_best_params_for_xgboost(self, train_x, train_y) -> XGBClassifier:
        """Get the parameters for XGBoost which give the best accuracy."""
        try:
            # initializing with different combination of parameters
            param_grid_xgboost = {
                "learning_rate": [0.5, 0.1, 0.01, 0.001],
                "max_depth": [3, 5, 10, 20],
                "n_estimators": [10, 50, 100, 200],
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(
                XGBClassifier(objective="binary:logistic"),
                param_grid_xgboost,
                verbose=3,
                cv=5,
            )

            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            learning_rate = grid.best_params_["learning_rate"]
            max_depth = grid.best_params_["max_depth"]
            n_estimators = grid.best_params_["n_estimators"]

            # creating a new model with the best parameters
            xgb = XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )

            # training the new model
            xgb.fit(train_x, train_y)
            self.logger.info("XGBoost best params: %s", str(grid.best_params_))
            return xgb

        except Exception as exception:
            self.logger.error("get best params for XGBoost failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_best_params_for_knn(
        self, train_x, train_y
    ) -> KNeighborsClassifier:
        """Get the parameters for KNN which give the best accuracy."""
        try:
            # initializing with different combination of parameters
            param_grid_knn = {
                "algorithm": ["ball_tree", "kd_tree", "brute"],
                "leaf_size": [10, 17, 24, 28, 30, 35],
                "n_neighbors": [4, 5, 8, 10, 11],
                "p": [1, 2],
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(
                KNeighborsClassifier(), param_grid_knn, verbose=3, cv=5
            )

            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            algorithm = grid.best_params_["algorithm"]
            leaf_size = grid.best_params_["leaf_size"]
            n_neighbors = grid.best_params_["n_neighbors"]
            power_param = grid.best_params_["p"]

            # creating a new model with the best parameters
            knn = KNeighborsClassifier(
                algorithm=algorithm,
                leaf_size=leaf_size,
                n_neighbors=n_neighbors,
                p=power_param,
                n_jobs=-1,
            )

            # training the new model
            knn.fit(train_x, train_y)
            self.logger.info("KNN best params: %s ", str(grid.best_params_))
            return knn

        except Exception as exception:
            self.logger.error("get best params for KNN failed")
            self.logger.exception(exception)
            raise Exception from exception

    def get_model_score(self, model, test_x, test_y, desc: str) -> float:
        """Get model score."""
        try:
            # prediction using the model
            prediction = model.predict(test_x)

            if 1 == len(test_y.unique()):
                # In case of single label, roc_auc_score returns error.
                # We will use accuracy in that case
                score = accuracy_score(test_y, prediction)
                self.logger.info("%s Accuracy:%s", desc, str(score))
            else:
                # prediction probabilities
                prediction_prob = model.predict_proba(test_x)[::, 1]

                score = roc_auc_score(test_y, prediction_prob)
                self.logger.info("AUC for %s:%s", desc, str(score))

                # plot roc auc curve
                plot_roc_auc_curve(
                    test_y,
                    prediction_prob,
                    desc,
                    f"{self.path}/reports/figures/{desc}_roc.png",
                )

                # plot confusion matrix
                plot_confusion_matrix(
                    test_y,
                    prediction,
                    desc,
                    f"{self.path}/reports/figures/{desc}_cm.png",
                    [0, 1],
                )
            return score

        except Exception as exception:
            self.logger.exception(exception)
            return 0

    def get_best_model(self, train_x, train_y, test_x, test_y, desc: str):
        """Get model which has the best AUC score."""
        try:
            # create best model for Random Forest
            model_rf = self.get_best_params_for_random_forest(train_x, train_y)

            # get score
            rf_score = self.get_model_score(
                model=model_rf,
                test_x=test_x,
                test_y=test_y,
                desc=f"RandomForest{desc}",
            )

            # create best model for KNN
            model_knn = self.get_best_params_for_knn(train_x, train_y)

            # get score
            knn_score = self.get_model_score(
                model=model_knn,
                test_x=test_x,
                test_y=test_y,
                desc=f"KNN{desc}",
            )

            # comparing the models
            if rf_score > knn_score:
                model_desc = "RandomForest"
                best_model = model_rf
            else:
                model_desc = "KNN"
                best_model = model_knn
            return model_desc, best_model

        except Exception as exception:
            self.logger.error("get best model failed")
            self.logger.exception(exception)
            raise Exception from exception
