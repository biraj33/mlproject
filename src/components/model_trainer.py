import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainierConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainierConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            logging.info("Data  is splited")

            models = {
                'Random Forest': LinearRegression(),
                'Desision Tree': AdaBoostRegressor(),
                'Gradient Bossting': RandomForestRegressor(),
                'Linaer Regression': DecisionTreeRegressor(),
                'k-Neighbors Classifier': KNeighborsRegressor(),
                'XGBClassifier': GradientBoostingRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(verbose=False),
                "Adaboost Classifier": AdaBoostRegressor()
            }
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test= y_test, models = models)

            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info("best model is found in both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            return(r2_score(y_test, predicted))


        except Exception as e:
            raise CustomException(e, sys)
            

