import os, sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Train test split initiated")
            X_train, X_test, y_train, y_test = (

                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
                )

            models = {

                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "K-Neighbor Classifier": KNeighborsClassifier()
                }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found ")
            logging.info("Best model found on both training and test set")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            print(f"Model used : {best_model_name}")
            return acc_score

        except Exception as e:
            raise CustomException(e, sys)
