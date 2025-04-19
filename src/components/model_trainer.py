import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Train Test SPlit Input Data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbours" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostregressor" : CatBoostRegressor(verbose=False),
                "Adaboost Regressor" : AdaBoostRegressor()
            }

            param_grid = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },
    
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },
    
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 10],
                    "subsample": [0.8, 1.0],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },
    
                "K-Neighbours": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
    
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 10],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 1, 5],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 1.5, 2]
                },
    
                "CatBoostregressor": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "depth": [4, 6, 10],
                    "l2_leaf_reg": [1, 3, 5],
                    "bagging_temperature": [0.1, 0.5, 1.0]
                },
    
                "Adaboost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }

            
            
            
            model_report : dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,
                                               params=param_grid)
            
            ## to get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            ## to get the best model name from the dictionary
            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best Found Model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_sco = r2_score(y_test,predicted)
            
            return r2_sco
            
            
        except Exception as e:
            raise CustomException(e,sys)
        