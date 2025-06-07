import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import mlflow
import numpy as np
from src.data import X_train, X_val, y_train, y_val
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from src.params import param_grids
from src.utils import eval_metrics

class ModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.param_grid = param_grids.get(self.model_name)

        if self.param_grid is None:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(param_grids.keys())}")

        self.model_class = self._get_model_class()

    def _get_model_class(self):
        if self.model_name == "ridge":
            return Ridge
        elif self.model_name == "elasticnet":
            return ElasticNet
        elif self.model_name == "xgboost":
            return XGBRegressor

    def train_and_log(self):
        for params in ParameterGrid(self.param_grid):
            print(f"Starting run with params: {params}")
            with mlflow.start_run(nested=True):
                mlflow.set_tag("model", self.model_name)
                model = self.model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                metrics = eval_metrics(y_val, y_pred)

                # Log input data
                mlflow.log_input(
                    mlflow.data.from_numpy(X_train.toarray() if hasattr(X_train, "toarray") else X_train),
                    context='Training dataset'
                )
                mlflow.log_input(
                    mlflow.data.from_numpy(X_val.toarray() if hasattr(X_val, "toarray") else X_val),
                    context='Validation dataset'
                )

                # Log parameters and metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    self.model_name,
                    input_example=X_train,
                    code_paths=[
                        "src/train.py",
                        "src/data.py",
                        "src/params.py",
                        "src/utils.py"
                    ]
                )

            print(f"Run logged → model: {self.model_name} | params: {params} | RMSE: {metrics['RMSE']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="elasticnet", help="Model to train")
    parser.add_argument("--experiment", type=str, default="housing-price-exp", help="MLflow experiment name")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(args.experiment)

    trainer = ModelTrainer(model_name=args.model)
    trainer.train_and_log()
