import mlflow
import argparse

def main(model_name: str, experiment_name: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    print(f"Launching MLflow Project with model: {model_name}")

    submitted_run = mlflow.projects.run(
        uri=".",
        entry_point="train",
        experiment_name=experiment_name,
        parameters={"model": model_name},
        env_manager="local"
    )

    print(f"Run submitted: {submitted_run.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="elasticnet", help="Model name: ridge | elasticnet | xgboost")
    parser.add_argument("--experiment", type=str, default="housing-price-exp", help="MLflow experiment name")
    args = parser.parse_args()

    main(args.model, args.experiment)
