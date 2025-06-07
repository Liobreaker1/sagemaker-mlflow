import mlflow

experiments = mlflow.search_experiments()

for exp in experiments:
    if exp.name != "Default":
        print(f"Deleting experiment: {exp.name} (ID: {exp.experiment_id})")
        mlflow.delete_experiment(exp.experiment_id)
