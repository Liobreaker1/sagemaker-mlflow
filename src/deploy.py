import argparse
import mlflow.sagemaker

def deploy_model(
    model_uri: str,
    endpoint_name: str,
    execution_role: str,
    image_url: str,
    region: str = "us-east-1",
    instance_type: str = "ml.m5.xlarge",
    instance_count: int = 1,
    bucket_name: str = "mlflow-project-artifacts"
):
    config = {
        "execution_role_arn": execution_role,
        "bucket_name": bucket_name,
        "image_url": image_url,
        "region_name": region,
        "archive": False,
        "instance_type": instance_type,
        "instance_count": instance_count,
        "synchronous": True
    }

    print(f"Deploying model: {model_uri} → Endpoint: {endpoint_name}")
    mlflow.sagemaker.deploy(
        app_name=endpoint_name,
        model_uri=model_uri,
        region_name=region,
        mode="create",
        execution_role_arn=execution_role,
        image_url=image_url,
        instance_type=instance_type,
        instance_count=instance_count,
        bucket=bucket_name,
        archive=False,
        synchronous=True
    )
    print(f"Model deployed successfully to endpoint: {endpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, required=True, help="MLflow model URI (e.g., 'runs:/<run_id>/model')")
    parser.add_argument("--endpoint-name", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--execution-role", type=str, required=True, help="IAM role ARN with SageMaker permissions")
    parser.add_argument("--image-url", type=str, required=True, help="Docker image URL for MLflow model")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--bucket-name", type=str, default="mlflow-project-artifacts")

    args = parser.parse_args()

    deploy_model(
        model_uri=args.model_uri,
        endpoint_name=args.endpoint_name,
        execution_role=args.execution_role,
        image_url=args.image_url,
        region=args.region,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        bucket_name=args.bucket_name
    )
