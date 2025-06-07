import os
import json
import boto3
import argparse
from src.data import test


def prepare_payload(data, n_instances=20):
    array = data[:n_instances]
    if hasattr(array, "toarray"):
        array = array.toarray()
    return json.dumps({'instances': array[:, :-1].tolist()})


def invoke_sagemaker_endpoint(endpoint_name, payload, region='us-east-1'):
    client = boto3.client('runtime.sagemaker', region_name=region)
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,
            ContentType='application/json'
        )
        return response['Body'].read().decode("utf-8")
    except Exception as e:
        print(f"Error calling endpoint {endpoint_name}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="prod-endpoint", help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    args = parser.parse_args()

    payload = prepare_payload(test)
    prediction = invoke_sagemaker_endpoint(args.endpoint, payload, region=args.region)

    if prediction:
        try:
            parsed = json.loads(prediction)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print("Raw prediction output:")
            print(prediction)
