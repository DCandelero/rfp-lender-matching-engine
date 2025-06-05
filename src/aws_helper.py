import os
import boto3
import time


def upload_csv_to_s3(file_path, bucket_name, s3_key=None, region='us-east-1'):
    """
    Uploads a CSV file to an S3 bucket for use with AWS Personalize.
    
    Parameters:
        file_path (str): Local path to the CSV file.
        bucket_name (str): Name of your S3 bucket.
        s3_key (str): S3 object key (filename in the bucket). Defaults to the file name.
        region (str): AWS region where the bucket exists.
    
    Returns:
        str: The full S3 path to the uploaded file.
    """
    # Default S3 key to the filename if not provided
    if not s3_key:
        s3_key = os.path.basename(file_path)

    # Create S3 client
    s3 = boto3.client('s3', region_name=region)

    # Upload file
    s3.upload_file(file_path, bucket_name, s3_key)

    s3_path = f"s3://{bucket_name}/{s3_key}"
    print(f"File uploaded to {s3_path}")
    return s3_path

def wait_for_dataset_to_be_ready(personalize, dataset_arn):
    while True:
        response = personalize.describe_dataset(datasetArn=dataset_arn)
        status = response["dataset"]["status"]
        print("Current dataset status:", status)
        if status in ("ACTIVE", "CREATE FAILED"):
            break
        time.sleep(10)
    
    if status == "ACTIVE":
        print(f"✅ Dataset is ready.\ndataset_arn: {dataset_arn}")
    else:
        print("❌ Dataset creation failed.")
        print("Failure reason:", response['dataset'].get('failureReason', 'No reason provided.'))

def wait_for_solution(personalize, solution_arn):
    status = "CREATE PENDING"
    while status != "ACTIVE" and status != "CREATE FAILED":
        response = personalize.describe_solution(solutionArn=solution_arn)
        status = response["solution"]["status"]
        print("Solution status:", status)
        if status in ['ACTIVE', 'CREATE FAILED']:
            break
        time.sleep(30)

    if status == "ACTIVE":
        print(f"✅ Solution is ready. You can now create a solution version.\nsolution_arn: {solution_arn}")
    else:
        print("❌ Solution creation failed.")
        print("Failure reason:", response['solution'].get('failureReason', 'No reason provided.'))

def wait_for_solution_version(personalize, solution_version_arn):
    while True:
        response = personalize.describe_solution_version(solutionVersionArn=solution_version_arn)
        status = response['solutionVersion']['status']
        print("Status:", status)
        if status in ['ACTIVE', 'CREATE FAILED']:
            break
        time.sleep(60)

    if status == "ACTIVE":
        print(f"✅ Solution version is ready. You can now create a campaign.\nsolution_version_arn: {solution_version_arn}")
    else:
        print("❌ Solution version creation failed.")
        print("Failure reason:", response['solutionVersion'].get('failureReason', 'No reason provided.'))