import os
import shutil
import boto3
from botocore.exceptions import NoCredentialsError

def create_dataset_directory():
    # Create a directory called 'dataset'
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    print("Directory 'dataset' created or already exists.")

def download_files_from_s3(bucket_name, local_directory='dataset'):
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')

    try:
        # Get a list of files in the specified S3 bucket
        response = s3.list_objects_v2(Bucket=bucket_name)

        # Check if the bucket is empty
        if 'Contents' not in response:
            print(f"No files found in bucket {bucket_name}.")
            return

        # Download each file
        for obj in response['Contents']:
            file_name = obj['Key']
            local_file_path = os.path.join(local_directory, file_name)

            # Create subdirectories if necessary
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file from S3
            s3.download_file(bucket_name, file_name, local_file_path)
            print(f"Downloaded {file_name} to {local_file_path}")

    except NoCredentialsError:
        print("Credentials not available.")

if __name__ == '__main__':
    bucket_name = 'your-s3-bucket-name'
    create_dataset_directory()
    download_files_from_s3(bucket_name)
