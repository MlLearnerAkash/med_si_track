import os
import boto3
from botocore.exceptions import NoCredentialsError

def create_dataset_directory():
    # Create a directory called 'dataset'
    if not os.path.exists('/mnt/data/dataset/s3_data'):
        os.makedirs('/mnt/data/dataset/s3_data')
    print("Directory 'dataset' created or already exists.")

def download_files_from_s3(bucket_name, local_directory='/mnt/data/dataset/s3_data'):
    # Initialize a session using Amazon S3
    s3 = boto3.client('s3')

    try:
        # Pagination logic for listing all objects in the S3 bucket
        continuation_token = None
        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=bucket_name)

            # Check if the bucket is empty
            if 'Contents' not in response:
                print(f"No files found in bucket {bucket_name}.")
                return

            # Download each file or directory
            for obj in response['Contents']:
                file_name = obj['Key']
                local_file_path = os.path.join(local_directory, file_name)

                # Create subdirectories if necessary
                if file_name.endswith('/'):  # This is a directory
                    os.makedirs(local_file_path, exist_ok=True)
                    print(f"Directory {file_name} created.")
                else:
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # Download the file from S3
                    s3.download_file(bucket_name, file_name, local_file_path)
                    print(f"Downloaded {file_name} to {local_file_path}")

            # Check if more objects are available for listing
            if response.get('IsTruncated'):  # More pages to fetch
                continuation_token = response.get('NextContinuationToken')
            else:
                break  # All objects have been processed

    except NoCredentialsError:
        print("Credentials not available.")

if __name__ == '__main__':
    bucket_name = 'opervubucket'
    create_dataset_directory()
    download_files_from_s3(bucket_name)
