import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging


class S3Manager:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    def upload_file(self, file_path, bucket_name, object_name=None):
        if object_name is None:
            object_name = file_path

        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name, ExtraArgs={"ContentType": "video/mp4"})
            logging.info(f"File {file_path} uploaded to {bucket_name}/{object_name}")
        except FileNotFoundError:
            logging.error(f"The file {file_path} was not found.")
        except NoCredentialsError:
            logging.error("Credentials not available.")
        except ClientError as e:
            logging.error(f"Client error: {e}")

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.s3_client.download_file(bucket_name, object_name, file_path)
            logging.info(f"File {object_name} downloaded from {bucket_name} to {file_path}")
        except FileNotFoundError:
            logging.error(f"The file path {file_path} was not found.")
        except NoCredentialsError:
            logging.error("Credentials not available.")
        except ClientError as e:
            logging.error(f"Client error: {e}")

    def list_files(self, bucket_name):
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                logging.info(f"No files found in bucket {bucket_name}.")
                return []
        except NoCredentialsError:
            logging.error("Credentials not available.")
            return []
        except ClientError as e:
            logging.error(f"Client error: {e}")
            return []
        
    def delete_file(self, bucket_name, object_name):
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            logging.info(f"File {object_name} deleted from bucket {bucket_name}.")
        except NoCredentialsError:
            logging.error("Credentials not available.")
        except ClientError as e:
            logging.error(f"Client error: {e}")


    def file_exists(self, bucket_name, object_name):
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logging.error(f"Client error: {e}")
                return False
        except NoCredentialsError:
            logging.error("Credentials not available.")
            return False
        
    def check_bucket_exists(self, bucket_name):
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logging.error(f"Client error: {e}")
                return False
        except NoCredentialsError:
            logging.error("Credentials not available.")
            return False
        
    def create_bucket(self, bucket_name):
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            logging.info(f"Bucket {bucket_name} created successfully.")
        except ClientError as e:
            logging.error(f"Client error: {e}")
        except NoCredentialsError:
            logging.error("Credentials not available.")


    def check_and_create_bucket(self, bucket_name):
        if not self.check_bucket_exists(bucket_name):
            logging.info(f"Bucket {bucket_name} does not exist. Creating bucket.")
            self.create_bucket(bucket_name)
        else:
            logging.info(f"Bucket {bucket_name} already exists.")
        

        

    