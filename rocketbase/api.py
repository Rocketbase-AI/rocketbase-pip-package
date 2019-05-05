import io
import itertools
import json
import os
import requests

import rocketbase.exceptions

import google
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from google.resumable_media import requests as gRequests
from google.resumable_media import common

class RocketAPI:
    def __init__(self):
        self.models = []
        self.selected_model = {}
        self.credentials_api_url = "https://europe-west1-rockethub.cloudfunctions.net/getUploadCredentials?token=blah"
        self.models_api_url = "https://europe-west1-rockethub.cloudfunctions.net/getAvailableModels"
        self.push_url = "https://europe-west1-rockethub.cloudfunctions.net/saveNewModel"
        self.bucket_name = "rockets-hangar"
        self.project_id = "rockethub"
        self.initialized = False
        self._chunk_size = 512 * 1024

    def get_rocket_info(self, rocket_info: dict):
        """ Get the information about the best Rocket in the API

        Args:
            rocket_info (dict): Basic information about the Rocket to request (e.g. {'username': 'igor', 'modelName': 'retinanet'}). The API will always returned the default model. If one wants a precise model, they should also include the <label> of the Rocket which can be its hash or a user defined string.

        Returns:
            models (list): List of models returned by the API corresponding to the criteria of the rocket_info.
        """
        # Check that at least the builder and the name of the Rocket are in the Rocket info
        if not set(['username', 'modelName']).issubset(rocket_info.keys()):
            raise rocketbase.exceptions.RocketNotEnoughInfo('Please specify the username and the modelName of the Rocket you want to get.')

        payload = {'username': rocket_info['username'], 'modelName': rocket_info['modelName']}

        if 'label' in rocket_info.keys():
            payload['label'] = rocket_info['label']
        
        printable_rocket_name = '\'' + payload['modelName'] + '\'(' +  payload['label'] + ')' if 'label' in payload.keys() else '\'' + payload['modelName']+ '\''
        
        print('Looking for the Rocket ' + printable_rocket_name + ' made by \'' + payload['username'] + '\'...')

        # Make the request (exceptions are catched outside)
        res = requests.get(self.models_api_url, params=payload)

        # if status != 200 then database is broken
        if not res.status_code == 200:
            raise rocketbase.exceptions.RocketAPIError('Database error. Please try again later. error({})'.format(res.status_code))

        models = res.json()

        # Test that the rocket exists
        if not models:
            raise rocketbase.exceptions.RocketNotFound('Rocket cannot be found in our database. Please check the spelling. ' + rocket_info['username'] + '/' + rocket_info['modelName'])
       
        print('{models_len} model versions found from the database.'.format(models_len=len(models)))
        
        return models

    def set_google_cloud_storage_client(self):
        """ Set the Google Cloud Storage Client using the service credentials.

        In order to set the Google Cloud Storage Client that will allow us to upload files to the Google Cloud Storage we will first need to download the service credentials and then use them to create the Cloud Client.

        Raises:
            RequestException: If the request to get the credentials failed in the process.
            CloudStorageCredentials: If the request to get the credentials finished but was not successful.
            CloudStorageRelatedExceptions: If the creation of the Google Cloud Storage Client failed. It can be a lot of different ones.

        """
        try:
            res = requests.get(self.credentials_api_url)
        except requests.exceptions.RequestException as e:  # Catch all the Exceptions relative to the request
            print('Problem retrieving the Cloud Storage Credentials')
            raise e
        else:
            # Raise exception if the retrieval of the credentials failed
            if not res.status_code == 200:
                raise rocketbase.exceptions.CloudStorageCredentials("Retrieving the Cloud Storage Credentials failed: {} \n\n Response message:\n {}".format(res.status_code, res.text))
            
            # Save the credentials on the disk
            with open("gcredentials.json", 'w') as credentials_handle:
                credentials_handle.write(json.dumps((res.json())))
                credentials_handle.close()
            
            # Try to connect to Google Cloud Storage
            try:
                self.storage_client = storage.Client.from_service_account_json("gcredentials.json")
            except Exception as e:
                raise e

            self._transport = AuthorizedSession(
                credentials=self.storage_client._credentials
            )

            # Try to connect to bucket
            try:
                self.bucket = self.storage_client.bucket(self.bucket_name)
            except google.cloud.exceptions.NotFound:
                print("Sorry, that bucket was not found")
            except Exception as e:
                raise e
            
            # If the initialization of the Cloud Storage Client was a success
            self.initialized = True

            # Remove the credentials from the disk
            os.remove("gcredentials.json")

    def push_file_to_rocket_storage(self, source_file_name: str, destination_blob_name: str):
        """Push file to the Cloud Storage for Rockets.

        Args:
            source_filename (str): Name/Path of the file to upload to the Cloud Storage for Rockets.
            destination_blob_name (str): Name of the blob 

        Returns:
            downloadUrl (str): Direct URL to download the upload file.
        """
        # Set the Google Cloud Storage Client to upload the Rocket
        self.set_google_cloud_storage_client()

        #
        blob = self.bucket.blob(destination_blob_name)

        # Uploading the file to the Cloud Storage for Rockets
        print("Please wait, your Rocket is leaving the atmosphere...")
        with open(source_file_name, 'rb') as f:
            blob.upload_from_file(f)

        # Get the url to newly uploaded file
        downloadUrl = blob.public_url

        return downloadUrl

    def push_rocket(self, 
                    rocket_username: str, 
                    rocket_modelName: str, 
                    rocket_hash: str, 
                    rocket_family:str, 
                    trainingDataset: str,
                    isTrainable: bool,
                    rocketRepoUrl: str,
                    paperUrl: str,
                    originRepoUrl:str,
                    description: str,
                    tar_file: str):
        """Push the latest version of a Rocket to the cloud

        Args:
            rocket_username (str): Author of the new Rocket
            rocket_modelName (str): Name of the Model contained in the Rocket
            rocket_hash (str): Version hash of the Rocket
            rocket_family (str): Rocket family this Rocket belongs to
            trainingDataset (str): Dataset name this Rocket was trained on
            isTrainable (str): Flag to indicate whether this Rocket has necessary components for training
            rocketRepoUrl (str): URL of the repository of the Rocket code
            paperUrl (str): URL of the original research publication
            originRepoUrl (str): URL of the original repository of the model
            description (str): Short description of the Rocket and its details
            tar_file (str): Path to the TAR archive of the Rocket
        """
        # Push Rocket to Cloud Storage
        storage_file_path = self.push_file_to_rocket_storage(
                                                source_file_name=tar_file,
                                                destination_blob_name=(rocket_username+'_'+rocket_modelName+'_'+rocket_hash+'.tar')) 

        payload = ({
            'modelName': rocket_modelName,
            'username': rocket_username,
            'family': rocket_family,
            'trainingDataset': trainingDataset,
            'isTrainable': isTrainable,
            'rocketRepoUrl': rocketRepoUrl,
            'paperUrl': paperUrl,
            'originRepoUrl': originRepoUrl,
            'description': description,
            'hash': rocket_hash,
            'downloadUrl': storage_file_path,
        })

        headers = {'Content-type': 'application/json'}

        res = requests.post(self.push_url, json = payload, headers=headers)

        if not res.status_code == 201:
            raise rocketbase.exceptions.RocketAPIError("Push Rocket Update has failed! Status code : {} \n\n Response message:\n {}".format(res.status_code, res.text))
        
        return res.status_code == 201