import io
import itertools
import json
import os
import requests

import google
from google.cloud import storage
from google.resumable_media import requests as gRequests
from google.resumable_media import common
from google.auth.transport.requests import AuthorizedSession

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

    def get_service_credentials(self):
        """ Fetch Service Credentials to allow Rocket launch

        """
        res = requests.get(self.credentials_api_url)
        if res.status_code == 200:
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
                self.initialized = True
                os.remove("gcredentials.json")


    def get_rocket_info(self, rocket: str):
        """ Parse the Rocket identifier.

        The Rocket identifier is a String following this pattern: rocket_author/rocket_name/rocket_version
        The rocket_version is not mandatory and if not provided by the user the newest version will be returned.

        Args:
            rocket (str): Rocket identifier that needs to be parsed
        """
        assert len(rocket) > 0, 'Please specify the rocket you want to get.'
        
        # Parse the Rocket url
        rocket_parsed = rocket.split('/')
        assert len(rocket_parsed) > 1, 'Please provide more information about the rocket you want to get.'

        rocket_author  = rocket_parsed[0]
        rocket_name    = rocket_parsed[1]

        print('Looking for the Rocket ' + rocket_name + ' made by ' + rocket_author + '...')
        payload = {'author': rocket_author, 'model': rocket_name}
        rocket_version = ''
        if len(rocket_parsed) > 2:
            rocket_version = rocket_parsed[2]
            payload['version'] = rocket_version
        res = requests.get(self.models_api_url, params=payload)
        
        # if status != 200 then database is broken
        assert res.status_code == 200, 'Database error. Please try again later.'
        self.models = res.json()

        # Test that the rocket exists
        assert len(self.models) > 0, rocket + 'rocket cannot be found from our database. Please check the spelling.'
        print('{models_len} model versions found from the database.'.format(models_len=len(self.models)))
        
        # TODO: Select using some better logic
        self.selected_model = self.models[0]
        if rocket_version:
            print('Version ' + rocket_version + 'selected.')
        else:
            print('You didn\'t specify the version so the newest one is used.')
            rocket_version = 'v1'

        return rocket_author, rocket_name, rocket_version

    def get_rocket_url(self, rocket_author: str, rocket_name: str, rocket_version: str):
        """ Get the url from which to download the Rocket.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
            rocket_version (str): Version of the Rocket
        """
        return self.selected_model['modelFilePath']
    
    def get_rocket_folder(self, rocket_author: str, rocket_name: str, rocket_version: str):
        """ Get the name of the folder where the Rocket is unpacked.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
            rocket_version (str): Version of the Rocket
        """
        return self.selected_model['folderName']
    
    def get_rocket_last_version(self, rocket_author: str, rocket_name: str):
        """Get the last version of a Rocket.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
        """
        # Verify the rocket exist
        assert rocket_author in self.hangar.keys(), rocket_author + ' can\'t be found as an author.'
        assert rocket_name in self.hangar[rocket_author].keys(), rocket_name + ' can\'t be found as a rocket from ' + rocket_author

        # Get list of versions for a specific Rockets
        list_versions = [v[1:] for v in self.hangar[rocket_author][rocket_name].keys() if v.startswith('v')]

        mainVersion = 0
        minorVersion = 'a'

        for version in list_versions:
            v = ["".join(x) for _, x in itertools.groupby(version, key=str.isdigit)]
            temp_mainVersion = int(v[0])
            temp_minorVersion = v[1]

            assert len(temp_minorVersion) == 1, 'Automatic selection of the newest version doesn\'t support minor version made of more than 1 char.' 

            if temp_mainVersion == mainVersion:
                if temp_minorVersion > minorVersion:
                    minorVersion = temp_minorVersion
            elif temp_mainVersion > mainVersion:
                mainVersion = temp_mainVersion
                minorVersion = 'a'
                if temp_minorVersion > minorVersion:
                    minorVersion = temp_minorVersion
        
        return 'v' + str(mainVersion) +  minorVersion


    def push_file_to_rocket_storage(self, source_file_name: str, destination_blob_name: str):
        """Push the latest version of a Rocket to the Cloud Storage

        Args:
            source_filename (str): Name/Path of the file to upload to Cloud Storage
            destination_blob_name (str): Name of the blob 
            chunk_size (int): Size of Chunk to be uploaded
        """
        self.get_service_credentials()

        blob = self.bucket.blob(destination_blob_name)

        print("Please wait.")

        with open(source_file_name, 'rb') as f:
            blob.upload_from_file(f)

        return blob.public_url

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

        assert res.status_code == 201, "Push Rocket Update has failed! Status code : {} \n\n Response message:\n {}".format(res.status_code, res.text)
        
        return res.status_code == 201