import importlib
import json
import os
import requests
import sys
import types

from datetime import datetime
from tqdm import tqdm

import rocketbase.api
import rocketbase.utils
from rocketbase.exceptions import *

def read_slug(rocket: str):
    """Parse the Rocket URL
    """
    rocket_parsed = rocket.split('/')
    assert len(rocket_parsed) > 1, "Please provide more information about the rocket"
    rocket_username = rocket_parsed[0].lower()
    rocket_modelName   = rocket_parsed[1].lower()
    rocket_hash= rocket_parsed[2] if len(rocket_parsed)>2 else ""
    return rocket_username, rocket_modelName, rocket_hash

def get_rocket_folder(rocket_slug: str):
    """Build Rocket folder name
    """
    rocket_username, rocket_modelName, rocket_hash = read_slug(rocket_slug)
    rocket_folder_name = rocket_username+'_'+rocket_modelName
    if len(rocket_hash) > 7:
        rocket_folder_name = rocket_folder_name+'_'+rocket_hash
    print("Rocket folder is {}".format(rocket_folder_name))
    return rocket_folder_name

def check_metadata(data: dict):
    """Verify the completness of the metadata provided in the info.json file
    """
    assert len(data['builder'])>1, "Please provide a builder name in info.json"
    assert '_' not in data['builder'], "You can not use underscores in the builder name"
    assert len(data['model'])>1, "Please provide a model name in info.json"
    assert '_' not in data['model'], "You can not use underscores in the model name"
    assert len(data['family'])>1, "Please provide the family name of the Rocket in info.json"
    valid_families = [
        "image_object_detection",
        "image_human_pose_estimation",
        "image_classification",
        "image_superresolution",
        "image_style_transfer",
        "image_segmentation",
        "image_instance_segmentation"
        ]
    assert data['family'] in valid_families, "Please enter a valid Rocket family in info.json. For a list of available families, please refer to the documentation."
    assert len(data['dataset'])>1, "Please specify the dataset this Rocket was trained on in info.json"
    assert len(data['rocketRepoUrl'])>1, "Please specify the URL of the Rocket code repository in info.json"
    assert len(data['paperUrl'])>1, "Please specify the URL of the scientific publication in info.json"
    assert len(data['originRepoUrl'])>1, "Please specify the URL of the origin code repository in info.json"
    assert len(data['description'])>1, "Please add a description¨of your rocket in info.json"
    assert len(data['blueprint'])>0, "Please add elements to the blueprint in info.json"
    assert type(data['isTrainable']) is bool, "Please enter 'true' or 'false' for isTrainable in info.json"

class Rocket:

    @staticmethod
    def land(rocket_slug: str, display_loading = True):
        """ Download or check that the Rocket is ready locally

        Download the Rocket if it is not yet locally here.

        In this function we are comparing 3 different source of information:
            - rocket_info_user: the information provided by the user in the rocket_slug.
            - rocket_info_api: the information provided by the api.
            - rocket_info_local: the information provided by the folder name of the local Rockets.

        Args:
            rocket_slug (str): Rocket identifier (username/modelName/(hash or label))
            display_loading (boolean): Display the loading bar. Can be useful to remove it when using it on a server with logs.

        Returns:
            model (nn.Module): Rocket containing the PyTorch model and the pre/post process model.
        """
        # Define the chunk size for the download
        CHUNK_SIZE = 512

        # Define the folder path for the Rocket
        FOLDER_PATH = 'rockets'

        # Parse the Rocket Slug
        rocket_info_user = rocketbase.utils.convert_slug_to_dict(rocket_slug)

        # Create the API object
        api = rocketbase.api.RocketAPI()

        # Check if the rocket exists and get the last version if not precised
        try:
            rocket_info_api = api.get_rocket_info(rocket_info_user)[0] #Only get one model
        except requests.exceptions.RequestException as e:  # Catch all the Exceptions relative to the request
            print('Problem with the API:', e)
            rocket_info_api = {}
        except rocketbase.api.RocketNotEnoughInfo as e:
            sys.exit(e)
        except rocketbase.api.RocketAPIError as e:
            print('API Error:', e)
            rocket_info_api = {}
        except rocketbase.api.RocketNotFound as e: 
            print('No Rocket found with the API using the slug:', rocket_slug)
            rocket_info_api = {}

        # Check if folder to download the rockets exists
        if not os.path.exists(FOLDER_PATH):
            os.makedirs(FOLDER_PATH)

        # If the API returned a Rocket
        if rocket_info_api:
            # Create the folder name
            rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(rocket_info_api)

            # Rocket already downloaded locally -- No need to download it
            if rocket_folder_name in os.listdir(FOLDER_PATH):
                print('Rocket has already landed. Using the local version:', rocket_folder_name)

            # Need to download the Rocket
            else:
                path_to_landing_rocket = os.path.join(FOLDER_PATH, 'landing_' + rocket_folder_name +'.tar')

                #Download URL
                print('Rocket approaching...')
                h = requests.head(rocket_info_api['downloadUrl'], allow_redirects=True)
                headers = h.headers
                content_type = headers.get('content-type')
                content_length = int(headers.get('content-length', None))
                # print('content-type', content_type)

                response = requests.get(rocket_info_api['downloadUrl'], stream=True)
            
                if display_loading: pbar = tqdm(total=content_length, ascii=True, desc='Rocket Landing')
                with open(path_to_landing_rocket, 'wb') as handle:
                    for data in response.iter_content(chunk_size = CHUNK_SIZE):
                        handle.write(data)
                        # update the progress bar
                        if display_loading: pbar.update(CHUNK_SIZE)
                
                if display_loading: pbar.close()
                
                # Unpack the downloaded tar file to a Rocket
                rocket_folder_path = rocketbase.utils.unpack_tar_to_rocket(path_to_landing_rocket, rocket_folder_name, FOLDER_PATH, remove_after_unpack = True)

                print('It is a success! The Rocket has landed!')

        else:
            # Get all the rocket_info from the Rockets in the folder
            list_rocket_info_local = rocketbase.utils.get_list_rocket_info_from_folder(FOLDER_PATH)

            # Get all the folders for the same Rocket (but different versions)
            list_rocket_info_local = [ri for ri in list_rocket_info_local if ri['username'] == rocket_info_user['username'] and ri['modelName'] == rocket_info_user['modelName']]

            if not list_rocket_info_local:
                raise RocketNotFound('No Rocket found locally using the slug: ' + rocket_slug)
            else:
                if 'label' in rocket_info_user.keys():
                    rocket_info_local = [ri for ri in list_rocket_info_local if ri['hash'] == rocket_info_user['label']]

                    if rocket_info_local:
                        rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(rocket_info_local[0])
                        print('Rocket found locally.')
                    else:
                         raise RocketNotFound('No Rocket found locally using the slug: {}'.format(rocket_slug))
                
                elif len(list_rocket_info_local) > 1:
                    raise RocketNotEnoughInfo('There are multiple local versions of the Rocket \'' + rocket_slug + '\'. Please choose a specific version by providing the hash of the Rocket.')
                
                else:
                    rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(list_rocket_info_local[0])
                    print('Rocket found locally.')
        
        print("Let's prepare the Rocket...")
        #Build the model
        module = importlib.import_module('rockets.{}.rocket_builder'.format(rocket_folder_name))
        build_func = getattr(module, 'build')
        model = build_func()
        
        return model

    @staticmethod
    def launch(rocket: str, folder_path = "rockets"):
        """ Upload the latest Rocket that is ready localy

        Upload the latest version of the Rocket that is localy available

        Args:
            rocket (str): Rocket Identifier (author/name/(version))
            folder_path (str): folder where to find the Rocket
        """
        # Get Rocket information
        rocket_username, rocket_modelName, rocket_hash = read_slug(rocket)

        # Get path to Rocket
        rocket_path = get_rocket_folder(rocket_slug=rocket)

        # Open info.json to verify information
        with open(os.path.join(folder_path, rocket_path, 'info.json')) as metadata_file:
            metadata_dict = json.load(metadata_file)
            check_metadata(metadata_dict)
            assert str(metadata_dict['builder']) == str(rocket_username), "The Rocket author name does not match the information in info.json. {} vs {}".format(rocket_username, metadata_dict['builder'])
            assert str(metadata_dict['model']) == str(rocket_modelName), "The Rocket model name does not match the information in info.json. {} vs {}".format(rocket_modelName, metadata_dict['model'])

        print("Let's load everything into the Rocket...")
        
        # Pack folder into archive
        path_to_launch_rocket = rocketbase.utils.pack_rocket_to_tar(folder_path, rocket_path, blueprint=metadata_dict['blueprint'])
        
        print("Let's get the new version name...")
        # Get new rocket hash
        new_rocket_hash = rocketbase.utils.get_file_SHA1_hash(path_to_launch_rocket)
        
        print("Rocket ready to launch!")

        # Init API for Rocket Upload
        api = rocketbase.api.RocketAPI()
        # Launch Rocket
        launch_success = api.push_rocket(
            rocket_username =rocket_username,
            rocket_modelName =rocket_modelName,
            rocket_hash =new_rocket_hash,
            rocket_family = metadata_dict['family'],
            trainingDataset = metadata_dict['dataset'],
            isTrainable = metadata_dict['isTrainable'],
            rocketRepoUrl = metadata_dict['rocketRepoUrl'], 
            paperUrl = metadata_dict['paperUrl'],
            originRepoUrl = metadata_dict['originRepoUrl'],
            description = metadata_dict['description'],
            tar_file=path_to_launch_rocket)

        print('Rocket reached its destination.' if launch_success else "There was a problem with the launch")
        if launch_success:
            os.remove(path_to_launch_rocket)
        return launch_success