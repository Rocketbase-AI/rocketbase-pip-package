import glob
import hashlib
import importlib
import json
import os
import requests
import sys
import tarfile
import types

from datetime import datetime
from tqdm import tqdm

import rockethub.api
from rockethub.exceptions import *


def unpack_tar_to_rocket(tar_path: str, rocket_folder_name: str, folder_path: str, remove_after_unpack: bool = True):
    """Unpack a TAR archive to a Rocket folder

    Unpack a TAR archive in a specific folder, rename it and then remove the tar file (or not if the user wants to)

    """
    with tarfile.open(tar_path, 'r') as t:
        tar_folder_name = os.path.commonprefix(t.getnames())
        t.extractall(folder_path) # unpack in the wrong folder

    # Should rename the folder once it is unpacked
    rocket_folder_path = os.path.join(folder_path, rocket_folder_name)
    os.rename(os.path.join(folder_path, tar_folder_name), rocket_folder_path)

    if remove_after_unpack:
        os.remove(tar_path)

    return rocket_folder_path

def pack_rocket_to_tar(path: str, rocket_folder: str, blueprint: list):
    """Packs a Rocket into a TAR archive
    
    Packs a Rocket's contents as described in the blueprint into a TAR archive for upload
    """
    with tarfile.open(os.path.join(path, rocket_folder + '_launch.tar'), "w") as tar_handle:
        for filename in glob.glob(os.path.join(path, rocket_folder)+"/**/*", recursive=True):
            _filename = filename.replace(os.path.join(path, rocket_folder), "").replace(str(os.sep), "", 1).replace(str(os.sep), "/")
            if _filename in blueprint:
                tar_handle.add(filename)

    return os.path.join(path, rocket_folder + '_launch.tar')

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

def get_rocket_hash(rocket_path: str):
    """Compute SHA-1 Hash of the Rocket TAR archive

    Args:
        rocket_path (str): Path to the TAR archive of the Rocket, not the Rocket folder
    """
    with open(rocket_path, 'rb') as f:
        buf = f.read()
        _hash = hashlib.sha1(buf).hexdigest()
        assert len(_hash)>1, "Version hash computation failed"
    return _hash

def check_metadata(data: dict):
    """Verify the completness of the metadata provided in the info.json file
    """
    assert len(data['builder'])>1, "Please provide a builder name in info.json"
    assert '_' not in data['builder'], "You can not use underscores in the builder name"
    assert len(data['model'])>1, "Please provide a model name in info.json"
    assert '_' not in data['model'], "You can not use underscores in the model name"
    assert len(data['family'])>1, "Please provide the family name of the Rocket in info.json"
    assert len(data['dataset'])>1, "Please specify the dataset this Rocket was trained on in info.json"
    assert len(data['rocketRepoUrl'])>1, "Please specify the URL of the Rocket code repository in info.json"
    assert len(data['paperUrl'])>1, "Please specify the URL of the scientific publication in info.json"
    assert len(data['originRepoUrl'])>1, "Please specify the URL of the origin code repository in info.json"
    assert len(data['description'])>1, "Please add a description¨of your rocket in info.json"
    assert len(data['blueprint'])>0, "Please add elements to the blueprint in info.json"
    assert type(data['isTrainable']) is bool, "Please enter 'true' or 'false' for isTrainable in info.json"

def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_list_rocket_info_from_folder(folder_path: str) -> list:
    """Get the list of rocket_info from folders name inside of a folder.
    
    Args:
        folder_path (str): Path to the folder containing the folders of the Rockets.

    Returns:
        list_rocket_info (list): List of rocket_info of the all the folders of the Rockets in folder_path. 
    """
    list_folders = [f for f in os.listdir(folder_path) if not f.startswith('.') and f.count('_') >= 2]

    list_rocket_info = [convert_slug_to_dict(f, '_', 'hash') for f in list_folders]

    return list_rocket_info

def convert_dict_to_foldername(rocket_info: dict, separation_char: str = '_') -> str:
    """Convert a dict containing the information about a Rocket to a folder name.
    
    Args:
        rocket_info (dict):  Dictionary containing the information about a Rocket.
        separation_char (str): Character used to separate the information in the name of the folder.

    Returns:
        rocket_folder_name (str): Name of the folder containing the Rocket.

    Raises:
        RocketNotEnoughInfo: If there are not enough information to create the folder name
    """
    missing_info = set(['username', 'modelName', 'hash']) - rocket_info.keys()

    if missing_info:
        raise RocketNotEnoughInfo('Missing the following information to create the Rocket\'s folder name: ' + ', '.join(missing_info))
    
    rocket_folder_name = rocket_info['username'] + '_' + rocket_info['modelName'] + '_' + rocket_info['hash']

    return rocket_folder_name

def convert_slug_to_dict(rocket_slug: str, parsing_char: str = '/', version_type: str = 'label') -> dict:
    """Convert a Rocket slug to a dictionary.
    
    Convert a Rocket slug of the shape <username>/<modelName/(<hash> or <label>) (e.g. igor/retinanet) to a dictonary with the following structure: {'username': <username>, 'modelName': <name>, 'version': <hash> or <label>}.
    All the arguments in the outputted dictionary are String. The <hash> or <label> in the Rocket slug is optional and will not be added to the output dictionary if it is not precised.
    
    Args:
        rocket_slug (str):  The Rocket slug in the shape <username>/<modelName>/(<hash> or <label>). The <hash> and <label> are optional. The <hash> should be complete.
        parsing_char (str): The character used to parse the information in the slug.
        version_type (str): The key to use to define the version (either label or hash)

    Returns:
        rocket_info (dict): A dict containing the information provided in rocket_slug.

    Raises:
        RocketNotEnoughInfo: If the <username> and the <modelName> of the Rocket are not in the Rocket slug.
    """
    # Cast the rocket_slug to a String with lower case
    rocket_slug = str(rocket_slug).lower()

    # Check if the rocket_slug is not empty
    if len(rocket_slug) < 1: 
            raise RocketNotEnoughInfo('Please specify the slug of the Rocket you want to get (e.g. <username>/<modelName>).')
        
    # Parse the Rocket url
    rocket_parsed = rocket_slug.split(parsing_char)
    if len(rocket_parsed) < 1:
        raise RocketNotEnoughInfo('\'' + rocket_slug + '\' is not a correct slug for a Rocket. Please provide more information about the Rocket you want to get (<username>/<modelName>).')

    rocket_username  = str(rocket_parsed[0])
    rocket_modelName = str(rocket_parsed[1])

    rocket_info = {'username': rocket_username, 'modelName': rocket_modelName}
    
    # Check if a specific hash or label has been precised
    if len(rocket_parsed) > 2:
        rocket_label = parsing_char.join(rocket_parsed[2:])
        rocket_info[version_type] = rocket_label

    return rocket_info



class Rocket:

    @staticmethod
    def land(rocket_slug: str, folder_path = 'rockets', display_loading = True):
        """ Download or check that the Rocket is ready locally

        Download the Rocket if it is not yet locally here.

        In this function we are comparing 3 different source of information:
            - rocket_info_user: the information provided by the user in the rocket_slug.
            - rocket_info_api: the information provided by the api.
            - rocket_info_local: the information provided by the folder name of the local Rockets.

        Args:
            rocket_slug (str): Rocket identifier (username/modelName/(hash or label))
            folder_path (str): folder where the Rockets are stored
            display_loading (boolean): Display the loading bar. Can be useful to remove it when using it on a server with logs.
        """
        # Define the chunk size for the download
        CHUNK_SIZE = 512

        # Parse the Rocket Slug
        rocket_info_user = convert_slug_to_dict(rocket_slug)

        # Create the API object
        api = rockethub.api.RocketAPI()

        # Check if the rocket exists and get the last version if not precised
        try:
            rocket_info_api = api.get_rocket_info(rocket_info_user)[0] #Only get one model
        except requests.exceptions.RequestException as e:  # Catch all the Exceptions relative to the request
            print('Problem with the API:', e)
            rocket_info_api = {}
        except rockethub.api.RocketNotEnoughInfo as e:
            sys.exit(e)
        except rockethub.api.RocketAPIError as e:
            print('API Error:', e)
            rocket_info_api = {}
        except rockethub.api.RocketNotFound as e: 
            print('No Rocket found with the API using the slug:', rocket_slug)
            rocket_info_api = {}

        # print(rocket_info_api)

        # Check if folder to download the rockets exists
        ensure_dir(folder_path)

        # If the API returned a Rocket
        if rocket_info_api:
            # Create the folder name
            rocket_folder_name = convert_dict_to_foldername(rocket_info_api)

            # Rocket already downloaded locally -- No need to download it
            if rocket_folder_name in os.listdir(folder_path):
                print('Rocket has already landed. Using the local version:', rocket_folder_name)

            # Need to download the Rocket
            else:
                path_to_landing_rocket = os.path.join(folder_path, 'landing_' + rocket_folder_name +'.tar')

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
                
                rocket_folder_path = unpack_tar_to_rocket(path_to_landing_rocket, rocket_folder_name, folder_path)
                print('It is a sucess! The Rocket has landed!')

        else:
            # Get all the rocket_info from the Rockets in the folder
            list_rocket_info_local = get_list_rocket_info_from_folder(folder_path)

            # Get all the folders for the same Rocket (but different versions)
            list_rocket_info_local = [ri for ri in list_rocket_info_local if ri['username'] == rocket_info_user['username'] and ri['modelName'] == rocket_info_user['modelName']]

            if not list_rocket_info_local:
                raise RocketNotFound('No Rocket found locally using the slug: ' + rocket_slug)
            else:
                if 'label' in rocket_info_user.keys():
                    rocket_info_local = [ri for ri in list_rocket_info_local if ri['hash'] == rocket_info_user['label']]

                    if rocket_info_local:
                        rocket_folder_name = convert_dict_to_foldername(rocket_info_local[0])
                        print('Rocket found locally.')
                    else:
                         raise RocketNotFound('No Rocket found locally using the slug: {}'.format(rocket_slug))
                
                elif len(list_rocket_info_local) > 1:
                    raise RocketNotEnoughInfo('There are multiple local versions of the Rocket \'' + rocket_slug + '\'. Please choose a specific version by providing the hash of the Rocket.')
                
                else:
                    rocket_folder_name = convert_dict_to_foldername(list_rocket_info_local[0])
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
        path_to_launch_rocket = pack_rocket_to_tar(folder_path, rocket_path, blueprint=metadata_dict['blueprint'])
        
        print("Let's get the new version name...")
        # Get new rocket hash
        new_rocket_hash = get_rocket_hash(path_to_launch_rocket)
        
        print("Rocket ready to launch!")

        # Init API for Rocket Upload
        api = rockethub.api.RocketAPI()
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