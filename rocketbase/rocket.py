import importlib
import os
import requests
import sys

from tqdm import tqdm

import rocketbase.api
import rocketbase.utils
import rocketbase.exceptions

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

        Raises:
            RocketNotFound: If the Rocket is not found either throught the API or locally.
            RocketNotEnoughInfo: If not enough information is provided to load the correct Rocket.

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
        except rocketbase.exceptions.RocketNotEnoughInfo as e:
            sys.exit(e)
        except rocketbase.exceptions.RocketAPIError as e:
            print('API Error:', e)
            rocket_info_api = {}
        except rocketbase.exceptions.RocketNotFound as e: 
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
                raise rocketbase.exceptions.RocketNotFound('No Rocket found locally using the slug: ' + rocket_slug)
            else:
                if 'label' in rocket_info_user.keys():
                    rocket_info_local = [ri for ri in list_rocket_info_local if ri['hash'] == rocket_info_user['label']]

                    if rocket_info_local:
                        rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(rocket_info_local[0])
                        print('Rocket found locally.')
                    else:
                         raise rocketbase.exceptions.RocketNotFound('No Rocket found locally using the slug: {}'.format(rocket_slug))
                
                elif len(list_rocket_info_local) > 1:
                    raise rocketbase.exceptions.RocketNotEnoughInfo('There are multiple local versions of the Rocket \'' + rocket_slug + '\'. Please choose a specific version by providing the hash of the Rocket.')
                
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
    def launch(rocket_slug: str):
        """ Upload the latest Rocket that is ready localy

        Upload the latest version of the Rocket that is localy available

        Args:
            rocket_slug (str): Rocket slug (<username>/<modelName>/<hash>). The <hash> is not optional.
        
        Returns:
            launch_success (bool): true is the launch was successful, false in the other case.
        
        Raises:
            RocketNotEnoughInfo: If the <hash> of the Rocket to upload is not given in the rocket_slug
        """
        # Define the folder path for the Rocket
        FOLDER_PATH = 'rockets'

        # Get Rocket information
        rocket_info_user = rocketbase.utils.convert_slug_to_dict(rocket_slug, version_type = 'hash')

        if not 'hash' in rocket_info_user.keys():
            raise rocketbase.exceptions.RocketNotEnoughInfo('Please include the hash of the version of the Rocket you want to launch.')

        # Get name of the Rocket's folder
        rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(rocket_info_user)

        # Open info.json to verify information
        rocket_folder_path = os.path.join(FOLDER_PATH, rocket_folder_name)
        rocket_info_local = rocketbase.utils.import_rocket_info_from_rocket_folder(rocket_folder_path)

        print("Let's load everything into the Rocket...")
        
        # Pack folder into archive
        path_to_rocket_ready_to_launch = rocketbase.utils.pack_rocket_to_tar(FOLDER_PATH, rocket_folder_name, blueprint=rocket_info_local['blueprint'])
        
        print("Let's get the new version name...")
        # Get new rocket hash
        new_rocket_hash = rocketbase.utils.get_file_SHA1_hash(path_to_rocket_ready_to_launch)

        # Add the hash of the Rocket to the info about the Rocket
        rocket_info_local['hash'] = new_rocket_hash
        
        print("Rocket ready to launch!")

        # Init API for Rocket Upload
        api = rocketbase.api.RocketAPI()
        # Launch Rocket
        try:
            launch_success = api.push_rocket(rocket_info_local, path_to_rocket_ready_to_launch)
            
            print('Rocket reached its destination.')

        except requests.exceptions.RequestException as e:  # Catch all the Exceptions relative to the request
            print('Problem with the request:', e)
            launch_success = False
        except rocketbase.exceptions.RocketAPIError as e:
            print('API Error:', e)
            launch_success = False
        except rocketbase.exceptions.RocketNotEnoughInfo as e:
            print('Not enough Information to upload the Rocket', e)
            launch_success = False
        except Exception as e:
            print(e)
            launch_success = False
        
        return launch_success