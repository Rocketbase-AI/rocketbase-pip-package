import importlib
import os
import sys

import requests
import torch
import tqdm

import rocketbase.api
import rocketbase.exceptions
import rocketbase.shadowRocket
import rocketbase.utils



class Rocket:

    @staticmethod
    def land(rocket_slug: str, device: str = 'GPU', force_local: bool = False, display_loading: bool = True):
        """ Download or check that the Rocket is ready locally

        Download the Rocket if it is not yet locally here.

        In this function we are comparing 3 different source of information:
            - rocket_info_user: the information provided by the user in the rocket_slug.
            - rocket_info_api: the information provided by the api.
            - rocket_info_local: the information provided by the folder name of the local Rockets.

        Args:
            rocket_slug (str): Rocket identifier (username/modelName/(hash or label))
            device (str): device the rocket should use to run. There are 4 possibilites:
                1. CPU: The Rocket will run only on CPU.
                2. GPU [Default]: The Rocket will run on the first available GPU.
                3. API: An API will be used instead of a local Rocket.
            force_local (bool): Default is False. Don't check the API when landing a rocket.
            display_loading (boolean): Display the loading bar. Can be useful to remove it when using it on a server with logs.

        Returns:
            model (nn.Module): Rocket containing the PyTorch model and the pre/post process model.

        Raises:
            RocketNotFound: If the Rocket is not found either through the API or locally.
            RocketNotEnoughInfo: If not enough information is provided to load the correct Rocket.

        """
        # Define the chunk size for the download
        CHUNK_SIZE = 512

        # Define the folder path for the Rocket
        FOLDER_PATH = 'rockets'

        # --- DEVICE ---
        # convert the device to a readable form
        device = device.lower().split(':')

        # Check if the device exists
        LIST_DEVICES = ['cpu', 'gpu', 'api']
        if device[0] not in LIST_DEVICES:
            raise rocketbase.exceptions.RocketDeviceNotFound(
                'The device \'{}\' was not found among the valid devices to use with a Rocket.'
            )

        # convert device to readable string by pytorch
        if device[0] == 'gpu' and torch.cuda.is_available():
            torch_device = torch.device('cuda:'+ str(device[1]) if len(device) > 1 else 'cuda')
        elif device[0] == 'api':
            torch_device = None
        else:
            torch_device = torch.device('cpu')
        
        # Inform the use if yes or no a GPU is detected
        if torch.cuda.is_available():
            print('GPU detected on the machine.')
        else:
            print('No GPU detected on the machine.')

        print('The selected device to execute the model is {}'.format(torch_device))

        # Strip the Rocket slug to avoid problem with leading and ending spaces
        rocket_slug = rocket_slug.strip()

        # Parse the Rocket Slug
        rocket_info_user = rocketbase.utils.convert_slug_to_dict(rocket_slug)

        if not force_local:
            # Create the API object
            api = rocketbase.api.RocketAPI()

            # Check if the rocket exists and get the last version if not precised
            try:
                # Only keep the first model
                rocket_info_api = api.get_rocket_info(rocket_info_user)[0]
            except requests.exceptions.RequestException as e:
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
        else:
            rocket_info_api = {}

        if device[0] in ['cpu', 'gpu']:
            # Check if folder to download the rockets exists
            if not os.path.exists(FOLDER_PATH):
                os.makedirs(FOLDER_PATH)

            # If the API returned a Rocket
            if rocket_info_api:
                # Create the folder name
                rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(
                    rocket_info_api)

                # Rocket already downloaded locally -- No need to download it
                if rocket_folder_name in os.listdir(FOLDER_PATH):
                    print('Rocket has already landed. Using the local version:{}'.format(
                        rocket_folder_name))

                # Need to download the Rocket
                else:
                    path_to_landing_rocket = os.path.join(
                        FOLDER_PATH, 'landing_' + rocket_folder_name + '.tar')

                    # Download URL
                    print('Rocket approaching...')
                    h = requests.head(
                        rocket_info_api['downloadUrl'], allow_redirects=True)
                    headers = h.headers
                    content_length = int(headers.get('content-length', None))

                    response = requests.get(
                        rocket_info_api['downloadUrl'], stream=True)

                    if display_loading:
                        pbar = tqdm.tqdm(total=content_length, ascii=True,
                                        desc='Rocket Landing')
                    with open(path_to_landing_rocket, 'wb') as handle:
                        for data in response.iter_content(chunk_size=CHUNK_SIZE):
                            handle.write(data)
                            # update the progress bar
                            if display_loading:
                                pbar.update(CHUNK_SIZE)

                    if display_loading:
                        pbar.close()

                    # Unpack the downloaded tar file to a Rocket
                    _ = rocketbase.utils.unpack_tar_to_rocket(
                        path_to_landing_rocket,
                        rocket_folder_name,
                        FOLDER_PATH,
                        remove_after_unpack=True)

                    print('It is a success! The Rocket has landed!')

            else:
                # Get all the rocket_info from the Rockets in the folder
                list_rocket_info_local = rocketbase.utils.get_list_rocket_info_from_folder(
                    FOLDER_PATH)

                # Get all the folders for the same Rocket (but different versions)
                list_rocket_info_local = [ri for ri in list_rocket_info_local if ri['username'] ==
                                        rocket_info_user['username'] and ri['modelName'] == rocket_info_user['modelName']]

                if not list_rocket_info_local:
                    raise rocketbase.exceptions.RocketNotFound(
                        'No Rocket found online and locally using the slug: \'{}\'. Please check the documentation online to see what are the Rockets available.'.format(rocket_slug))

                if 'label' in rocket_info_user.keys():
                    rocket_info_local = [
                        ri for ri in list_rocket_info_local if ri['hash'] == rocket_info_user['label']]

                    if rocket_info_local:
                        rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(
                            rocket_info_local[0])
                        print('Rocket found locally.')
                    else:
                        raise rocketbase.exceptions.RocketNotFound(
                            'No Rocket found locally using the slug: {}'.format(rocket_slug))

                elif len(list_rocket_info_local) > 1:
                    raise rocketbase.exceptions.RocketNotEnoughInfo(
                        'There are multiple local versions of the Rocket \'' + rocket_slug + '\'. Please choose a specific version by providing the hash of the Rocket.')

                else:
                    rocket_folder_name = rocketbase.utils.convert_dict_to_foldername(
                        list_rocket_info_local[0])
                    print('Rocket found locally. Using the local version:{}'.format(
                        rocket_folder_name))

            
            # Build the model
            module = importlib.import_module(
                'rockets.{}.rocket_builder'.format(rocket_folder_name))
            build_func = getattr(module, 'build')
            model = build_func()

            model.to(torch_device)

            print("Rocket successfully built.")

        elif device[0] in ['api']:
            if 'apiUrl' not in rocket_info_api.keys():
                raise rocketbase.exceptions.RocketNotEnoughInfo(
                    'No \'apiUrl\' found for the Rocket \'{}\''.format(rocket_slug)
                )
            print('Using the Shadow Rocket of the Rocket \'{}\''.format(rocket_slug))

            model = rocketbase.shadowRocket.ShadowRocket(rocket_info_api)

        return model

    @staticmethod
    def launch(rocket_folder_path: str, remove_launched_tar: bool = True):
        """ Upload the latest Rocket that is ready locally

        Upload the latest version of the Rocket that is locally available

        Args:
            rocket_folder_path (str): Path to the folder containing the info.json file of the Rocket.
            remove_launched_tar (bool): Default True. Remove the .tar file once it has been uploaded.

        Returns:
            launched_rocket_slug (str): slug of the successfully uploaded Rocket. If the upload failed then it returns an empty string.

        Raises:
            RocketNotEnoughInfo: If the <hash> of the Rocket to upload is not given in the rocket_slug
        """

        # Open info.json to verify information
        rocket_info_local = rocketbase.utils.import_rocket_info_from_rocket_folder(
            rocket_folder_path)

        print("Let's load everything into the Rocket...")

        rocket_name_in_tar = rocketbase.utils.convert_dict_to_foldername(
                rocket_info_local,
                include_hash= False
            )

        # Pack folder into archive
        path_to_rocket_ready_to_launch = rocketbase.utils.pack_rocket_to_tar(
            rocket_path = rocket_folder_path,
            rocket_folder_name = rocket_name_in_tar,
            blueprint=rocket_info_local['blueprint'])

        print("Let's get the new version name...")

        # Get new rocket hash
        new_rocket_hash = rocketbase.utils.get_file_sha1_hash(
            path_to_rocket_ready_to_launch)

        # Add the hash of the Rocket to the info about the Rocket
        rocket_info_local['hash'] = new_rocket_hash

        print("Rocket ready to launch!")

        # Init API for Rocket Upload
        api = rocketbase.api.RocketAPI()
        # Launch Rocket
        try:
            launched_rocket_slug = api.push_rocket(
                rocket_info_local, path_to_rocket_ready_to_launch)

            print('Rocket reached its destination.')

        except requests.exceptions.RequestException as e:
            print('Problem with the request:', e)
            launched_rocket_slug = ''
        except rocketbase.exceptions.RocketAPIError as e:
            print('API Error:', e)
            launched_rocket_slug = ''
        except rocketbase.exceptions.RocketNotEnoughInfo as e:
            print('Not enough Information to upload the Rocket', e)
            launched_rocket_slug = ''

        # Remove the tar file
        if remove_launched_tar:
            os.remove(path_to_rocket_ready_to_launch)

        return launched_rocket_slug
