import glob
import hashlib
import importlib
import json
import os
import requests
import tarfile
import types
from tqdm import tqdm
from rockethub.api import RocketAPI
from datetime import datetime

def unpack_archive(path: str):
    ensure_dir('rockets')
    t = tarfile.open(path, 'r')
    model_name = os.path.commonprefix(t.getnames())
    t.extractall('rockets')
    t.close()
    os.remove(path)

    return model_name

def pack_archive(path: str, rocket_folder: str, blueprint: list):
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
    rocket_author = rocket_parsed[0].lower()
    rocket_name   = rocket_parsed[1].lower()
    rocket_version= rocket_parsed[2] if len(rocket_parsed)>2 else ""
    return rocket_author, rocket_name, rocket_version

def get_rocket_folder(rocket_slug: str):
    """Build Rocket folder name
    """
    rocket_author, rocket_name, rocket_version = read_slug(rocket_slug)
    rocket_folder_name = rocket_author+'_'+rocket_name
    if len(rocket_version) > 7:
        rocket_folder_name = rocket_folder_name+'_'+rocket_version[:8]
    print("Rocket folder is {}".format(rocket_folder_name))
    return rocket_folder_name

def get_rocket_version(rocket_path: str):
    """Compute SHA-1 Hash of the Rocket Tar
    """
    with open(rocket_path, 'rb') as f:
        buf = f.read()
        version = hashlib.sha1(buf).hexdigest()
        assert len(version)>1, "Version hash computation failed"
    return version

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

def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class Rocket:

    @staticmethod
    def land(rocket: str, folder_path = 'rockets', chunk_size = 512):
        """ Download or check that the Rocket is ready locally

        Download the Rocket if it is not yet locally here.

        Args:
            rocket (str): Rocket identifier (author/name/(version))
            folder_path (str): folder where to check if the Rocket is here already or where to download it
            chunk_size (int): size of the chunk when downloading the Rocket
        """
        api = RocketAPI()
        # Check if the rocket exists and get the last version if not precised
        rocket_author, rocket_name, rocket_version = api.get_rocket_info(rocket)

        # Check if folder to download the rockets exists
        ensure_dir(folder_path)
        
        # Check if the rocket has already been downloaded
        rocket_folder = api.get_rocket_folder(rocket_author, rocket_name, rocket_version)

        if rocket_folder in os.listdir(folder_path): 
            model_name = rocket_folder
            print('Rocket has already landed.')
        
        else:
            # Get the rocket's url
            url = api.get_rocket_url(rocket_author, rocket_name, rocket_version)
            path_to_landing_rocket = os.path.join(folder_path, 'landing_rocket.tar')

            #Download URL
            print('Rocket approaching...')
            h = requests.head(url, allow_redirects=True)
            headers = h.headers
            content_type = headers.get('content-type')
            content_length = int(headers.get('content-length', None))
            # print('content-type', content_type)

            response = requests.get(url, stream=True)
            
            pbar = tqdm(total=content_length, ascii=True, desc='Rocket Landing')
            with open(path_to_landing_rocket, 'wb') as handle:
                for data in response.iter_content(chunk_size = chunk_size):
                    handle.write(data)
                    # update the progress bar
                    pbar.update(chunk_size)
            
            pbar.close()
            
            model_name = unpack_archive(path_to_landing_rocket)
            print('It is a sucess! The Rocket has landed!')

        print("Let's prepare the Rocket...")
        #Build the model
        module = importlib.import_module('rockets.{}.rocket_builder'.format(model_name))
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
        rocket_author, rocket_name, rocket_version = read_slug(rocket)

        # Get path to Rocket
        rocket_path = get_rocket_folder(rocket_slug=rocket)

        # Open info.json to verify information
        with open(os.path.join(folder_path, rocket_path, 'info.json')) as metadata_file:
            metadata_dict = json.load(metadata_file)
            check_metadata(metadata_dict)
            assert str(metadata_dict['builder']) == str(rocket_author), "The Rocket author name does not match the information in info.json. {} vs {}".format(rocket_author, metadata_dict['builder'])
            assert str(metadata_dict['model']) == str(rocket_name), "The Rocket model name does not match the information in info.json. {} vs {}".format(rocket_name, metadata_dict['model'])

        print("Let's load everything into the Rocket...")
        
        # Pack folder into archive
        path_to_launch_rocket = pack_archive(folder_path, rocket_path, blueprint=metadata_dict['blueprint'])
        
        print("Let's get the new version number...")
        # Get new rocket version
        new_rocket_version = get_rocket_version(path_to_launch_rocket)
        
        print("Rocket ready to launch!")

        # Init API for Rocket Upload
        api = RocketAPI()
        # Launch Rocket
        launch_success = api.push_rocket(
            rocket_author =rocket_author,
            rocket_name =rocket_name,
            rocket_version =new_rocket_version,
            rocket_family = metadata_dict['family'],
            trainingDataset = metadata_dict['dataset'],
            isTrainable = metadata_dict['isTrainable'] if type(metadata_dict['isTrainable']) is bool else False,
            rocketRepoUrl = metadata_dict['rocketRepoUrl'], 
            paperUrl = metadata_dict['paperUrl'],
            originRepoUrl = metadata_dict['originRepoUrl'],
            description = metadata_dict['description'],
            tar_file=path_to_launch_rocket)

        print('Rocket reached its destination.' if launch_success else "There was a problem with the launch")
        if launch_success:
            os.remove(path_to_launch_rocket)
        return launch_success