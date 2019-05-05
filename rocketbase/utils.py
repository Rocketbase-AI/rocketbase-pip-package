import glob
import hashlib
import os
import tarfile

import rocketbase.exceptions

# --- TAR ARCHIVE --- 
def unpack_tar_to_rocket(tar_path: str, rocket_folder_name: str, folder_path: str, remove_after_unpack: bool = True):
    """Unpack a tar archive to a Rocket folder

    Unpack a tar archive in a specific folder, rename it and then remove the tar file (or not if the user doesn't want to)

    Args:
        tar_path (str): path to the tar file containing the Rocket which should be unpacked
        rocket_folder_name (str): folder name for the Rocket (to change the one from the tar file)
        folder_path (str): folder where the Rocket should be moved once unpacked.
        remove_after_unpack (bool, optional): choose to remove the tar file once the Rocket is unpacked. Defaults to True.

    Returns:
        rocket_folder_path(str): path to the Rocket folder once unpacked.
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

def pack_rocket_to_tar(folder_path: str, rocket_folder: str, blueprint: list):
    """Packs a Rocket into a tar archive
    
    Packs a Rocket's contents as described in the blueprint list of files into a tar archive

    Args:
        folder_path (str): path to folder containing the Rocket's folder and where the tar file will be created.
        rocket_folder (str): name of the Rocket's folder.
        blueprint (List[str]): list of all the file in the Rocket's folder that should be included in the tar file.

    Returns:
        tar_path (str): path the newly created tar file containing the Rocket.
    """
    # Path to the tar file
    tar_path = os.path.join(folder_path, rocket_folder + '_launch.tar')
    
    # Glob to explore files in Rocket's folder
    rocket_glob = glob.glob(os.path.join(folder_path, rocket_folder)+"/**/*", recursive=True)

    # Create tar file
    with tarfile.open(tar_path, "w") as tar_handle:
        for filename in rocket_glob:
            _filename = filename.replace(os.path.join(folder_path, rocket_folder), "").replace(str(os.sep), "", 1).replace(str(os.sep), "/")
            if _filename in blueprint:
                tar_handle.add(filename)

    return tar_path

def get_file_SHA1_hash(file_path: str):
    """Compute SHA-1 Hash of a file

    Args:
        file_path (str): Path to the file we want to compute the hash from.
    
    Returns:
        hash (str): SHA-1 hash of the referenced file.
    
    Raises:

    """
    LENGTH_SHA1_HASH = 40

    with open(file_path, 'rb') as f:
        buf = f.read()
        hash = hashlib.sha1(buf).hexdigest()
    
    if len(hash) != LENGTH_SHA1_HASH:
        raise rocketbase.exceptions.RocketHashNotValid('SHA-1 hash computation failed on file: {}'.format(file_path))

    return hash

# --- INFO CONVERSION ---
def convert_slug_to_dict(rocket_slug: str, parsing_char: str = '/', version_type: str = 'label') -> dict:
    """Convert a Rocket slug to a dictionary.
    
    Convert a Rocket slug of the shape <username>/<modelName/(<hash> or <label>) (e.g. igor/retinanet) to a dictonary with the following structure: {'username': <username>, 'modelName': <name>, '<version_type>': <hash> or <label>}.
    All the arguments in the outputted dictionary are String. The <hash> or <label> in the Rocket slug is optional and will not be added to the output dictionary if it is not in the slug.
    
    Args:
        rocket_slug (str):  The Rocket slug in the shape <username>/<modelName>/(<hash> or <label>). The <hash> and <label> are optional. The <hash> should be complete.
        parsing_char (str): The character used to parse the information in the slug.
        version_type (str): The key to define the version (either label or hash)

    Returns:
        rocket_info (dict): A dict containing the information provided in rocket_slug.

    Raises:
        RocketNotEnoughInfo: If the <username> and/or the <modelName> of the Rocket are missing in the Rocket slug.
    """
    # Cast the rocket_slug to a String with lower case
    rocket_slug = str(rocket_slug).lower()

    # Check if the rocket_slug is not empty
    if len(rocket_slug) < 1: 
            raise rocketbase.exceptions.RocketNotEnoughInfo('Please specify the slug of the Rocket you want to get (e.g. <username>/<modelName>).')
        
    # Parse the Rocket url
    rocket_parsed = rocket_slug.split(parsing_char)
    if len(rocket_parsed) < 1:
        raise rocketbase.exceptions.RocketNotEnoughInfo('\'{}\' is not a correct slug for a Rocket. Please provide more information about the Rocket you want to get (<username>/<modelName>).'.format(rocket_slug))

    rocket_username  = str(rocket_parsed[0])
    rocket_modelName = str(rocket_parsed[1])

    rocket_info = {'username': rocket_username, 'modelName': rocket_modelName}
    
    # Check if a specific hash or label has been precised
    if len(rocket_parsed) > 2:
        rocket_label = parsing_char.join(rocket_parsed[2:])
        rocket_info[version_type] = rocket_label

    return rocket_info

def get_list_rocket_info_from_folder(folder_path: str) -> list:
    """Get the list of rocket_info from folders name inside of a folder.
    
    Args:
        folder_path (str): Path to the folder containing the folders of the Rockets.

    Returns:
        list_rocket_info (list): List of rocket_info of all the folders of the Rockets in folder_path. 
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
        raise rocketbase.exceptions.RocketNotEnoughInfo('Missing the following information to create the Rocket\'s folder name: ' + ', '.join(missing_info))
    
    rocket_folder_name = rocket_info['username'] + '_' + rocket_info['modelName'] + '_' + rocket_info['hash']

    return rocket_folder_name