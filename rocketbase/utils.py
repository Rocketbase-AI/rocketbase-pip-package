import glob
import json
import hashlib
import os
import tarfile

import rocketbase.exceptions

# --- CONSTANTS ---
# List of all the required information
LIST_REQUIRED_INFO = [
    'username',
    'modelName',
    'family',
    'trainingDataset',
    'isTrainable',
    'rocketRepoUrl',
    'originRepoUrl',
    'description',
    'blueprint'
]

# List of all the valid Rocket families
LIST_ROCKET_FAMILIES = [
    'image_object_detection',
    'image_human_pose_estimation',
    'image_classification',
    'image_superresolution',
    'image_style_transfer',
    'image_segmentation',
    'image_instance_segmentation'
]

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
        t.extractall(folder_path)  # unpack in the wrong folder

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
    rocket_glob = glob.glob(os.path.join(
        folder_path, rocket_folder)+"/**/*", recursive=True)

    # Create tar file
    with tarfile.open(tar_path, "w") as tar_handle:
        for filename in rocket_glob:
            _filename = filename.replace(os.path.join(folder_path, rocket_folder), "").replace(
                str(os.sep), "", 1).replace(str(os.sep), "/")
            if _filename in blueprint:
                tar_handle.add(filename)

    return tar_path


def get_file_sha1_hash(file_path: str):
    """Compute SHA-1 Hash of a file

    Args:
        file_path (str): Path to the file we want to compute the hash from.

    Returns:
        hash (str): SHA-1 hash of the referenced file.

    Raises:
        RocketHashNotValid: If the computed SHA-1 has a different length from the constant LENGTH_SHA1_HASH.
    """
    LENGTH_SHA1_HASH = 40

    with open(file_path, 'rb') as f:
        buf = f.read()
        file_hash = hashlib.sha1(buf).hexdigest()

    if len(file_hash) != LENGTH_SHA1_HASH:
        raise rocketbase.exceptions.RocketHashNotValid(
            'SHA-1 hash computation failed on file: {}'.format(file_path))

    return file_hash

# --- ROCKET INFO + CONVERSION ---


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
    if not rocket_slug:
        raise rocketbase.exceptions.RocketNotEnoughInfo(
            'Please specify the slug of the Rocket you want to get (e.g. <username>/<modelName>).')

    # Parse the Rocket url
    rocket_parsed = rocket_slug.split(parsing_char)
    if not rocket_parsed:
        raise rocketbase.exceptions.RocketNotEnoughInfo(
            '\'{}\' is not a correct slug for a Rocket. Please provide more information about the Rocket you want to get (<username>/<modelName>).'.format(rocket_slug))

    rocket_username = str(rocket_parsed[0])
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
    list_folders = [f for f in os.listdir(
        folder_path) if not f.startswith('.') and f.count('_') >= 2]

    list_rocket_info = [convert_slug_to_dict(
        f, '_', 'hash') for f in list_folders]

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
        raise rocketbase.exceptions.RocketNotEnoughInfo(
            'Missing the following information to create the Rocket\'s folder name: ' + ', '.join(missing_info))

    rocket_folder_name = rocket_info['username'] + str(separation_char) + \
        rocket_info['modelName'] + str(separation_char) + rocket_info['hash']

    return rocket_folder_name


def import_rocket_info_from_rocket_folder(rocket_folder_path: str, metadata_json_filename: str = 'info.json'):
    """ Import the metadata information about a Rocket from its folder.

        Import the information in the json file named with <metadata_json_filename> and check the information to see if they corresponds to LIST_REQUIRED_INFO and LIST_ROCKET_FAMILIES.

    Args:
        rocket_folder_path (str): path to the Rocket's folder
        metadata_json_filename (str): name of the .json file containing the metadata information about the Rocket.

    Returns:
        rocket_info (dict): dictionary containing all the Rocket metadata information.

    Raises:
        RocketNotEnoughInfo: If there is not enough information in the json file to launch the Rocket.
        RocketInfoFormat: If some information about the Rocket are not formatted the right way.
    """
    # Path to the file containing the information about the Rocket
    metadata_json_path = os.path.join(
        rocket_folder_path, metadata_json_filename)

    # Load the information from the .json file
    with open(metadata_json_path) as info_json:
        rocket_info = json.load(info_json)

    # -- INFO CHECK --
    # Check if some fields are missing
    missing_info = set(LIST_REQUIRED_INFO) - rocket_info.keys()

    if missing_info:
        raise rocketbase.exceptions.RocketNotEnoughInfo(
            'Missing some information about the Rocket in the file: ' + metadata_json_path + '. Missing the following information: ' + ', '.join(missing_info))

    # Check if some info are empty
    list_empty_info = [key for key, item in rocket_info.items() if not isinstance(
        item, bool) and not item and key in LIST_REQUIRED_INFO]

    if list_empty_info:
        raise rocketbase.exceptions.RocketNotEnoughInfo('Missing some information about the Rocket in the file: ' +
                                                        metadata_json_path + '. Please provide more information for the following field(s): ' + ', '.join(list_empty_info))

    # Check if the username contains a '_'
    if '_' in rocket_info['username']:
        raise rocketbase.exceptions.RocketInfoFormat(
            'In the file \'{}\', the username \'{}\' is not valid. It can\'t contains a \'_\'.'.format(metadata_json_path, rocket_info['username']))

    # Check if the modelName contains a '_'
    if '_' in rocket_info['modelName']:
        raise rocketbase.exceptions.RocketInfoFormat(
            'In the file \'{}\', the modelName \'{}\' is not valid. It can\'t contains a \'_\'.'.format(metadata_json_path, rocket_info['modelName']))

    # Check if the rocket family is in the list of valid families
    if not rocket_info['family'] in LIST_ROCKET_FAMILIES:
        raise rocketbase.exceptions.RocketInfoFormat(
            'In the file \'{}\', the family \'{}\' is not valid. Please refer to the documentation for a list of valid family names.'.format(metadata_json_path, rocket_info['family']))

    # Check if isTrainable is a boolean
    if not isinstance(rocket_info['isTrainable'], bool):
        raise rocketbase.exceptions.RocketInfoFormat(
            'In the file \'{}\',the field isTrainable needs to be a Boolean'.format(
                metadata_json_path))

    # Check if blueprint is a list
    if not isinstance(rocket_info['blueprint'], list):
        raise rocketbase.exceptions.RocketInfoFormat(
            'In the file \'{}\',the field blueprint needs to be a list of filenames.'.format(
                metadata_json_path))

    return rocket_info
