import glob
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