import os
import tarfile

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