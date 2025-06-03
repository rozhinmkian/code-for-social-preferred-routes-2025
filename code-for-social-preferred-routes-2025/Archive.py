import os
from datetime import datetime
import re
import tqdm as tqdm
import networkx as nx
from concurrent.futures import ThreadPoolExecutor


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Archiving and File Loading functions


def list_folder(directory):
    """
    Lists all the entries in the given directory
    Args:
        directory (str): path of directory.
    Returns:
        list: of folder names in the directory
    """
    return [entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))]


def create_folder_in_wd(folder_name, given_dir = None):
    """
    Create a folder with the specified name in the current working directory.
    Args:
        folder_name (str): Name of the folder to create.
    """
    # Get the current working directory
    current_directory = given_dir if given_dir is not None else os.getcwd()
        
    print(f"Current Working Directory: {current_directory}")
    
    # Create the full path for the new folder
    folder_path = os.path.join(current_directory, folder_name)
    
    # Create the folder if it doesn't already exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created successfully at {folder_path}.")
    else:
        print(f"Folder '{folder_name}' already exists at {folder_path}.")

    return folder_path


def save_function_arguments(file_name, kwargs):
    """
    Save the parameters passed to the function as arguments in a text file.
    
    Args:
        file_name (str): Name of the text file to save the parameters.
        **kwargs: Arbitrary keyword arguments representing the parameters and their values.
    """
    with open(file_name, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}= {value}\n")
    print(f"Parameters saved to {file_name}.")



def generate_magic_number():
    """
    generates magic number of the form mm-hh-MM-YY
    """
    now = datetime.now()
    magic_number = f"{now.minute:02d}-{now.hour:02d}-{now.day:02d}-{now.month:02d}-{now.year % 100:02d}"
    return magic_number



def extract_patterns_from_filenames(directory, pattern):
    """
    searches for pattern in files' names in the directory
    Args:
        directory (str): path of directory.
        pattern (str): pattern searched for

    Returns:
        list: of of extrated parts of file names that match the pattern
    """

    # List to store extracted numeric pattern parts
    pattern_parts = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the expected pattern
        match = re.search(pattern, filename)
        if match:
            pattern_parts.append(match.group(1))

    return pattern_parts



def read_graphml_file(file_dir):
    return nx.read_graphml(file_dir)

def mass_read_graphs(hashtags, main_directory, suffix):
    """
    Reads .graphml files in parallel for given hashtags.

    Args:
        hashtags (list): List of hashtags to look for.
        main_directory (str): The main directory containing hashtag subfolders.

    Returns:
        list of networkx.Graph: List of loaded graphs.
    """
    raw_data = []

    file_paths = []
    for hashtag in tqdm(hashtags, desc="Finding .graphml files"):
        folder_dir = os.path.join(main_directory, hashtag)
        if not os.path.exists(folder_dir):
            raise FileNotFoundError(f"Directory '{folder_dir}' does not exist.")

        matching_files = [
            file for file in os.listdir(folder_dir)
            if hashtag in file and suffix in file and file.endswith(".graphml")
        ]

        if not matching_files:
            raise FileNotFoundError(f"No matching file found for hashtag '{hashtag}' in {folder_dir}")

        file_paths.append(os.path.join(folder_dir, matching_files[0]))
    
    # Parallel file reading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        raw_data = list(tqdm(executor.map(read_graphml_file, file_paths), total=len(file_paths), desc="Reading .graphml files"))

    return raw_data