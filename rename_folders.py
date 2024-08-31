import os
import glob
import re

def rename_folders_and_files_with_substring(directory_path, substring, replacement):
    for root, dirs, files in os.walk(directory_path):
        for folder_name in dirs:
            if substring in folder_name:
                print(folder_name)
                old_path = os.path.join(root, folder_name)
                new_folder_name = folder_name.replace(substring, replacement)
                new_path = os.path.join(root, new_folder_name)
                os.rename(old_path, new_path)
                print(f"Renamed folder '{folder_name}' to '{new_folder_name}'.")
        for file in files:
            if substring in file:
                old_path = os.path.join(root, file)
                new_file_name = file.replace(substring, replacement)
                new_path = os.path.join(root, new_file_name)
                #os.rename(old_path, new_pah)
                print(f"Renamed file '{file}' to '{new_file_name}'.")

# Example usage:
directory_to_search = "."
substring_to_check = ":"
substring_to_replace = "T"

rename_folders_and_files_with_substring(directory_to_search, substring_to_check, substring_to_replace)