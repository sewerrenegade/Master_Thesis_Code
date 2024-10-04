#!/bin/env python
print("Strating Interactive script")

#salloc --nodes=1 --cpus-per-task=4 --mem=20G --time=3:00:00 --partition=interactive_gpu_p --qos=interactive_gpu --nice=10000 --gres=gpu:1 --job-name=std_master_allocation
#u need to make this file executable
#insert main function here


if __name__ == '__main__':
    from train import setup_and_start_training
    setup_and_start_training(1)
    # import os

    # def rename_tif_files(directory):
    #     # Walk through all directories and subdirectories
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             # Check if the file has the .TIF extension
    #             if file.endswith('.TIF'):
    #                 old_file_path = os.path.join(root, file)
    #                 new_file_path = os.path.join(root, file.replace('.TIF', '.tif'))
                    
    #                 # Rename the file
    #                 os.rename(old_file_path, new_file_path)
    #                 print(f'Renamed: {old_file_path} -> {new_file_path}')

    # # Call the function with the root directory you want to search in
    # directory_path = "data/SCEMILA/image_data/"
    # rename_tif_files(directory_path)