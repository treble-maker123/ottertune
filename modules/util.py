'''
Utility functions for the repository.
'''
import os


def clear_temp_models():
    '''
    This function clears all of the files in the models/temp folder.
    '''
    temp_dir = './models/temp'
    files = [f for f in os.listdir(temp_dir) if f != '.gitkeep']

    for f in files:
        os.remove(f'{temp_dir}/{f}')
