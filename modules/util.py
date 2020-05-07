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


def clear_wl_models():
    '''
    This function clears all of the workload GPR models in models/.
    '''
    dir = './models'
    files = [f for f in os.listdir(dir) if f[:2] == 'wl']

    for f in files:
        os.remove(f'{dir}/{f}')
