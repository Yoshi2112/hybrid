# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:29:50 2019

@author: Yoshi
"""
import os
from shutil import copytree




def copy_analysis():
    # Just a quick code that copies the analysis folders of the specified run series
    # TO DO: Copy the summary textfile and any extracted files as well
    
    # Drives containing the 'runs' directory
    source_drive      = 'F:/'
    destination_drive = 'G:/'
    
    series_to_copy    = ['reflective_boundary_tests_small']
    
    if series_to_copy == 'all':
        pass
        # Get series list from os.listdir
    
    for series in series_to_copy:
        series_folder = source_drive + '/runs/' + series
        dest_folder   = destination_drive  + '/runs/' + series
        
        if os.path.exists(dest_folder) == False:
            os.makedirs(dest_folder)
            print('Creating directory ', dest_folder)
        
        for run in os.listdir(series_folder):
            dest_run = dest_folder + '/' + run + '/'
            if os.path.exists(dest_run) == False:
                os.makedirs(dest_run)
                print('Creating directory ', dest_run)
            
            analysis_folder = series_folder + '/' + run + '/analysis/'
            
            print('Copying contents of {} analysis'.format(run))
            copytree(analysis_folder, dest_run + '/analysis/')
    return


def create_directory_structure(n_runs=10):
    '''
    Create directory structure for fast and easy drag-and-drop from SSH. Creates:
        - Main series dir
            - run_XX
                - data
                - analysis
                - extracted
    '''
    destination_drive = 'F:/'
    
    series_dirs = ['new_Jul25_ParaRuns_changing_cHe', 'new_Jul25_ParaRuns_changingA', 'new_Jul25_ParaRuns_changingT']
    
    for series in series_dirs:
        for ii in range(n_runs):
            for folder in ['data', 'analysis', 'extracted']:
                dest_folder = destination_drive  + 'runs/' + series + '/run_{}/'.format(ii) + folder
                if os.path.exists(dest_folder) == False:
                    os.makedirs(dest_folder)
    return


if __name__ == '__main__':
    #copy_analysis()
    create_directory_structure()