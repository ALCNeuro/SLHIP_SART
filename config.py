#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:10:58 2023

@author: arthurlecoz

config.py

"""
# %% Paths

rootpath = "/Volumes/DDE_ALC/PhD/SLHIP"

rawDataPath = rootpath + "/00_Raw"
behavDataPath = rootpath + "/01_BehavData"
cleanDataPath = rootpath + "/02_Preproc"

# %% Variables

pilot_id = "pilot_JS01"

sub_ids = [""]

# %% Functions

def find_closest_value(value1, arr2):
    import numpy as np
    index_closest_value = []
    closest_value = []
    d_array = [
        abs(value1 - value2) for i2, value2 in enumerate(arr2)
               ]
    index_closest_value.append(np.argmin(d_array))
    closest_value.append(arr2[np.argmin(d_array)])
    
    return(index_closest_value, closest_value)

def display_matrices_info(mat_type):
    """
    Parameters
    ----------
    mat_type : str : 'test_res' or 'probe_res'
        Display detail for either part of the matrix

    Returns
    -------
    None.

    """
    
    assert mat_type in ["test_res", "probe_res"], "Careful, mat_type can only be 'test_res' or 'probe_res'"
    if mat_type == "test_res" :
        print('''
        - test_res (matrices des résultats du SART, 1 ligne = 1 essai)
    
        Col 1: block number
        Col 2: block condition (always 2)
        Col 3: image set (always 3)
        Col 4: trial number
        Col 5: digit displayed
        Col 6: nogo digit
        Col 7: response key
        Col 8: stimulus onset (in seconds, PTB time)
        Col 9: duration presentation (in seconds)
        Col 10: response time (in seconds, PTB time)
        Col 11: correctness on nogo trial
        Col 12: correctness on go trial
        ''')
    elif mat_type == 'probe_res' :
        print('''
        - probe_res (résultats des probes, 1 ligne = 1 probe)
    
        Col 1: probe number
        Col 2: probe time  (in seconds, PTB time, theoretical)
        Col 3: probe time  (in seconds, PTB time, actual)
        Col 4: block number
        Col 5: block condition
        Col 6: trial number
        Col 7: Probe Question 1 - Response key
        Col 8: Probe Question 2 - Response key 
        Col 9: Probe Question 3 - Response key 
        Col 10: Probe Question 1 - Response time
        Col 11: Probe Question 2 - Response time 
        Col 12: Probe Question 3 - Response time 
        Col 13: Probe Question 1 - Question time
        Col 14: Probe Question 2 - Question time
        Col 15: Probe Question 3 - Question time
        Col 16: Probe Question 1 - Response value
        Col 17: Probe Question 2 - Response value
        Col 18: Probe Question 3 - Response value
    
        Probe Q1 : Etat d'esprit juste avant l'interruption.
            Ans :   1 - J'étais concentré-e sur la tâche 
                    2 - Je pensais à autre chose
                    3 - Je ne pensais à rien
                    4 - Je ne me souviens pas
    
        Probe Q2 : Sur quoi étiez-vous concentré-e?
                Sur quelque-chose:
            Ans :   1 - Dans la pièce
                    2 - Une pensee venant de vous
                    3 - Une pensée liée a la tâche
                    4 - Une hallucination dans mon champs de vision
    
        Probe Q3 : Notez votre vigilance
            Ans : 1 - 9 
        ''')
