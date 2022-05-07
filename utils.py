# -*- coding: utf-8 -*-

import logging
import json
import numpy as np
from pathlib import Path
import pandas as pd

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger


def load_param(param_file):
    '''
    Parameters
    ----------
    param_file : Hyperparameter json file.
        

    Returns
    -------
    Model Hyperparameters.

    '''
    with open(param_file) as f:
            params = json.load(f)
    return params
    
def log_hyperparameter(logger,params_file):
    '''
    Parameters
    ----------
    logger : Logging Handler
    para_file : 
        Model hyperparameters file
    
    Function Description:
        Save the model hyperparameters to the log file

    Returns
    -------
    None.

    '''
    parameters = load_param(params_file)
    for key, value in parameters.items():
        logger.info("%s : %s", key, value)

def save_data(filename, headerList,train_results):
    path_to_file = filename
    path = Path(path_to_file)
    results = pd.DataFrame([train_results],columns=headerList)
            
    if path.is_file():
        results.to_csv(filename,mode ='a', index =False, header = False)
    else:
        results.to_csv(filename,index =False)
