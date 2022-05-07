# adversarial_seizure_detection
## Paper: Adversarial representation learning for robust patient-independent epileptic seizure detection

## Description
Link: https://arxiv.org/abs/1909.10868

## Overview
This repository contains the codes for reproducing the adversarial_seizure_detection model of the paper titled "Adversarial representation learning for robust patient-independent epileptic seizure detection"

## Dependencies

If you are using conda, it is recommended to switch to a new virtual environment.

## Requirements

1. Tensorflow 1.15
2. Python 3.7

## Data
Download the data from the following box folder.
https://uofi.box.com/s/z2ji0ilz1n7ylamv1v3s5pa4bmo9imyc

## File description
1. model.py -> script used for the three CNN models 
2. util.py -> script used to load the configruation file, log the hyperparameter and save the test results
3. adversarial_seizure_detection.py ->script used for running the model

## Training & Evaluation

To train and evaluate the model performance, run the following commands.

python adversarial_seizure_detection.py

## Results

| Subject ID |  |  |    |    |     |     |     |    |     |     |    |     |     |Average Accuracy|
| --- | --- | --- | --- |--- | --- | --- | --- |--- | --- | --- |--- | --- | --- | --- |
|0    | 1   | 2   | 3   | 4  | 5   |6    | 7   | 8  | 9   | 10  | 11 | 12  |13   |     | 
| 0.755|0.814|0.965|0.781|0.762|0.719|0.731|0.904|0.772|0.629|0.92|0.578|0.694|0.755|0.77|

## Citation
@inproceedings{
author = {Xiang Zhang, Lina Yao, Manqing Dong, Zhe Liu, Yu Zhang and Yong Li },
title = {Adversarial Representation Learning for Robust Patient-Independent Epileptic Seizure Detection},
year = {2020},
isbn = {9781450383592},
publisher = {IEEE},
url = {https://ieeexplore.ieee.org/abstract/document/8994148?casa_token=oQVEBO55ba8AAAAA:e3O3w4iyrSDLPWXY2O10ceD47DfB-cR5-KZboDCtHKYQ2QcKD3kN3iI6JC1PKA5kQa4-Yg14},
doi = {10.1109/JBHI.2020.2971610},
booktitle = {IEEE Journal of Biomedical and Health Informatics},
pages = {2852 - 2859},
numpages = {8},
keywords = {Non-invasive EEG,seizure detection,patient-independent,adversarial deep learning},
}
