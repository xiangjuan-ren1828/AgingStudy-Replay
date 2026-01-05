#!/usr/bin/env python
# coding: utf-8

# # Scikit-learn decoding: accuracy with different HRF peaks
# ========================================================================================================================================================
# Write by XR @ Nov 4th 2025
# This .py file was converted using the following command: 
# jupyter nbconvert --to script /Users/ren/Projects-NeuroCode/MyExperiment/Aging-SeqMemTask/fMRI-code/AgingReplay_LocalizerDecoding_EightCateory_HPC.ipynb
# ========================================================================================================================================================
# HPC-ready version:
# - Accept ROI name as a command-line argument (for SLURM array jobs)
# - Prints key parameters at startup


# In[1]:
import os
import sys
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
from os.path import join as opj


# In[2]:
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm

from nilearn import datasets
from nilearn import plotting, signal
from nilearn.image import mean_img, index_img, load_img,new_img_like
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.decoding import Decoder
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from nilearn.maskers import NiftiMasker
from nilearn.masking import intersect_masks
import nibabel as nib


# In[ ]:
# ================================================================
# --- 1. Handle command-line argument for ROI name ---
# ================================================================
if len(sys.argv) > 1:
    ROI_names = [sys.argv[1]]
else:
    ROI_names = ['VISventral']  # default ROI if not specified

print("===================================================")
print(f" Running decoding for ROI(s): {ROI_names}")
print("===================================================\n")

# ================================================================
# --- 2. Define paths and basic parameters ---
# ================================================================
# behavioral data 
path_behv         = '/home/mpib/ren/rxj-neurocode/AgingStudy/AgingReplay-InHouseData/AgingReplay-UKE-fMRI-data/'
# preprocessed data
fMRI_predata_path = '/home/mpib/ren/rxj-neurocode/AgingStudy/AgingStudy-fMRI-data/fMRI-PreprocessedData'
# after preprocessed data
fMRI_preRes_path  = '/home/mpib/ren/rxj-neurocode/AgingStudy/AgingStudy-fMRI-data/fMRI-PreprocessResult'
# masks
path_ROI          = '/home/mpib/ren/rxj-neurocode/AgingStudy/AgingStudy-fMRI-data/Decoding/output_mask'

subj_list = ['JI9FBD', 'DS56RI', '0TSX26', '0KDSM1', '34HZBU', 'JFD947', '1YLTBH', '2IMSQ1', 'F0MP4R', 'ISAL7K', # sub0-9
             'V75YLW', 'G39NYH', 'KB9Y23', 'PQV62B', 'PN1J6S', 'D7ZC32', 'ZTB3C6', 'PAE1Z6', 'H9YF0P', 'NGS98A', # sub11-21
             'I5O2AG', '0HUA8H', '9UQ4LO', 'BVAN57', '1OLZ78', 'R94NKY', 'HV9GS2', 'X30G9L', 'AC14EG', 'FVL29M', # sub22-32; 
             '5KU60C', '4POAI5', 'AW73C4', 'J7ZK18', '0NMLX1', '5L31VH', 'WJ4V80', '0CH2V5', '3FL47P', '4GDKS3',
             'DP3HN2', '54BKAN', 'Q5UD9N', '0MHR1L', '498ITS', 'ZLO65J', 'A0L3GF', 'JSD705', '29RT0L', '7H3LT4',
             'P8HG10', 'AXY95M', 'TF7SV8', 'F2T3IP', 'O9D3TK', '9BAZY6', 'MFV73X', 'PGK25A', 'ZE40UX', '7S84EL', 
             'IUC51R', 'P1GL2A', '0UC5EX', 'GN8H7S', '4YX0QJ', 'R86IAV', 'DVA10E', 'THY413', 'WN9SJ7', 'HW48KL',
             '38VXJW', 'XWT5Q7', 'SGH1L7', 'PKHU64', 'X7FW58', 'AN2Q4P', 'UE2N3V', 'EOQ68N', '1CSZ8U', '6YD50L',
             'R5NM7F', 'M125IE', 'W15AIE', '8H0SUC', 'T0Z37Y', 'SX06IP', 'EIH2T8', '3YQFD7', 'IJS35F', # 'S28NAH': no behavioral data and fMRI data in the block 5
             'Q20RXI', 'E46LTX', 'X6N1OV', 'FDOC83', '7WXVN6', 'M2NPT9', '34UGKJ', 'BDX6S2', 'U19AMO', '6GBYQ8', 
             'R0IJF6', '0DK8JB', 'O6P4HN', '659CYO', 'C274BR', 'JS5XW1', '85VOFG', 'F34K9G', 'BY5FA7', 'UM25JN'] # 'TSD0O8';
subj_conds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 0: six 3-reports trials + two recons-only trials in each block
              0, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 1: from block 3, all one-report trials, including 6 marginal reports and 2 recons-only trials
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, # 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 1, 
subj_ids  = ['sub-00', 'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09',
             'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-20', 'sub-21', 
             'sub-22', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 
             'sub-33', 'sub-34', 'sub-35', 'sub-36', 'sub-37', 'sub-38', 'sub-39', 'sub-40', 'sub-41', 'sub-42',
             'sub-43', 'sub-44', 'sub-45', 'sub-46', 'sub-47', 'sub-48', 'sub-49', 'sub-50', 'sub-51', 'sub-52',
             'sub-54', 'sub-55', 'sub-56', 'sub-57', 'sub-58', 'sub-59', 'sub-60', 'sub-61', 'sub-62', 'sub-63',
             'sub-64', 'sub-65', 'sub-66', 'sub-67', 'sub-68', 'sub-69', 'sub-70', 'sub-71', 'sub-72', 'sub-73',
             'sub-74', 'sub-75', 'sub-76', 'sub-77', 'sub-78', 'sub-79', 'sub-80', 'sub-81', 'sub-82', 'sub-83',
             'sub-84', 'sub-85', 'sub-86', 'sub-87', 'sub-89', 'sub-91', 'sub-92', 'sub-93', 'sub-94', # 'sub-90', 
             'sub-96', 'sub-97', 'sub-99', 'sub-100','sub-101','sub-102','sub-103','sub-104','sub-105','sub-106',
             'sub-107','sub-108','sub-109','sub-110','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116'] # 'sub-53'
subj_Grp  = ['YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'OA',     'YA',
             'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'YA',     'YA',
             'OA',     'YA',     'OA',     'YA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA', 
             'OA',     'OA',     'OA',     'YA',     'OA',     'YA',     'OA',     'OA',     'OA',     'OA',
             'OA',     'OA',     'OA',     'YA',     'YA',     'OA',     'OA',     'OA',     'OA',     'OA',
             'YA',     'YA',     'YA',     'YA',     'OA',     'YA',     'OA',     'YA',     'OA',     'YA',
             'YA',     'YA',     'YA',     'YA',     'OA',     'YA',     'YA',     'YA',     'YA',     'OA',
             'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'YA',     'OA',
             'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA', # 'OA',     
             'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',
             'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA',     'OA'] # 'OA',     
subj_Gen  = ['F',      'F',      'M',      'F',      'M',      'M',      'M',      'F',      'F',      'F',
             'F',      'M',      'M',      'M',      'F',      'M',      'F',      'F',      'F',      'F',
             'M',      'M',      'M',      'M',      'M',      'F',      'F',      'F',      'F',      'M', 
             'M',      'F',      'F',      'F',      'M',      'M',      'M',      'F',      'F',      'M', 
             'F',      'M',      'M',      'M',      'F',      'F',      'M',      'M',      'M',      'M',
             'M',      'M',      'F',      'F',      'F',      'M',      'F',      'M',      'F',      'F',
             'M',      'M',      'F',      'F',      'F',      'M',      'M',      'F',      'F',      'F',
             'F',      'F',      'F',      'M',      'F',      'F',      'F',      'M',      'M',      'M',
             'F',      'F',      'F',      'F',      'F',      'F',      'F',      'M',      'M', # 'F',      
             'M',      'M',      'F',      'F',      'M',      'F',      'M',      'F',      'M',      'F',
             'F',      'M',      'F',      'M',      'M',      'F',      'F',      'F',      'F',      'M'] # 'F',      
subLen    = len(subj_list)
smtData_list = ['2023-09-28_08h10.21.524', '2023-11-07_12h21.54.858', '2023-11-08_12h03.23.545', '2023-11-08_15h10.19.205', '2023-11-14_12h26.41.641',
                '2023-11-15_16h25.22.624', '2023-11-29_13h10.27.075', '2023-12-04_14h21.48.292', '2023-12-13_15h26.11.242', '2023-12-20_14h13.37.842',
                '2024-03-06_14h51.12.188', '2024-03-12_09h00.28.550', '2024-03-13_13h39.04.471', '2024-03-13_15h22.04.307', '2024-03-19_09h07.10.114',
                '2024-03-20_13h29.23.858', '2024-03-25_14h55.23.013', '2024-03-27_15h09.18.625', '2024-04-15_16h15.06.663', '2024-04-22_15h16.52.756',
                '2024-06-11_12h34.56.855', '2024-07-16_09h04.02.385', '2024-07-23_10h31.20.475', '2024-07-24_13h03.12.582', '2024-07-30_09h42.04.952',
                '2024-07-30_11h40.23.996', '2024-08-07_13h35.38.022', '2024-08-14_13h05.46.869', '2024-08-20_09h02.05.721', '2024-08-27_08h56.33.363', 
                '2024-09-10_10h04.49.686', '2024-09-17_09h33.29.370', '2024-09-17_11h08.13.256', '2024-10-08_10h14.17.919', '2024-10-08_11h27.05.290',
                '2024-10-15_09h09.15.617', '2024-10-15_12h50.58.451', '2024-10-22_12h42.43.034', '2024-10-28_15h04.42.286', '2024-10-30_15h46.36.217',
                '2024-11-05_09h14.48.552', '2024-11-08_10h49.02.025', '2024-11-12_14h50.54.626', '2024-11-15_13h38.16.342', '2024-11-20_15h48.30.080',
                '2024-11-25_15h10.17.731', '2025-02-04_09h56.11.417', '2025-01-28_09h17.21.761', '2025-01-21_09h11.31.992', '2025-02-11_08h51.27.911', 
                '2025-02-25_08h51.57.056', '2025-03-04_09h10.24.733', '2025-03-04_10h45.44.121', '2025-03-04_17h43.56.474', '2025-03-08_09h09.12.369', 
                '2025-03-08_11h02.15.424', '2025-03-11_17h26.05.761', '2025-03-12_13h06.54.798', '2025-03-18_09h07.54.814', '2025-03-18_10h48.52.340',
                '2025-03-18_17h52.43.858', '2025-03-22_09h08.29.675', '2025-03-22_10h39.34.519', '2025-03-22_14h08.39.650', '2025-03-29_13h57.05.875',
                '2025-04-01_09h01.28.280', '2025-04-01_16h04.27.502', '2025-04-05_09h05.18.704', '2025-04-05_10h42.43.204', '2025-04-05_14h11.03.375',
                '2025-04-08_09h05.18.220', '2025-04-15_12h43.28.098', '2025-04-19_10h03.27.794', '2025-04-19_11h34.22.980', '2025-04-19_13h56.10.038',
                '2025-04-22_08h56.05.004', '2025-04-22_17h02.46.579', '2025-04-24_17h01.02.940', '2025-04-29_17h19.51.289', '2025-05-20_09h10.23.876',
                '2025-06-24_09h01.09.686', '2025-07-05_12h29.01.635', '2025-07-08_09h10.37.608', '2025-07-12_12h39.33.346', '2025-07-15_10h50.24.239',
                '2025-07-22_09h08.14.717', '2025-07-22_10h50.10.422', '2025-07-26_10h58.25.697', '2025-07-29_09h11.54.623', # '2025-07-15_12h24.42.883', 
                '2025-07-29_12h22.55.149', '2025-08-09_10h06.29.871', '2025-08-09_17h52.23.229', '2025-08-09_16h09.19.996', '2025-08-11_16h14.00.079',
                '2025-08-13_08h58.45.554', '2025-08-13_10h38.04.617', '2025-08-13_16h20.39.496', '2025-08-19_09h15.43.593', '2025-08-19_10h48.14.146',
                '2025-08-19_16h07.42.928', '2025-08-21_16h23.19.920', '2025-09-09_09h14.21.589', '2025-09-09_16h19.23.893', '2025-09-13_08h55.23.633',
                '2025-09-13_10h42.16.239', '2025-09-16_08h54.36.888', '2025-09-16_10h40.38.770', '2025-09-20_09h05.08.118', '2025-09-20_10h57.13.396'] 

locData_list = ['2023-09-28_14h12.56.518', '2023-11-08_16h48.30.873', '2023-11-09_16h11.02.674', '2023-11-10_16h15.27.003', '2023-11-14_16h14.58.453',
                '2023-11-16_16h19.27.518', '2023-11-30_16h26.12.803', '2023-12-05_10h55.41.830', '2023-12-14_17h22.16.905', '2023-12-21_14h48.08.816', 
                '2024-03-07_15h47.49.146', '2024-03-15_14h01.51.453', '2024-03-14_16h10.58.431', '2024-03-18_14h43.06.229', '2024-03-20_15h24.15.997',
                '2024-03-21_15h40.30.309', '2024-03-27_13h21.24.043', '2024-03-28_12h07.59.279', '2024-04-16_08h56.05.511', '2024-04-24_14h39.13.617', 
                '2024-06-12_14h58.24.877', '2024-07-17_13h06.22.770', '2024-07-24_11h21.01.554', '2024-07-31_12h06.33.131', '2024-07-31_13h40.36.111',
                '2024-07-31_15h14.59.533', '2024-08-26_08h17.05.765', '2024-08-21_11h31.01.752', '2024-08-21_13h05.21.235', '2024-08-28_11h28.51.541', 
                '2024-09-11_11h46.21.555', '2024-09-18_11h31.25.597', '2024-09-18_13h35.01.061', '2024-10-09_12h37.51.712', '2024-10-09_14h08.38.731',
                '2024-10-23_13h16.08.319', '2024-10-16_13h40.56.719', '2024-10-25_12h46.34.787', '2024-10-29_15h50.53.023', '2024-11-01_12h51.41.305',
                '2024-11-06_15h03.52.289', '2024-11-11_15h02.01.801', '2024-11-13_15h17.56.569', '2024-11-18_15h04.23.076', '2024-11-22_15h13.23.596',
                '2025-02-05_10h22.15.474', '2025-02-05_11h56.20.664', '2025-01-29_11h31.23.686', '2025-01-22_11h34.15.216', '2025-02-12_11h30.38.491',
                '2025-02-26_11h51.08.964', '2025-03-05_11h31.01.311', '2025-03-05_13h16.10.270', '2025-03-05_17h22.22.571', '2025-03-09_08h55.47.043', 
                '2025-03-09_11h14.45.199', '2025-03-12_15h59.04.598', '2025-03-14_17h00.43.853', '2025-03-19_11h31.37.973', '2025-03-19_13h23.08.607',
                '2025-03-19_17h48.38.268', '2025-03-23_09h01.48.727', '2025-03-23_10h44.52.869', '2025-03-23_12h23.55.884', '2025-03-31_17h22.15.850',
                '2025-04-02_11h32.25.159', '2025-04-02_16h06.53.734', '2025-04-06_08h54.46.511', '2025-04-06_10h58.29.983', '2025-04-06_12h39.18.492',
                '2025-04-09_14h06.02.361', '2025-04-17_17h16.34.354', '2025-04-20_08h52.06.111', '2025-04-20_10h48.25.255', '2025-04-20_12h27.05.023',
                '2025-04-25_17h00.52.891', '2025-04-23_16h38.16.664', '2025-04-25_13h35.09.338', '2025-04-30_17h12.52.913', '2025-05-21_16h07.48.679',
                '2025-06-25_08h55.15.622', '2025-07-06_12h17.06.370', '2025-07-11_08h32.56.172', '2025-07-13_11h52.46.942', '2025-07-18_10h19.57.273',
                '2025-07-25_08h36.42.589', '2025-07-25_10h05.00.383', '2025-07-27_10h41.11.084', '2025-08-01_08h34.18.084', # '2025-07-18_11h56.32.412',
                '2025-08-01_11h27.46.474', '2025-08-10_10h30.07.978', '2025-08-10_17h13.33.724', '2025-08-10_15h31.02.947', '2025-08-12_16h29.31.238',
                '2025-08-14_16h05.35.469', '2025-08-14_17h40.25.833', '2025-08-15_16h59.33.246', '2025-08-20_09h12.28.877', '2025-08-20_10h39.55.897',
                '2025-08-20_16h11.52.133', '2025-08-22_08h55.44.631', '2025-09-10_09h20.15.132', '2025-09-10_10h57.53.482', '2025-09-14_08h57.50.046',
                '2025-09-14_10h37.51.272', '2025-09-18_08h55.50.912', '2025-09-18_10h27.46.358', '2025-09-21_08h56.54.549', '2025-09-21_11h04.08.482']

locStim_csv_list = ['x',  'x',  'x',  '01', '02', '03', '04', '05', '01', '01', 
                    '02', '04', '03', '05', '01', '02', '03', '04', '01', '02', # Except for the last participant ('ISAL7K'), the index of the joint stimuli for the remaining participants ranges from 1 to 64
                    '03', '01', '02', '03', '04', '05', '01', '02', '03', '04', 
                    '05', '01', '02', '03', '04', '01', '05', '02', '03', '04',
                    '05', '01', '02', '03', '04', '05', '01', '03', '02', '04',
                    '02', '04', '05', '02', '02', '03', '04', '05', '01', '02',
                    '03', '04', '05', '01', '03', '04', '05', '01', '02', '03',
                    '04', '02', '02', '03', '04', '05', '01', '02', '03', '04',
                    '05', '01', '02', '03', '04', '01', '02', '05', '01', # '05', 
                    '03', '04', '01', '02', '03', '04', '05', '01', '02', '03', 
                    '04', '05', '02', '03', '04', '05', '01', '02', '03', '04'] # ; '01';  From participant '0HUA8H', we used a new localizer task, the stimuli are named 'LocMemRecStim0x.csv'
session_words = ['EpiMemTask', 'LocalizerTask']

eyeball_anal = 'no'
if eyeball_anal == 'yes':
    subj_list     = ['0UC5EX', '7S84EL', '38VXJW', '498ITS',
                     'AN2Q4P', 'AXY95M', 'DVA10E', 'EOQ68N',
                     'GN8H7S', 'IUC51R', 'NGS98A', 'P1GL2A',
                     'PKHU64', 'R86IAV', 'SGH1L7', 'TF7SV8',
                     'THY413', 'UE2N3V', 'WN9SJ7', 'X7FW58']
    subj_ids      = ['sub-66', 'sub-63', 'sub-74', 'sub-47',
                     'sub-79', 'sub-55', 'sub-70', 'sub-81',
                     'sub-67', 'sub-64', 'sub-21', 'sub-65',
                     'sub-77', 'sub-69', 'sub-76', 'sub-56',
                     'sub-71', 'sub-80', 'sub-72', 'sub-78'] 
    smtData_list  = ['2025-03-22_10h39.34.519', '2025-03-18_10h48.52.340', '2025-04-08_09h05.18.220', '2024-11-20_15h48.30.080',
                     '2025-04-22_08h56.05.004', '2025-03-04_09h10.24.733', '2025-04-01_16h04.27.502', '2025-04-24_17h01.02.940',
                     '2025-03-22_14h08.39.650', '2025-03-18_17h52.43.858', '2024-04-22_15h16.52.756', '2025-03-22_09h08.29.675',
                     '2025-04-19_11h34.22.980', '2025-04-01_09h01.28.280', '2025-04-19_10h03.27.794', '2025-03-04_10h45.44.121'
                     '2025-04-05_09h05.18.704', '2025-04-22_17h02.46.579', '2025-04-05_10h42.43.204', '2025-04-19_13h56.10.038']
    etData_list   = ['2025_03_22_10_39',        '2025_03_18_10_48',        '2025_04_08_09_05',        '2024_11_20_15_48',
                     '2025_04_22_08_56',        '2025_03_04_09_10',        '2025_04_01_16_04',        '2025_04_24_17_01',
                     '2025_03_22_14_08',        '2025_03_18_17_52',        '2024_04_22_15_16',        '2025_03_22_09_08',
                     '2025_04_19_11_34',        '2025_04_01_09_01',        '2025_04_19_10_03',        '2025_03_04_10_45',
                     '2025_04_05_09_05',        '2025_04_22_17_02',        '2025_04_05_10_42',        '2025_04_19_13_56']
    session_words = ['EpiMemTask', 'LocalizerTask']
    subLen        = len(subj_list)


# In[4]:


# baisc parameters for the localizer session
nPos   = 8 # 8 positions uniformly distributed on a circle
nImg   = 8 # 8 unique images
nComb  = nPos * nImg
nBlock = 8 # 64 trials in each block
nCatch = 8 # 8 catch trials


# In[5]:


# basic parameters for first-level GLM
sesName = 'ses-locTask'
datatype = 'func'
t_r = 2 # RepetitionTime = 2 s
# The 6 parameters correspond to three translations and three rotations describing rigid body motion
# confound_vars = ['trans_x','trans_y','trans_z',
#                  'rot_x','rot_y','rot_z']
confound_vars = ['global_signal', 'framewise_displacement',
                 'trans_x', 'trans_y', 'trans_z',
                 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                 'rot_x', 'rot_y', 'rot_z',
                 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                 'a_comp_cor_00','a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03','a_comp_cor_04', 'a_comp_cor_05']
# HRF
hrf_model = 'spm'
# drift model
drift_model = "polynomial" # 'cosine'
# drift order
drift_order = 3
# high-pass filter
high_pass = 0.01
# full-width at half maximum (width of the filter)
smoothing_fwhm = 6 # try 6 mm or 8 mm
# pre-whitening (or autocorrelation)
noise_model = 'ar1'
# whether standarize the time-series
standarize = False


# ## Whole brain decoding

# ### Raw data
# 
# <div class="alert alert-success">
# <b>Descriptions</b>:
#     
# Using the preprocessed data (3D X time) for decoding: shift the onset time of each stimulus 4-5 seconds (similar to HRF: a maximal activation) foward to find its corresponding TR.
# 
# !!!The catch trials have already been deleted!!!
# </div>


# In[ ]:


# --------The folder contains the anatomical runs--------
session_List = ['', # sub-00
                '', # sub-01
                '', # sub-02
                'ses-SeqMemTask', # sub-03
                'ses-SeqMemTask', # sub-04
                'ses-SeqMemTask', # sub-05
                'ses-SeqMemTask', # sub-06
                'ses-SeqMemTask', # sub-07
                'ses-locTask', # sub-08
                'ses-SeqMemTask', # sub-09
                'ses-SeqMemTask', # sub-11
                'ses-SeqMemTask', # sub-12
                'ses-SeqMemTask', # sub-13
                'ses-SeqMemTask', # sub-14
                'ses-SeqMemTask', # sub-15
                'ses-SeqMemTask', # sub-16
                'ses-SeqMemTask', # sub-17
                'ses-SeqMemTask', # sub-18
                'ses-SeqMemTask', # sub-20
                'ses-SeqMemTask', # sub-21
                'ses-SeqMemTask', # sub-22
                'ses-SeqMemTask', # sub-24
                'ses-SeqMemTask', # sub-25
                'ses-SeqMemTask', # sub-26
                'ses-SeqMemTask', # sub-27
                'ses-SeqMemTask', # sub-28
                'ses-SeqMemTask', # sub-29
                'ses-SeqMemTask', # sub-30
                'ses-SeqMemTask', # sub-31
                'ses-SeqMemTask', # sub-32
                'ses-SeqMemTask', # sub-33
                'ses-SeqMemTask', # sub-34
                'ses-SeqMemTask', # sub-35
                'ses-SeqMemTask', # sub-36
                'ses-SeqMemTask', # sub-37
                'ses-SeqMemTask', # sub-38
                'ses-SeqMemTask', # sub-39
                'ses-SeqMemTask', # sub-40
                'ses-SeqMemTask', # sub-41
                'ses-SeqMemTask', # sub-42
                'ses-SeqMemTask', # sub-43
                'ses-SeqMemTask', # sub-44
                'ses-SeqMemTask', # sub-45
                'ses-SeqMemTask', # sub-46
                'ses-SeqMemTask', # sub-47
                'ses-SeqMemTask', # sub-48
                'ses-SeqMemTask', # sub-49
                'ses-SeqMemTask', # sub-50
                'ses-SeqMemTask', # sub-51
                'ses-SeqMemTask', # sub-52 # 'ses-SeqMemTask', # sub-53
                'ses-SeqMemTask', # sub-54
                'ses-SeqMemTask', # sub-55
                'ses-SeqMemTask', # sub-56
                'ses-SeqMemTask', # sub-57
                'ses-SeqMemTask', # sub-58
                'ses-SeqMemTask', # sub-59
                'ses-SeqMemTask', # sub-60
                'ses-SeqMemTask', # sub-61
                'ses-SeqMemTask', # sub-62
                'ses-SeqMemTask', # sub-63
                'ses-SeqMemTask', # sub-64
                'ses-SeqMemTask', # sub-65
                'ses-SeqMemTask', # sub-66
                'ses-SeqMemTask', # sub-67
                'ses-SeqMemTask', # sub-68
                'ses-SeqMemTask', # sub-69
                'ses-SeqMemTask', # sub-70
                'ses-SeqMemTask', # sub-71
                'ses-SeqMemTask', # sub-72
                'ses-SeqMemTask', # sub-73
                'ses-SeqMemTask', # sub-74
                'ses-SeqMemTask', # sub-75
                'ses-SeqMemTask', # sub-76
                'ses-SeqMemTask', # sub-77
                'ses-SeqMemTask', # sub-78
                'ses-SeqMemTask', # sub-79
                'ses-SeqMemTask', # sub-80
                'ses-SeqMemTask', # sub-81
                'ses-SeqMemTask', # sub-82 
                'ses-SeqMemTask', # sub-83... 78 
                'ses-SeqMemTask', # sub-84... 84 
                'ses-SeqMemTask', # sub-85... 85 
                'ses-SeqMemTask', # sub-86... 86 
                'ses-SeqMemTask', # sub-87... 87
                'ses-SeqMemTask', # sub-89... 88
                'ses-SeqMemTask', # sub-90... 89
                'ses-SeqMemTask', # sub-91... 90
                'ses-SeqMemTask', # sub-92... 91
                'ses-SeqMemTask', # sub-93... 92
                'ses-SeqMemTask', # sub-94... 93
                'ses-SeqMemTask', # sub-96... 94
                'ses-SeqMemTask', # sub-97... 95
                'ses-SeqMemTask', # sub-99... 96
                'ses-SeqMemTask', # sub-100... 97
                'ses-SeqMemTask', # sub-101... 98
                'ses-SeqMemTask', # sub-102... 99
                'ses-SeqMemTask', # sub-103... 100
                'ses-SeqMemTask', # sub-104... 101
                'ses-SeqMemTask', # sub-105... 102
                'ses-SeqMemTask', # sub-106... 103
                'ses-SeqMemTask', # sub-107... 104
                'ses-SeqMemTask', # sub-108... 105
                'ses-SeqMemTask', # sub-109... 106
                'ses-SeqMemTask', # sub-110... 107
                'ses-SeqMemTask', # sub-111... 108
                'ses-SeqMemTask', # sub-112... 109
                'ses-SeqMemTask', # sub-113... 110
                'ses-SeqMemTask', # sub-114... 111
                'ses-SeqMemTask', # sub-115... 112
                'ses-SeqMemTask'  # sub-116... 113     
] # store the folder for the anatomical image


# #### 8 contents & 8 positions

# In[ ]:


event_word_list = ["con-pos-fix", "jointstim-fix"]
event_word = event_word_list[0] 
# ========loop across all subjects========
TRwords = ['nearestOne', 'averageTwo']
TRword  = TRwords[1]
spaces = ['T1w','MNI152NLin2009cAsym']
space  = spaces[0]
signal_ratio = [.0,.3,.5,.8]
T = signal_ratio[1]

# ----------Define which ROI mask should the script use----------
maskSource_list = ['funcRunsLoc', 'funcRunsAll', 'anatRun', 'funcRuns+anatRun']
maskId = 1
maskSource = maskSource_list[maskId]

# ----------create a new folder to save the decoding scores----------
decoding_saveDir = os.path.join(*[fMRI_preRes_path, 'decoding-folder', sesName, f'TR-{TRword}'])
if not os.path.exists(decoding_saveDir):
    os.makedirs(decoding_saveDir)

# ------8 contents------
# conditions_con = ['cat', 'hat', 'sunflower', 'key', 'castle', 'female', 'cream', 'car']
# conditions_con = ['cat', 'cherry', 'couch', 'cutter', 'girl', 'hand', 'house', 'violin']
# ------8 positions------
conditions_pos = ['0AngRight', '1AngRightup', '2AngUp', '3AngLeftup', '4AngLeft', '5AngLeftdown', '6AngDown', '7AngRightdown']
HRF_peaks = [1, 3, 5, 7, 9] # different HRF peak latencies
# ROI_names = ['VISventral'] # {'VIS', 'HPC', 'LPFC', 'VISloc', 'VISffa', 'VISppa', 'VISventral', 'wholeBrain'}

for i_roi in range(0, len(ROI_names)): # range(0, len(ROI_names))
    ROI_name = ROI_names[i_roi]
    print('++++++++++++++++++ ROI: ' + ROI_name + '++++++++++++++++++' )

    for i_hrf in range(0, len(HRF_peaks)):
        HRF_peak = HRF_peaks[i_hrf] #5 # for each event onset, adding another 5 seconds to find the peak of the activation
        print('++++++++++++++++++ HRF: ' + str(HRF_peak) + ' s++++++++++++++++++' )

        scores_con_pos = np.zeros((subLen, 4)) # 4 columns: col1&2-content and position decoding accuracy from the wrapper; col3&4: the same decoding accuracy from the manually procedure
        for subIdx in range(3, subLen): #
            subID_str   = subj_list[subIdx]
            subID_num   = subj_ids[subIdx]
            anat_folder = session_List[subIdx]
            print('++++++++++++++++++ subj: ' + subID_str + ' ++++++++++++++++++' )
            if subIdx < 10:
                tr_start = 2 # the stimuli will be displayed in the beginning of the 3rd TR
            else:
                tr_start = 3 # the stimuli will be displayed in the beginning of the 4th TR
            # --------blocks for the sequential memory task--------
            if subID_str == 'F0MP4R':
                nBlock_SMT = 2
            elif subID_str == 'H9YF0P':
                nBlock_SMT = 3
            elif subID_str == 'YSZO14':
                nBlock_SMT = 0
            elif subID_str == 'J7ZK18' or subID_str == '4GDKS3' or subID_str == 'TSD0O8':
                nBlock_SMT = 4
            elif subID_str == 'Q5UD9N' or subID_str == 'JSD705':
                nBlock_SMT = 5
            else:
                nBlock_SMT = 6

            # --------Blocks for the localizer task--------
            if subID_str == 'H9YF0P' or subID_str == 'AC14EG' or subID_str == 'A0L3GF' or subID_str == '7S84EL' or subID_str == '6GBYQ8':
                nBlock = 7
            elif subID_str == 'NGS98A' or subID_str == 'R94NKY' or subID_str == 'TF7SV8' or subID_str == 'X7FW58':
                nBlock = 6
            else:
                nBlock = 8 

            # refine the blocks from the localizer session

            if subIdx < 21:
                conditions_con = ['cat', 'hat', 'sunflower', 'key', 'castle', 'female', 'cream', 'car']
                event_word     = "con-pos-resp-fix"
            else:
                conditions_con = ['cat', 'cherry', 'couch', 'cutter', 'girl', 'hand', 'house', 'violin']
                event_word     = event_word_list[0] # "con-pos-fix"
            
            # ----------Read the z-map for each block and each object----------
            # concatenate the mask images
            if maskSource == 'funcRunsLoc': # only the ROI mask from the localizer task
                mask_images_blc = [] 
                if ROI_name == 'wholeBrain':
                    for iBlc in range(0, nBlock):
                        # ----------Load whole brain mask from functional run (exactly the same data used in localizer_maskROI.ipynb to extract the ROI mask)----------
                        # fMRI_predata_path = '/Volumes/fMRI-UKE/AgingStudy-fMRI-data/fMRI-PreprocessedData'
                        # mask_path = fMRI_predata_path + '/' + subID_num + '/' + 'output/' + subID_num + '/' + 'SeqMemTask' + '/' + datatype + '/'
                        mask_path = fMRI_predata_path + '/' + subID_num + '/' + 'output/' + subID_num + '/' + sesName + '/' + datatype + '/'
                        mask_fname = subID_num + '_' + sesName + '_task-localizer_run-0' + str(iBlc+1) + '_space-' + space + '_desc-brain_mask.nii.gz'
                        mask_file  = os.path.join(mask_path, mask_fname)
                        mask_img = nib.load(mask_file)
                        mask_images_blc.append(mask_img) 

                else:
                    for iBlc in range(0, nBlock):
                        # ----------Load ROI mask----------
                        mask_path = path_ROI + '/' + subID_num + '/' + sesName + '/' + 'ROIs/' 
                        mask_fname = ROI_name + '_' + subID_num + '_' + sesName + '_space-' + space + '_T-' + str(int(T*100)) + '_run-0' + str(iBlc+1) + '.nii'
                        mask_file  = os.path.join(mask_path, mask_fname)
                        print(f"functional nifti image (3D) is at: {mask_file}")
                        mask_img = nib.load(mask_file)
                        mask_images_blc.append(mask_img) 
                # Compute intersection of several masks
                mask_intersect = intersect_masks(mask_images_blc, threshold=0.5) # threshold=0.5 (default); the intersected mask should have the same shape and affine

            elif maskSource == 'funcRunsAll': # both the ROI mask from the main task (SeqMemTask) and localizer task
                mask_images_blc = [] 
                # ----------SeqMemTask----------
                sesName_SMT = 'ses-SeqMemTask'
                if nBlock_SMT != 0:
                    for iBlc in range(0, nBlock_SMT):
                        mask_path  = path_ROI + '/output_mask_SMT/' + subID_num + '/' + sesName_SMT + '/' + 'ROIs/' 
                        mask_fname = ROI_name + '_' + subID_num + '_' + sesName_SMT + '_space-' + space + '_T-' + str(int(T*100)) + '_run-0' + str(iBlc+1) + '.nii'
                        mask_file  = os.path.join(mask_path, mask_fname)
                        print(f"functional nifti image (3D) is at: {mask_file}")
                        mask_img = nib.load(mask_file)
                        mask_images_blc.append(mask_img) 

                # ----------Localizer task----------
                sesName_loc = 'ses-locTask'
                for iBlc in range(0, nBlock):
                    # ----------Load ROI mask----------
                    mask_path  = path_ROI + '/' + subID_num + '/' + sesName_loc + '/' + 'ROIs/' 
                    mask_fname = ROI_name + '_' + subID_num + '_' + sesName_loc + '_space-' + space + '_T-' + str(int(T*100)) + '_run-0' + str(iBlc+1) + '.nii'
                    mask_file  = os.path.join(mask_path, mask_fname)
                    print(f"functional nifti image (3D) is at: {mask_file}")
                    mask_img = nib.load(mask_file)
                    mask_images_blc.append(mask_img) 
                # Compute intersection of several masks
                mask_intersect = intersect_masks(mask_images_blc, threshold=0.5) # threshold=0.5 (default); the intersected mask should have the same shape and affine
                
            # Compute intersection of several masks
            masker = NiftiMasker(
                mask_img=mask_intersect, 
                smoothing_fwhm=smoothing_fwhm,
                standardize=True
            )

            # condition labels
            conditions_con_label    = []
            conditions_pos_label    = []
            # block labels
            run_con_label    = []
            run_pos_label    = []
            # masked func data across runs
            masked_data_con    = []
            masked_data_pos    = []
            func_path = fMRI_predata_path + '/' + subID_num + '/' + 'output/' + subID_num + '/' + sesName + '/' + datatype + '/'
            for iBlc in range(0, nBlock):
                # ----------Load func----------
                # load NifTi file from the local machine
                # ----------The preprocessed data----------
                func_fname = subID_num + '_' + sesName + '_task-localizer_run-0' + str(iBlc+1) + '_space-T1w_desc-preproc_bold.nii.gz' # '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
                func_file  = os.path.join(func_path, func_fname)
                # print basic information of the data
                print(f"functional nifti image (4D) is at: {func_file}")
                fun_img = nib.load(func_file)
                # Remove the first 2 TRs from the BOLD signal
                fun_img_used = fun_img.slicer[:, :, :, tr_start:]
                n_scans = np.shape(fun_img_used.get_fdata())[-1]

                # ----------Maker the functional data----------
                masked_data = masker.fit_transform(fun_img_used) # shape=(n_timepoints or n_scans, n_voxels)

                # +++++++++++++++++++Clean the data-++++++++++++++++++
                # the original raw-data decoding doesn't contain this part
                # ----------Load confound_timeseries----------
                confound_fname = subID_num + '_' + sesName + '_task-localizer_run-0' + str(iBlc+1) + '_desc-confounds_timeseries.tsv'
                confound_file  = os.path.join(func_path, confound_fname)
                print(f"Read the confound_timeseries: {confound_file}")
                #Delimiter is \t --> tsv is a tab-separated spreadsheet
                confound_df = pd.read_csv(confound_file, delimiter='\t')
                # Remove the first 2 TRs
                confound_df_used = confound_df.iloc[tr_start:] 
                # The 6 motion parameters
                confound_glm = confound_df_used[confound_vars]
                confounds_matrix = confound_glm.values
                masked_data_clean = signal.clean(signals=masked_data,
                                                 t_r=t_r, 
                                                 detrend=True,
                                                 standardize='zscore_sample', # zscore_sample
                                                 confounds=confounds_matrix)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
                
                # ----------Load event----------
                event_path  = fMRI_preRes_path + '/ses-locTask-events/' + subID_num + '_' + subID_str + '/'
                event_fname = event_path + 'events-' + event_word + '-block' + str(iBlc) + '.tsv'
                event_file  = os.path.join(event_path, event_fname)
                print(f"Read the event: {event_file}")
                events = pd.read_csv(event_file, delimiter='\t')

                # ----------Label the condition for the corresponding scans----------
                # loop over all events
                events_id = events['trial_type']
                # For the following 4 variables, the length should be <= 56 (already delete the responded trials; so if there is any false alarm within a block, the stimulus length will be < 56)
                con_onsets   = [] # onset time (plus HRF_peak) for concent
                pos_onsets   = [] # onset time (plus HRF_peak) for position
                for i_e in range(0, np.shape(events)[0]):
                    event_i = events_id[i_e]
                    if event_i in conditions_con:
                        # --------condition and run labels--------
                        conditions_con_label.append(event_i)
                        run_con_label.append(iBlc)
                        # --------onset time--------
                        event_i_on = events['onset'][i_e] + HRF_peak
                        con_onsets.append(event_i_on)
                        
                    elif event_i in conditions_pos:
                        # --------condition and run labels--------
                        conditions_pos_label.append(event_i)
                        run_pos_label.append(iBlc)
                        # --------onset time--------
                        event_i_on = events['onset'][i_e] + HRF_peak
                        pos_onsets.append(event_i_on)

                # the corresponding scans contain the peak activation of stimuli
                frame_times = np.arange(n_scans) * t_r  
                scan_con_idx    = [] # extract the scans that contain the activation of a stimulus
                scan_pos_idx    = [] # this should be exactly the same as scan_con_idx
                scan_jnStim_idx = []
                if TRword == 'nearestOne':
                    # ------only selecting the TR closest to the stimulus onset time + HRF_peak------
                    for i_c in range(0, len(con_onsets)):
                        abs_dif = np.abs(con_onsets[i_c] - frame_times)
                        scan_con_idx.append(np.argmin(abs_dif))
                    for i_p in range(0, len(pos_onsets)):
                        abs_dif = np.abs(pos_onsets[i_p] - frame_times)
                        scan_pos_idx.append(np.argmin(abs_dif))
                    # ----------Extracting the imaging data----------
                    masked_data_con_iBlc = masked_data_clean[scan_con_idx, :] # shape=(n_TR, n_voxels);
                    masked_data_pos_iBlc = masked_data_clean[scan_pos_idx, :] 

                elif TRword == 'averageTwo':
                    # ------averaging across two TRs closest to the stimulus onset time + HRF_peak------
                    # ------content------
                    for i_c in range(0, len(con_onsets)):
                        rel_dif = con_onsets[i_c] - frame_times
                        rel_dif = rel_dif[rel_dif >= 0] 
                        if len(rel_dif) < n_scans:
                            scan_con_idx.append([len(rel_dif)-1, len(rel_dif)])
                        else:
                            scan_con_idx.append([len(rel_dif)-1, len(rel_dif)-1])
                    # ------position------
                    for i_p in range(0, len(pos_onsets)):
                        rel_dif = pos_onsets[i_p] - frame_times
                        rel_dif = rel_dif[rel_dif >= 0]
                        if len(rel_dif) < n_scans:
                            scan_pos_idx.append([len(rel_dif)-1, len(rel_dif)])
                        else:
                            scan_pos_idx.append([len(rel_dif)-1, len(rel_dif)-1])
                    # ----------Extracting the imaging data----------
                    masked_data_con_iBlc = np.zeros((len(con_onsets), np.shape(masked_data_clean)[1])) # Time * Voxel
                    for i_c in range(0, len(con_onsets)):
                        masked_data_con_iBlc[i_c, :] = np.mean(masked_data_clean[scan_con_idx[i_c], :], axis=0) # shape=(n_TR, n_voxels);

                    masked_data_pos_iBlc = np.zeros((len(pos_onsets), np.shape(masked_data_clean)[1]))
                    for i_p in range(0, len(pos_onsets)):
                        masked_data_pos_iBlc[i_p, :] = np.mean(masked_data_clean[scan_pos_idx[i_p], :], axis=0)
                        
                # ----------Concatenate the masked data across runs----------
                masked_data_con.append(masked_data_con_iBlc)
                masked_data_pos.append(masked_data_pos_iBlc)
                
            # ----------squeeze the masked data----------
            masked_data_con = np.vstack(masked_data_con)
            masked_data_pos = np.vstack(masked_data_pos)
        
            # --------Leave-on-out group cross validation--------
            # specify the decoder
            logo = LeaveOneGroupOut() # leave one group out
            clf  = LogisticRegression(solver="lbfgs", penalty="l2") # "liblinear", "lbfgs"
            scaler = StandardScaler() 

            # **********(1) Decoder wrapper**********
            standardize_true_false = False # True #
            if standardize_true_false:
                pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
            else:
                pipeline = Pipeline([("clf", clf)])

            decoder_sklearn_con = cross_validate(
                pipeline,
                X=masked_data_con,
                y=conditions_con_label,
                groups=run_con_label,
                cv=logo,
                n_jobs=-1,
                scoring="accuracy",
            )

            decoder_sklearn_pos = cross_validate(
                pipeline,
                X=masked_data_pos,
                y=conditions_pos_label,
                groups=run_pos_label,
                cv=logo,
                n_jobs=-1,
                scoring="accuracy",
            )

            print(
                f"Wrapper decoding: Mean content & position decoding score in {subID_str}:",
                np.mean(decoder_sklearn_con["test_score"]),
                np.mean(decoder_sklearn_pos["test_score"]),        
            )

            scores_con_pos[subIdx, 0] = np.mean(decoder_sklearn_con["test_score"])
            scores_con_pos[subIdx, 1] = np.mean(decoder_sklearn_pos["test_score"])

            # **********(2) Step-by-step decoding**********
            # !!!!!!!!!!(1) & (2) should output the same results!!!!!!!!!!
            # ----------Content decoding----------
            X = masked_data_con
            # Convert y to a numpy array
            y = np.array(conditions_con_label)
            groups = run_con_label
            
            logo = LeaveOneGroupOut()
            #clf = LogisticRegression(solver='liblinear', penalty='l2')
            clf = LogisticRegression(solver="lbfgs", penalty="l2") # "liblinear", "lbfgs"
            scaler = StandardScaler()
            scores_con = []
            for train_index, test_index in logo.split(X, y, groups):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test)
            
                clf.fit(X_train_scaled, y_train)
            
                score = clf.score(X_test_scaled, y_test)
                scores_con.append(score)
            
            # ----------Position decoding----------
            X = masked_data_pos
            # Convert y to a numpy array
            y = np.array(conditions_pos_label)
            groups = run_pos_label
            
            logo = LeaveOneGroupOut()
            #clf = LogisticRegression(solver='liblinear', penalty='l2')
            clf = LogisticRegression(solver="lbfgs", penalty="l2") # "liblinear", "lbfgs"
            scaler = StandardScaler()
            scores_pos = []
            for train_index, test_index in logo.split(X, y, groups):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test)
            
                clf.fit(X_train_scaled, y_train)
            
                score = clf.score(X_test_scaled, y_test)
                scores_pos.append(score)
                
            print("Cross-validation scores:", scores_con, scores_pos)
            print(
                f"Manually decoding: Mean content & position decoding score in {subID_str}:",
                np.mean(scores_con),
                np.mean(scores_pos),        
            ) 
            scores_con_pos[subIdx, 2] = np.mean(scores_con)
            scores_con_pos[subIdx, 3] = np.mean(scores_pos)

            if standardize_true_false:
                standardWord = 'True'
            else:
                standardWord = 'False'

            # ~~~~~~~~~~~ Save data for individual participant ~~~~~~~~~~~
            scores_con_pos_iSub = scores_con_pos[subIdx, :]
            scores_con_pos_iSub_pd = pd.DataFrame(scores_con_pos_iSub)

            path_score_iSub = decoding_saveDir + '/' + subID_num + '-' + subID_str + '-' + sesName + '_space-' + space + '_signalT-' + str(int(T*100)) + '-' + ROI_name + '-standard' + standardWord + '-HRF' + str(HRF_peak) + '-scores_con_pos_iSub_pd.csv'
            scores_con_pos_iSub_pd.to_csv(path_or_buf = path_score_iSub) 
            # ~~~~~~~~~~~ ~~~~~~~~~~~ ~~~~~~~~~~~ ~~~~~~~~~~~ ~~~~~~~~~~~

        # --------Save the results from wrapper and step-to-step decoding separately--------
        scores_con_pos = scores_con_pos[3:, :]
        scores_con_pos_pd = pd.DataFrame(scores_con_pos)

        path_score = decoding_saveDir + '/' + sesName + '_space-' + space + '_signalT-' + str(int(T*100)) + '-' + ROI_name + '-standard' + standardWord + '-HRF' + str(HRF_peak) + '-scores_con_pos_pd.csv'
        scores_con_pos_pd.to_csv(path_or_buf = path_score) 

print("\nAll decoding runs completed successfully.\n")



