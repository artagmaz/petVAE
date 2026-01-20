# %% [markdown]
# # Quality control for registered PET scans
# 
# ## ============================================================================
# ## SECTION 1: Setup and Data Import
# ## ============================================================================

# %%
## ============================================================================
## SECTION 1.1: Import Libraries
## ============================================================================
# All libraries are used:
# - pandas, numpy: data manipulation
# - nibabel: NIfTI file I/O
# - scipy, statsmodels: statistical operations
# - math: mathematical functions
# - matplotlib.pyplot, seaborn: visualization
# - ants: neuroimaging operations
# - sys, os, fnmatch: system and file operations
import pandas as pd
import numpy as np
import nibabel as nib
import scipy
import statsmodels
import math

import matplotlib.pyplot as plt
import seaborn as sns

import ants

import sys
import os, fnmatch


# %%
## ============================================================================
## SECTION 1.2: Load Metadata
## ============================================================================
# Load PET-MRI pairs metadata
# Old file reference (commented out): 'metafile_ADDLpipeline_abeta_mri_02_06_2024.csv'
meta = pd.read_csv('/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/pet_mri_pairs.csv',header=[0], index_col=[0])

meta.reset_index(drop=True, inplace = True)
meta = meta.iloc[:,1:]

# %%
## ============================================================================
## SECTION 2: Load MNI Template
## ============================================================================
# Load MNI152 template for quality control metrics
mni_t1 = ants.image_read('/csc/epitkane/home/atagmazi/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz')

# %%
## EXPLORATION CODE: Display metadata columns (not needed for pipeline)
# meta.columns

# %%
## ============================================================================
## SECTION 3: Attach Registered Scan Paths
## ============================================================================
# Match Image.Data.ID and MRI_ID to registered scan files
# Creates PATH_registered and MRI_PATH_registered columns
for i in range(0,np.shape(meta)[0]):
    name = meta['Image.Data.ID'][i]+'_registered.nii'
    filename = fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_pet/'), name)
    if len(filename)> 0:
        meta.loc[i,'PATH_registered'] = '/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_pet/' + filename[0]
    
    name_mri = meta['MRI_ID'][i]+'_registered.nii'
    filename = fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_mri/'), name_mri)
    if len(filename)> 0:
        meta.loc[i,'MRI_PATH_registered'] = '/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_mri/' + filename[0]
         
    #print(i)

# %%
## ============================================================================
## SECTION 4: Filter Scans with Valid Registered Paths
## ============================================================================
# Keep only rows where PATH_registered is not null
meta2 = meta[meta['PATH_registered'].notna()] #!!
meta2.reset_index(drop=True, inplace = True)

# %%
## UNUSED CODE: Sequential quality control (commented out, using parallel version instead)
# Old sequential implementation - replaced by parallel version in Cell 9
'''meta2['MI'] = np.nan
meta2['correlation'] = np.nan
meta2['MSE'] = np.nan
#for i in range(150,152):
for i in range(meta2.shape[0]):
    pet = ants.image_read(meta2['PATH_registered'][i])
    if 0 in pet.spacing:
        print(f"Skipping image {i}, invalid spacing: {pet.spacing}")
        continue
    try:
        meta2['MI'][i] = ants.image_mutual_information(mni_t1, pet)
        meta2['correlation'][i] = ants.image_similarity(mni_t1, pet, metric_type="Correlation")
        meta2['MSE'][i] = ants.image_similarity(mni_t1, pet, metric_type="MeanSquares")
    except RuntimeError as e:
        print(f"Skipping {i}: {e}")
        

    print(i)'''
    

# %%
## ============================================================================
## SECTION 5: Parallel Quality Control Computation
## ============================================================================
## SECTION 5.1: Import Additional Libraries for Parallel Processing
# Import libraries for parallel quality control computation
# Note: ants, numpy, pandas already imported in Cell 1
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Assuming mni_t1 is already loaded
# mni_t1 = ants.image_read('/path/to/mni_t1.nii.gz')


def compute_metrics(i):
    print(f"Processing index: {i}")
    
    # Initialize default return values
    pet_mi = pet_corr = pet_mse = np.nan
    mri_mi = mri_corr = mri_mse = np.nan
    error_messages = []

    try:
        pet = ants.image_read(meta2.loc[i, 'PATH_registered'])
        if 0 in pet.spacing:
            error_messages.append(f"Invalid PET spacing: {pet.spacing}")
        else:
            pet_mi = ants.image_mutual_information(mni_t1, pet)
            pet_corr = ants.image_similarity(mni_t1, pet, metric_type="Correlation")
            pet_mse = ants.image_similarity(mni_t1, pet, metric_type="MeanSquares")
    except Exception as e:
        error_messages.append(f"PET error: {type(e).__name__}: {str(e)}")

    try:
        mri = ants.image_read(meta2.loc[i, 'MRI_PATH_registered'])
        if 0 in mri.spacing:
            error_messages.append(f"Invalid MRI spacing: {mri.spacing}")
        else:
            mri_mi = ants.image_mutual_information(mni_t1, mri)
            mri_corr = ants.image_similarity(mni_t1, mri, metric_type="Correlation")
            mri_mse = ants.image_similarity(mni_t1, mri, metric_type="MeanSquares")
    except Exception as e:
        error_messages.append(f"MRI error: {type(e).__name__}: {str(e)}")

    # Join error messages if any
    error_msg = "; ".join(error_messages) if error_messages else None

    return i, pet_mi, pet_corr, pet_mse, mri_mi, mri_corr, mri_mse, error_msg


# Initialize columns
meta2['pet_MI'] = np.nan
meta2['pet_correlation'] = np.nan
meta2['pet_MSE'] = np.nan

meta2['mri_MI'] = np.nan
meta2['mri_correlation'] = np.nan
meta2['mri_MSE'] = np.nan
# Run in parallel
with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(compute_metrics, range(meta2.shape[0])))

# Fill results into the DataFrame
for i, pet_mi, pet_corr, pet_mse, mri_mi, mri_corr, mri_mse, error in results:
    if error:
        print(f"Skipping {i}: {error}")
    else:
        meta2.at[i, 'pet_MI'] = pet_mi
        meta2.at[i, 'pet_correlation'] = pet_corr
        meta2.at[i, 'pet_MSE'] = pet_mse
        
        meta2.at[i, 'mri_MI'] = mri_mi
        meta2.at[i, 'mri_correlation'] = mri_corr
        meta2.at[i, 'mri_MSE'] = mri_mse
        
print('ready')


# %%
#meta2.to_csv('metafile_ADDLpipeline_abeta_mri_02_06_2024_regQC.csv', index=False)
meta2.to_csv('metafile_ADDLpipeline_abeta_mri_27_05_2025_regQC.csv', index=False)

# %%


# %% [markdown]
# ## OPTIONAL: Post-QC exploration
# # Post-analysis and visualization cells below are optional; commented for cleanliness.

# %% [markdown]
# 

# %%
## OPTIONAL: Reload saved QC file (not needed for pipeline)
# meta2 = pd.read_csv('/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/registration/metafile_ADDLpipeline_abeta_mri_27_05_2025_regQC.csv')

# %%
## VISUALIZATION CODE: Scatter plot of correlations (optional)
# plt.scatter(meta2.pet_correlation, meta2.mri_correlation)

# %%
## EXPLORATION CODE: Max PET correlation (optional)
# meta2.pet_correlation.max()

# %%
## EXPLORATION CODE: Max MRI correlation (optional)
# meta2.mri_correlation.max()

# %%
## OPTIONAL FILTERING: Select high-confidence registrations (mri_correlation >= 0.6)
# certain_meta2 = meta2[abs(meta2['mri_correlation'])>= 0.6]
# certain_meta2.reset_index(drop=True, inplace = True)

# %%
## OPTIONAL: Save filtered high-confidence set (not needed for pipeline)
# certain_meta2.to_csv('metafile_ADDLpipeline_abeta_mri_27_05_2025_afterQC.csv', index=False)

# %%
## OPTIONAL FILTERING: Select uncertain registrations (mri_correlation > 0.3)
# unsertain_meta2 = meta2[abs(meta2['mri_correlation'])> 0.3]
# #unsertain_meta2 = unsertain_meta2[abs(unsertain_meta2['correlation'])< 0.6]
# unsertain_meta2.reset_index(drop=True, inplace = True)

# %%
## VISUALIZATION CODE: Histogram of correlation (optional)
# plt.hist(meta2['correlation'], bins=20, color='skyblue', edgecolor='black')

# %%
## VISUALIZATION CODE: Scatter correlation vs MI (optional)
# ax = sns.scatterplot(x="correlation", y="MI", data=certain_meta2)

# %%
## VISUALIZATION CODE: Load example scans for overlay (optional)
# pet = ants.image_read(certain_meta2['PATH_registered'][0])
# mri = ants.image_read(certain_meta2['MRI_PATH_registered'][0])
# pet = pet.numpy()
# mni_t1 = ants.image_read('/csc/epitkane/home/atagmazi/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz')
# mni_t1 = mni_t1.numpy()

# %%
## VISUALIZATION CODE: MRI slice (optional)
# plt.imshow(mri[70, :, :].copy(), cmap = 'jet',alpha = 0.5)

# %%
## VISUALIZATION CODE: MRI slice (optional)
# plt.imshow(mri[70, :, :].copy(), cmap = 'jet',alpha = 0.5)

# %%
## VISUALIZATION CODE: Overlay MRI on template (optional)
# plt.imshow(np.rot90(mni_t1[70, :, :].copy()), cmap = 'gray')
# plt.imshow(np.rot90(mri[70, :, :].copy()), cmap = 'jet',alpha = 0.5) 
# plt.axis('off')

# %%
## VISUALIZATION CODE: Uncertain PET slice (optional)
# pet = ants.image_read(unsertain_meta2['PATH'][2])
# plt.imshow(np.rot90(pet[100, :, :].copy()), cmap = 'jet') 
# plt.axis('off')

# %%
## VISUALIZATION CODE: Load uncertain scans for overlay (optional)
# pet = ants.image_read(unsertain_meta2['PATH_registered'][0])
# mri = ants.image_read(unsertain_meta2['MRI_PATH_registered'][0])
# pet = pet.numpy()
# mni_t1 = ants.image_read('/csc/epitkane/home/atagmazi/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz')
# mni_t1 = mni_t1.numpy()

# %%
## VISUALIZATION CODE: Overlay MRI on template (optional)
# plt.imshow(np.rot90(mni_t1[100, :, :].copy()), cmap = 'gray')
# plt.imshow(np.rot90(mri[100, :, :].copy()), cmap = 'jet',alpha = 0.5) 
# plt.axis('off')

# %%



